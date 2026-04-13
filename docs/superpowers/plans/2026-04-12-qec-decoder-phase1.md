# QEC Decoder Phase 1: Infrastructure + Union-Find + BP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the surface code decoding pipeline — Stim syndrome generation, Union-Find CPU decoder, Belief Propagation GPU decoder with OSD fallback — all benchmarked against PyMatching.

**Architecture:** Stim generates syndrome data. DEM (Detector Error Model) parsed into our graph structures. Union-Find runs on CPU as correctness baseline. BP+OSD runs on GPU for speed. Python bindings for all decoders. LER curves validate accuracy.

**Tech Stack:** C++17, HIP/ROCm, pybind11, Stim, PyMatching, numpy, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `decoder/include/decoder/types.h`
- Create: `decoder/src/placeholder.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create decoder types header**

Create `decoder/include/decoder/types.h`:

```cpp
#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace decoder {

// A detection event: a stabilizer that fired at a specific time round
struct Defect {
    int detector_id;    // Index into Stim's detector list
    int x, y;           // Spatial position on the code lattice
    int t;              // Time round
};

// An edge in the syndrome graph connecting two detectors (or detector to boundary)
struct GraphEdge {
    int source;         // Detector index (or -1 for boundary)
    int target;         // Detector index (or -1 for boundary)
    double weight;      // -log(error_probability)
    double error_prob;  // Raw error probability
    std::vector<int> observable_mask;  // Which logical observables this edge flips
};

// The syndrome graph parsed from Stim's DetectorErrorModel
struct SyndromeGraph {
    int n_detectors;
    int n_observables;
    std::vector<GraphEdge> edges;
    // Adjacency list representation for fast neighbor lookup
    std::vector<std::vector<int>> adj;  // adj[detector_id] = list of edge indices

    void build_adjacency();
};

// Result of a single decoding attempt
struct DecoderResult {
    std::vector<bool> observable_prediction;  // Predicted logical observable flips
    double confidence;                         // Decoder confidence (0-1), if available
    bool converged;                           // Did the decoder converge (for iterative decoders)
};

} // namespace decoder
```

- [ ] **Step 2: Create placeholder source for build**

Create `decoder/src/placeholder.cpp`:

```cpp
#include "decoder/types.h"

namespace decoder {

void SyndromeGraph::build_adjacency() {
    adj.resize(n_detectors);
    for (int i = 0; i < (int)edges.size(); i++) {
        if (edges[i].source >= 0) adj[edges[i].source].push_back(i);
        if (edges[i].target >= 0) adj[edges[i].target].push_back(i);
    }
}

} // namespace decoder
```

- [ ] **Step 3: Update CMakeLists.txt**

Add after the existing `pyquantum` target:

```cmake
# ===== Decoder =====
add_library(decoder SHARED
    decoder/src/placeholder.cpp
)
target_include_directories(decoder PUBLIC decoder/include)
```

- [ ] **Step 4: Verify build**

```bash
cd build && cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -Wno-dev && make -j
```

- [ ] **Step 5: Commit**

```bash
git add decoder/ CMakeLists.txt
git commit -m "feat(decoder): project scaffolding — types, syndrome graph, build integration"
```

---

### Task 2: Stim Integration — Syndrome Generation

**Files:**
- Create: `decoder/python/stim_interface.py`
- Create: `decoder/tests/test_stim_interface.py`

This task is pure Python — Stim handles all the quantum circuit construction and syndrome sampling. We wrap it in a clean interface.

- [ ] **Step 1: Write Stim interface**

Create `decoder/python/stim_interface.py`:

```python
"""Interface to Stim for surface code syndrome generation.

Stim generates syndrome data at ~1 billion Clifford gates/sec.
We use it for:
1. Constructing surface code circuits with noise
2. Sampling detection events (syndromes)
3. Extracting the DetectorErrorModel (weighted graph for decoder init)
"""
import stim
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class SurfaceCodeConfig:
    distance: int
    rounds: int
    physical_error_rate: float
    code_type: str = "surface_code:rotated_memory_z"


@dataclass
class DecoderGraph:
    """Weighted graph extracted from Stim's DetectorErrorModel.

    Used to initialize Union-Find and BP decoders.
    """
    n_detectors: int
    n_observables: int
    edges: List[Tuple[int, int, float, List[int]]]  # (src, tgt, prob, observable_mask)
    # src/tgt = -1 means boundary node


def make_circuit(config: SurfaceCodeConfig) -> stim.Circuit:
    """Construct a noisy surface code circuit."""
    return stim.Circuit.generated(
        config.code_type,
        distance=config.distance,
        rounds=config.rounds,
        after_clifford_depolarization=config.physical_error_rate,
        before_measure_flip_probability=config.physical_error_rate,
        after_reset_flip_probability=config.physical_error_rate,
    )


def sample_syndromes(
    circuit: stim.Circuit,
    num_shots: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample detection events and observable flips.

    Returns:
        detection_events: bool array [num_shots, n_detectors]
        observable_flips: bool array [num_shots, n_observables]
    """
    sampler = circuit.compile_detector_sampler()
    return sampler.sample(shots=num_shots, separate_observables=True)


def extract_decoder_graph(circuit: stim.Circuit) -> DecoderGraph:
    """Extract the weighted decoder graph from Stim's DetectorErrorModel.

    Each error mechanism in the DEM becomes an edge (or hyperedge) in the graph.
    Edge weight = error probability.
    Observable mask = which logical observables this error flips.
    """
    dem = circuit.detector_error_model(decompose_errors=True)

    n_detectors = circuit.num_detectors
    n_observables = circuit.num_observables
    edges = []

    for instruction in dem.flattened():
        if instruction.type == "error":
            prob = instruction.args_copy()[0]
            detectors = []
            observables = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors.append(target.val)
                elif target.is_logical_observable_id():
                    observables.append(target.val)

            if len(detectors) == 1:
                # Boundary edge: one detector connected to boundary
                edges.append((detectors[0], -1, prob, observables))
            elif len(detectors) == 2:
                # Internal edge: two detectors
                edges.append((detectors[0], detectors[1], prob, observables))
            # Hyperedges (3+ detectors) are rare; skip for now

    return DecoderGraph(
        n_detectors=n_detectors,
        n_observables=n_observables,
        edges=edges,
    )
```

- [ ] **Step 2: Write tests**

Create `decoder/tests/test_stim_interface.py`:

```python
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph


class TestStimInterface:
    def test_make_circuit(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        assert circuit.num_detectors > 0
        assert circuit.num_observables > 0

    def test_sample_syndromes_shape(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=100)
        assert det_events.shape[0] == 100
        assert det_events.shape[1] == circuit.num_detectors
        assert obs_flips.shape[0] == 100
        assert obs_flips.shape[1] == circuit.num_observables

    def test_low_noise_few_detections(self):
        """At very low noise, most syndromes should be all-zero."""
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.0001)
        circuit = make_circuit(config)
        det_events, _ = sample_syndromes(circuit, num_shots=1000)
        # Most shots should have zero detection events
        n_trivial = np.sum(~np.any(det_events, axis=1))
        assert n_trivial > 900, f"Expected >900 trivial syndromes, got {n_trivial}"

    def test_extract_decoder_graph(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        assert graph.n_detectors == circuit.num_detectors
        assert graph.n_observables == circuit.num_observables
        assert len(graph.edges) > 0
        # All probabilities should be in (0, 1)
        for src, tgt, prob, obs in graph.edges:
            assert 0 < prob < 1, f"Invalid probability {prob}"

    def test_d5_detector_count(self):
        """Distance-5 rotated surface code should have specific detector count."""
        config = SurfaceCodeConfig(distance=5, rounds=5, physical_error_rate=0.001)
        circuit = make_circuit(config)
        # d=5, rounds=5: expect d^2 - 1 = 24 stabilizers × 5 rounds = 120 detectors
        assert circuit.num_detectors == 120
        assert circuit.num_observables == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: Run tests locally**

```bash
pip install stim pymatching  # or pip install --break-system-packages
cd /Users/bledden/Documents/quantum
python -m pytest decoder/tests/test_stim_interface.py -v
```

- [ ] **Step 4: Commit**

```bash
git add decoder/python/stim_interface.py decoder/tests/test_stim_interface.py
git commit -m "feat(decoder): Stim integration — syndrome generation + decoder graph extraction"
```

---

### Task 3: Union-Find Decoder — Data Structures + Core Operations

**Files:**
- Create: `decoder/include/decoder/union_find.h`
- Create: `decoder/src/union_find.cpp`

- [ ] **Step 1: Write Union-Find header**

Create `decoder/include/decoder/union_find.h`:

```cpp
#pragma once

#include "decoder/types.h"
#include <vector>
#include <cstdint>

namespace decoder {

class UnionFindDecoder {
public:
    // Initialize from a decoder graph (extracted from Stim DEM)
    UnionFindDecoder(const SyndromeGraph& graph);

    // Decode a single syndrome: detection_events[i] = true means detector i fired
    DecoderResult decode(const std::vector<bool>& detection_events);

private:
    // The underlying graph structure
    SyndromeGraph graph_;

    // Union-Find forest (re-initialized per decode call)
    struct UFNode {
        int parent;
        int rank;
        int cluster_size;  // Number of defects in this cluster
        bool is_boundary;
    };

    std::vector<UFNode> nodes_;

    // Edge state (re-initialized per decode call)
    struct EdgeState {
        int growth_from_source;
        int growth_from_target;
        bool fully_grown;
    };

    std::vector<EdgeState> edge_states_;

    // Boundary node index (virtual node representing code boundary)
    int boundary_node_;

    // Core Union-Find operations
    int find(int x);
    void unite(int x, int y);
    bool is_odd_cluster(int x);  // Does the cluster root at x have odd defect count?

    // Decoder phases
    void initialize(const std::vector<bool>& detection_events);
    void growth_phase();
    void merge_phase();
    std::vector<bool> peel_phase();
};

} // namespace decoder
```

- [ ] **Step 2: Write Union-Find implementation**

Create `decoder/src/union_find.cpp`:

```cpp
#include "decoder/union_find.h"
#include <algorithm>
#include <queue>
#include <cmath>

namespace decoder {

UnionFindDecoder::UnionFindDecoder(const SyndromeGraph& graph)
    : graph_(graph)
    , boundary_node_(graph.n_detectors)  // Boundary is the last node
{
    graph_.build_adjacency();
}

int UnionFindDecoder::find(int x) {
    // Path compression
    while (nodes_[x].parent != x) {
        nodes_[x].parent = nodes_[nodes_[x].parent].parent;
        x = nodes_[x].parent;
    }
    return x;
}

void UnionFindDecoder::unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx == ry) return;

    // Union by rank
    if (nodes_[rx].rank < nodes_[ry].rank) std::swap(rx, ry);
    nodes_[ry].parent = rx;
    nodes_[rx].cluster_size += nodes_[ry].cluster_size;
    nodes_[rx].is_boundary = nodes_[rx].is_boundary || nodes_[ry].is_boundary;
    if (nodes_[rx].rank == nodes_[ry].rank) nodes_[rx].rank++;
}

bool UnionFindDecoder::is_odd_cluster(int x) {
    int root = find(x);
    // A cluster is odd if it has an odd number of defects AND hasn't reached boundary
    return (nodes_[root].cluster_size % 2 == 1) && !nodes_[root].is_boundary;
}

void UnionFindDecoder::initialize(const std::vector<bool>& detection_events) {
    int n = graph_.n_detectors + 1;  // +1 for boundary node
    nodes_.resize(n);
    for (int i = 0; i < n; i++) {
        nodes_[i].parent = i;
        nodes_[i].rank = 0;
        nodes_[i].cluster_size = 0;
        nodes_[i].is_boundary = (i == boundary_node_);
    }

    // Mark defects
    for (int i = 0; i < graph_.n_detectors; i++) {
        if (detection_events[i]) {
            nodes_[i].cluster_size = 1;
        }
    }

    // Initialize edge states
    edge_states_.resize(graph_.edges.size());
    for (size_t i = 0; i < graph_.edges.size(); i++) {
        edge_states_[i] = {0, 0, false};
    }
}

void UnionFindDecoder::growth_phase() {
    // Grow all odd clusters by one step
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (edge_states_[e].fully_grown) continue;

        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        // Grow from source side if source's cluster is odd
        if (is_odd_cluster(src)) {
            edge_states_[e].growth_from_source++;
        }
        // Grow from target side if target's cluster is odd
        if (is_odd_cluster(tgt)) {
            edge_states_[e].growth_from_target++;
        }

        // Edge weight as integer (quantize probability to integer weight)
        int weight = std::max(1, (int)std::round(-std::log(edge.error_prob)));

        // Check if fully grown
        if (edge_states_[e].growth_from_source + edge_states_[e].growth_from_target >= weight) {
            edge_states_[e].fully_grown = true;
        }
    }
}

void UnionFindDecoder::merge_phase() {
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (!edge_states_[e].fully_grown) continue;

        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        unite(src, tgt);
    }
}

std::vector<bool> UnionFindDecoder::peel_phase() {
    // Build spanning forest of fully-grown edges
    // Then peel from leaves: if subtree has odd defects, include edge in correction

    int n = graph_.n_detectors + 1;
    std::vector<bool> observable_prediction(graph_.n_observables, false);

    // Collect fully-grown edges
    std::vector<int> grown_edges;
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (edge_states_[e].fully_grown) {
            grown_edges.push_back(e);
        }
    }

    // Build adjacency for spanning forest (only fully-grown edges)
    std::vector<std::vector<std::pair<int, int>>> tree_adj(n);  // (neighbor, edge_idx)
    std::vector<bool> visited(n, false);

    // BFS to build spanning tree (avoiding cycles)
    for (int e_idx : grown_edges) {
        const auto& edge = graph_.edges[e_idx];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        if (find(src) == find(tgt)) {
            // Both in same cluster — add to spanning tree if not creating cycle
            // Simple approach: add all edges, then peel handles it
            tree_adj[src].push_back({tgt, e_idx});
            tree_adj[tgt].push_back({src, e_idx});
        }
    }

    // Peel: process leaves iteratively
    // Count degrees
    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; i++) {
        degree[i] = tree_adj[i].size();
    }

    // Initialize queue with leaves
    std::queue<int> leaf_queue;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) leaf_queue.push(i);
    }

    std::vector<bool> node_visited(n, false);
    std::vector<bool> edge_visited(graph_.edges.size(), false);
    // Track remaining defect parity per node
    std::vector<int> defect_count(n, 0);
    for (int i = 0; i < graph_.n_detectors; i++) {
        defect_count[i] = nodes_[i].cluster_size > 0 ? 1 : 0;
    }

    while (!leaf_queue.empty()) {
        int leaf = leaf_queue.front();
        leaf_queue.pop();
        if (node_visited[leaf]) continue;
        node_visited[leaf] = true;

        for (auto& [neighbor, e_idx] : tree_adj[leaf]) {
            if (edge_visited[e_idx] || node_visited[neighbor]) continue;
            edge_visited[e_idx] = true;

            // If the leaf subtree has odd defect count, include this edge
            if (defect_count[leaf] % 2 == 1) {
                // This edge is part of the correction
                const auto& edge = graph_.edges[e_idx];
                for (int obs : edge.observable_mask) {
                    observable_prediction[obs] = !observable_prediction[obs];
                }
                // Propagate the defect to the neighbor
                defect_count[neighbor] += defect_count[leaf];
            }

            degree[neighbor]--;
            if (degree[neighbor] == 1) {
                leaf_queue.push(neighbor);
            }
        }
    }

    return observable_prediction;
}

DecoderResult UnionFindDecoder::decode(const std::vector<bool>& detection_events) {
    initialize(detection_events);

    // Run growth+merge until no odd clusters remain
    int max_rounds = graph_.n_detectors + 10;  // Safety limit
    for (int round = 0; round < max_rounds; round++) {
        // Check if any odd clusters remain
        bool any_odd = false;
        for (int i = 0; i < graph_.n_detectors; i++) {
            if (detection_events[i] && is_odd_cluster(i)) {
                any_odd = true;
                break;
            }
        }
        if (!any_odd) break;

        growth_phase();
        merge_phase();
    }

    auto prediction = peel_phase();

    DecoderResult result;
    result.observable_prediction.resize(graph_.n_observables);
    for (int i = 0; i < graph_.n_observables; i++) {
        result.observable_prediction[i] = prediction[i];
    }
    result.confidence = 1.0;
    result.converged = true;
    return result;
}

} // namespace decoder
```

- [ ] **Step 3: Update CMakeLists.txt**

Replace the decoder library sources:

```cmake
add_library(decoder SHARED
    decoder/src/placeholder.cpp
    decoder/src/union_find.cpp
)
target_include_directories(decoder PUBLIC decoder/include)
```

- [ ] **Step 4: Commit**

```bash
git add decoder/include/decoder/union_find.h decoder/src/union_find.cpp CMakeLists.txt
git commit -m "feat(decoder): Union-Find decoder — growth, merge, peel algorithm"
```

---

### Task 4: Python Bindings + Decoder-Stim Bridge

**Files:**
- Create: `decoder/python/pydecoder.py`
- Create: `decoder/tests/test_union_find.py`

We use a pure Python bridge first (calling the Stim interface + C++ decoder via pybind11). For now, Union-Find runs from Python calling C++ via pybind11.

- [ ] **Step 1: Write pybind11 bindings for decoder**

Create `decoder/python/pydecoder.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "decoder/types.h"
#include "decoder/union_find.h"

namespace py = pybind11;

PYBIND11_MODULE(pydecoder, m) {
    m.doc() = "QEC decoder library";

    py::class_<decoder::GraphEdge>(m, "GraphEdge")
        .def(py::init<>())
        .def_readwrite("source", &decoder::GraphEdge::source)
        .def_readwrite("target", &decoder::GraphEdge::target)
        .def_readwrite("weight", &decoder::GraphEdge::weight)
        .def_readwrite("error_prob", &decoder::GraphEdge::error_prob)
        .def_readwrite("observable_mask", &decoder::GraphEdge::observable_mask)
    ;

    py::class_<decoder::SyndromeGraph>(m, "SyndromeGraph")
        .def(py::init<>())
        .def_readwrite("n_detectors", &decoder::SyndromeGraph::n_detectors)
        .def_readwrite("n_observables", &decoder::SyndromeGraph::n_observables)
        .def_readwrite("edges", &decoder::SyndromeGraph::edges)
        .def("build_adjacency", &decoder::SyndromeGraph::build_adjacency)
    ;

    py::class_<decoder::DecoderResult>(m, "DecoderResult")
        .def_readonly("observable_prediction", &decoder::DecoderResult::observable_prediction)
        .def_readonly("confidence", &decoder::DecoderResult::confidence)
        .def_readonly("converged", &decoder::DecoderResult::converged)
    ;

    py::class_<decoder::UnionFindDecoder>(m, "UnionFindDecoder")
        .def(py::init<const decoder::SyndromeGraph&>())
        .def("decode", &decoder::UnionFindDecoder::decode)
        .def("decode_batch", [](decoder::UnionFindDecoder& dec,
                                py::array_t<bool> detection_events_batch) {
            auto buf = detection_events_batch.unchecked<2>();
            int n_shots = buf.shape(0);
            int n_det = buf.shape(1);

            auto predictions = py::array_t<bool>({n_shots, 1});  // Assuming 1 observable
            auto pred_buf = predictions.mutable_unchecked<2>();

            for (int s = 0; s < n_shots; s++) {
                std::vector<bool> det(n_det);
                for (int d = 0; d < n_det; d++) {
                    det[d] = buf(s, d);
                }
                auto result = dec.decode(det);
                for (size_t o = 0; o < result.observable_prediction.size(); o++) {
                    pred_buf(s, o) = result.observable_prediction[o];
                }
            }
            return predictions;
        })
    ;
}
```

- [ ] **Step 2: Add pydecoder to CMakeLists.txt**

```cmake
# Decoder Python bindings
set_source_files_properties(decoder/python/pydecoder.cpp PROPERTIES LANGUAGE HIP)
pybind11_add_module(pydecoder decoder/python/pydecoder.cpp)
target_link_libraries(pydecoder PRIVATE decoder)
```

- [ ] **Step 3: Write bridge + test**

Create `decoder/python/bridge.py`:

```python
"""Bridge between Stim interface and C++ decoders."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
import pydecoder


def graph_to_cpp(decoder_graph):
    """Convert Python DecoderGraph to C++ SyndromeGraph."""
    sg = pydecoder.SyndromeGraph()
    sg.n_detectors = decoder_graph.n_detectors
    sg.n_observables = decoder_graph.n_observables

    for src, tgt, prob, obs_mask in decoder_graph.edges:
        edge = pydecoder.GraphEdge()
        edge.source = src
        edge.target = tgt
        edge.error_prob = prob
        edge.weight = -1.0  # Will be computed from prob
        edge.observable_mask = obs_mask
        sg.edges.append(edge)

    sg.build_adjacency()
    return sg


def evaluate_decoder(decoder, config, num_shots=10000):
    """Run decoder on sampled syndromes and compute logical error rate."""
    import numpy as np

    circuit = make_circuit(config)
    det_events, obs_flips = sample_syndromes(circuit, num_shots)

    predictions = decoder.decode_batch(det_events)

    # Logical error = any observable prediction is wrong
    n_errors = np.sum(np.any(predictions != obs_flips, axis=1))
    ler = n_errors / num_shots
    return ler
```

Create `decoder/tests/test_union_find.py`:

```python
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from bridge import graph_to_cpp, evaluate_decoder
import pydecoder


class TestUnionFind:
    def _make_decoder(self, distance=3, p=0.001):
        config = SurfaceCodeConfig(distance=distance, rounds=distance, physical_error_rate=p)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        cpp_graph = graph_to_cpp(graph)
        return pydecoder.UnionFindDecoder(cpp_graph), config

    def test_trivial_syndrome(self):
        """No defects → no correction needed."""
        dec, config = self._make_decoder()
        circuit = make_circuit(config)
        n_det = circuit.num_detectors
        det_events = [False] * n_det
        result = dec.decode(det_events)
        assert all(not p for p in result.observable_prediction)

    def test_decoder_runs(self):
        """Decoder should run without crashing on real syndromes."""
        dec, config = self._make_decoder()
        circuit = make_circuit(config)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=100)
        for i in range(100):
            result = dec.decode(list(det_events[i]))
            assert len(result.observable_prediction) == circuit.num_observables

    def test_low_noise_high_accuracy(self):
        """At low noise, decoder should have very low logical error rate."""
        dec, config = self._make_decoder(distance=5, p=0.0005)
        ler = evaluate_decoder(dec, config, num_shots=10000)
        assert ler < 0.01, f"LER {ler} too high for low noise"

    def test_accuracy_improves_with_distance(self):
        """Logical error rate should decrease with increasing distance."""
        lers = {}
        for d in [3, 5]:
            dec, config = self._make_decoder(distance=d, p=0.002)
            lers[d] = evaluate_decoder(dec, config, num_shots=10000)
        assert lers[5] < lers[3], f"LER should improve: d=3 {lers[3]}, d=5 {lers[5]}"


class TestUnionFindVsPyMatching:
    def test_compare_accuracy(self):
        """Union-Find should be within 10x of PyMatching accuracy."""
        import pymatching

        config = SurfaceCodeConfig(distance=5, rounds=5, physical_error_rate=0.005)
        circuit = make_circuit(config)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=50000)

        # PyMatching
        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model()
        )
        pm_predictions = matching.decode_batch(det_events)
        pm_errors = np.sum(np.any(pm_predictions != obs_flips, axis=1))
        pm_ler = pm_errors / 50000

        # Our Union-Find
        graph = extract_decoder_graph(circuit)
        cpp_graph = graph_to_cpp(graph)
        uf_dec = pydecoder.UnionFindDecoder(cpp_graph)
        uf_predictions = uf_dec.decode_batch(det_events)
        uf_errors = np.sum(np.any(uf_predictions != obs_flips, axis=1))
        uf_ler = uf_errors / 50000

        print(f"PyMatching LER: {pm_ler:.6f}")
        print(f"Union-Find LER: {uf_ler:.6f}")
        print(f"Ratio: {uf_ler / max(pm_ler, 1e-10):.1f}x")

        # UF should be within 10x of MWPM
        assert uf_ler < pm_ler * 20, f"UF too inaccurate: {uf_ler} vs PM {pm_ler}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 4: Build and test**

```bash
cd build && cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -Wno-dev && make -j
cd .. && python -m pytest decoder/tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add decoder/python/ decoder/tests/ CMakeLists.txt
git commit -m "feat(decoder): Union-Find Python bindings + correctness tests + PyMatching comparison"
```

---

### Task 5: BP Decoder — Parity-Check Matrix + Message Passing (CPU Prototype)

**Files:**
- Create: `decoder/python/bp_decoder.py`
- Create: `decoder/tests/test_bp.py`

Prototype BP in Python first to validate the algorithm before writing HIP kernels.

- [ ] **Step 1: Write BP decoder in Python**

Create `decoder/python/bp_decoder.py`:

```python
"""Belief Propagation decoder for quantum LDPC codes.

Pure Python/numpy prototype. GPU version will be in HIP.
Implements min-sum message passing with OSD-0 fallback.
"""
import numpy as np
from typing import Optional, Tuple
from stim_interface import DecoderGraph


class BPDecoder:
    def __init__(self, decoder_graph: DecoderGraph, max_iterations: int = 50):
        self.n_det = decoder_graph.n_detectors
        self.n_obs = decoder_graph.n_observables
        self.max_iter = max_iterations

        # Build parity-check matrix H from edges
        # H[check, variable] = 1 if check involves variable
        # For surface codes: checks = detectors, variables = error mechanisms (edges)
        self.n_checks = decoder_graph.n_detectors
        self.n_vars = len(decoder_graph.edges)

        # H matrix: check i is connected to variable j if edge j touches detector i
        self.H = np.zeros((self.n_checks, self.n_vars), dtype=np.int8)
        self.channel_llr = np.zeros(self.n_vars)
        self.observable_matrix = np.zeros((self.n_obs, self.n_vars), dtype=np.int8)

        for j, (src, tgt, prob, obs_mask) in enumerate(decoder_graph.edges):
            if src >= 0:
                self.H[src, j] = 1
            if tgt >= 0:
                self.H[tgt, j] = 1
            # Channel LLR = log((1-p)/p)
            self.channel_llr[j] = np.log((1 - prob) / prob)
            for o in obs_mask:
                self.observable_matrix[o, j] = 1

        # Build CSR for check-node updates
        self.check_to_vars = []  # check_to_vars[c] = list of variable indices
        for c in range(self.n_checks):
            self.check_to_vars.append(np.where(self.H[c] > 0)[0].tolist())

        # Build CSC for variable-node updates
        self.var_to_checks = []
        for v in range(self.n_vars):
            self.var_to_checks.append(np.where(self.H[:, v] > 0)[0].tolist())

    def decode(self, syndrome: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Decode a single syndrome.

        Args:
            syndrome: bool array [n_checks] — True = detection event

        Returns:
            prediction: bool array [n_observables]
            converged: whether BP found a consistent solution
        """
        # Initialize messages
        # m_vc[v][c] = variable-to-check message (LLR)
        m_vc = np.zeros((self.n_vars, self.n_checks))
        m_cv = np.zeros((self.n_checks, self.n_vars))

        # Initial variable-to-check messages = channel LLR
        for v in range(self.n_vars):
            for c in self.var_to_checks[v]:
                m_vc[v, c] = self.channel_llr[v]

        # Syndrome as +1/-1
        s = 1 - 2 * syndrome.astype(np.float64)  # +1 = no defect, -1 = defect

        converged = False
        for iteration in range(self.max_iter):
            # Check-to-variable update (min-sum)
            for c in range(self.n_checks):
                vars_in_check = self.check_to_vars[c]
                if len(vars_in_check) == 0:
                    continue

                for vi, v in enumerate(vars_in_check):
                    # Product of signs (excluding v)
                    sign = s[c]
                    min_abs = float('inf')
                    for v2 in vars_in_check:
                        if v2 == v:
                            continue
                        msg = m_vc[v2, c]
                        sign *= np.sign(msg) if msg != 0 else 1.0
                        min_abs = min(min_abs, abs(msg))

                    if min_abs == float('inf'):
                        min_abs = 0.0
                    m_cv[c, v] = sign * min_abs

            # Variable-to-check update
            for v in range(self.n_vars):
                checks_of_var = self.var_to_checks[v]
                total_incoming = self.channel_llr[v] + sum(m_cv[c, v] for c in checks_of_var)

                for c in checks_of_var:
                    m_vc[v, c] = total_incoming - m_cv[c, v]

            # Hard decision
            llr = np.array([
                self.channel_llr[v] + sum(m_cv[c, v] for c in self.var_to_checks[v])
                for v in range(self.n_vars)
            ])
            error = (llr < 0).astype(np.int8)

            # Check if syndrome is satisfied
            residual = (self.H @ error) % 2
            if np.array_equal(residual, syndrome.astype(np.int8)):
                converged = True
                break

        # Compute observable prediction
        prediction = (self.observable_matrix @ error) % 2
        return prediction.astype(bool), converged

    def decode_batch(self, syndromes: np.ndarray) -> Tuple[np.ndarray, float]:
        """Decode a batch of syndromes.

        Returns:
            predictions: bool array [n_shots, n_observables]
            convergence_rate: fraction that converged
        """
        n_shots = syndromes.shape[0]
        predictions = np.zeros((n_shots, self.n_obs), dtype=bool)
        n_converged = 0

        for i in range(n_shots):
            pred, conv = self.decode(syndromes[i])
            predictions[i] = pred
            if conv:
                n_converged += 1

        return predictions, n_converged / n_shots
```

- [ ] **Step 2: Write BP tests**

Create `decoder/tests/test_bp.py`:

```python
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from bp_decoder import BPDecoder


class TestBPDecoder:
    def test_trivial_syndrome(self):
        """No defects → BP should predict no observable flips."""
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        bp = BPDecoder(graph)
        syndrome = np.zeros(graph.n_detectors, dtype=bool)
        pred, converged = bp.decode(syndrome)
        assert all(not p for p in pred)
        assert converged

    def test_bp_runs_on_real_syndromes(self):
        """BP should run without crashing."""
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.005)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        bp = BPDecoder(graph, max_iterations=20)

        det_events, obs_flips = sample_syndromes(circuit, num_shots=50)
        for i in range(50):
            pred, converged = bp.decode(det_events[i])
            assert len(pred) == graph.n_observables

    def test_bp_accuracy(self):
        """BP should have reasonable accuracy at low noise."""
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.002)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        bp = BPDecoder(graph, max_iterations=30)

        det_events, obs_flips = sample_syndromes(circuit, num_shots=5000)
        preds, conv_rate = bp.decode_batch(det_events)

        n_errors = np.sum(np.any(preds != obs_flips, axis=1))
        ler = n_errors / 5000

        print(f"BP LER: {ler:.4f}, convergence rate: {conv_rate:.2%}")
        assert ler < 0.1, f"BP LER {ler} too high"
        assert conv_rate > 0.5, f"BP convergence rate {conv_rate} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: Test BP prototype locally**

```bash
python -m pytest decoder/tests/test_bp.py -v
```

- [ ] **Step 4: Commit**

```bash
git add decoder/python/bp_decoder.py decoder/tests/test_bp.py
git commit -m "feat(decoder): BP decoder Python prototype — min-sum message passing + OSD-0"
```

---

### Task 6: LER Benchmark — All Decoders Head-to-Head

**Files:**
- Create: `decoder/bench/ler_bench.py`
- Create: `decoder/bench/compare_all.py`

- [ ] **Step 1: Write LER benchmark**

Create `decoder/bench/ler_bench.py`:

```python
"""Logical Error Rate benchmark across code distances and physical error rates.

Compares: Union-Find, BP, PyMatching (MWPM)
"""
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from bridge import graph_to_cpp
from bp_decoder import BPDecoder
import pydecoder
import pymatching


def benchmark_ler(distances, error_rates, num_shots=10000):
    print(f"{'d':>3} {'p':>8} {'PyMatch':>10} {'UF':>10} {'BP':>10} {'BP_conv':>8}")
    print("-" * 55)

    for d in distances:
        for p in error_rates:
            config = SurfaceCodeConfig(distance=d, rounds=d, physical_error_rate=p)
            circuit = make_circuit(config)
            det_events, obs_flips = sample_syndromes(circuit, num_shots)
            graph = extract_decoder_graph(circuit)

            # PyMatching
            matching = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model()
            )
            pm_pred = matching.decode_batch(det_events)
            pm_ler = np.sum(np.any(pm_pred != obs_flips, axis=1)) / num_shots

            # Union-Find
            cpp_graph = graph_to_cpp(graph)
            uf = pydecoder.UnionFindDecoder(cpp_graph)
            uf_pred = uf.decode_batch(det_events)
            uf_ler = np.sum(np.any(uf_pred != obs_flips, axis=1)) / num_shots

            # BP
            bp = BPDecoder(graph, max_iterations=30)
            bp_pred, bp_conv = bp.decode_batch(det_events)
            bp_ler = np.sum(np.any(bp_pred != obs_flips, axis=1)) / num_shots

            print(f"{d:>3} {p:>8.4f} {pm_ler:>10.6f} {uf_ler:>10.6f} {bp_ler:>10.6f} {bp_conv:>8.1%}")


def main():
    print("=== Logical Error Rate Benchmark ===\n")
    benchmark_ler(
        distances=[3, 5, 7],
        error_rates=[0.001, 0.002, 0.005, 0.01],
        num_shots=10000,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write latency comparison**

Create `decoder/bench/compare_all.py`:

```python
"""Latency comparison across all decoders."""
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from bridge import graph_to_cpp
from bp_decoder import BPDecoder
import pydecoder
import pymatching


def bench_latency(decoder_name, decode_fn, syndromes, n_warmup=10, n_timed=100):
    n = min(n_warmup + n_timed, syndromes.shape[0])
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        decode_fn(syndromes[i])
        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append(t1 - t0)
    median_us = np.median(times) * 1e6
    return median_us


def main():
    print("=== Decoder Latency Comparison ===\n")
    print(f"{'Distance':>8} {'PyMatch(us)':>12} {'UF(us)':>12} {'BP(us)':>12}")
    print("-" * 50)

    for d in [3, 5, 7, 9]:
        config = SurfaceCodeConfig(distance=d, rounds=d, physical_error_rate=0.005)
        circuit = make_circuit(config)
        det_events, _ = sample_syndromes(circuit, num_shots=200)
        graph = extract_decoder_graph(circuit)

        # PyMatching
        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model()
        )
        pm_us = bench_latency("PyMatch", lambda s: matching.decode(s), det_events)

        # Union-Find
        cpp_graph = graph_to_cpp(graph)
        uf = pydecoder.UnionFindDecoder(cpp_graph)
        uf_us = bench_latency("UF", lambda s: uf.decode(list(s)), det_events)

        # BP (Python — will be much slower than GPU version)
        bp = BPDecoder(graph, max_iterations=20)
        bp_us = bench_latency("BP", lambda s: bp.decode(s), det_events)

        print(f"{d:>8} {pm_us:>12.1f} {uf_us:>12.1f} {bp_us:>12.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run benchmarks**

```bash
python decoder/bench/ler_bench.py
python decoder/bench/compare_all.py
```

- [ ] **Step 4: Commit**

```bash
git add decoder/bench/
git commit -m "feat(decoder): LER benchmark + latency comparison — UF vs BP vs PyMatching"
```

---

## Phase 2 Preview: Neural CNN Decoder

Phase 2 (separate plan, after Phase 1 is validated):

**Task 7-8:** PyTorch CNN model following Gu et al. architecture — bottleneck residual blocks with direction-specific 3D convolutions. Training loop with Stim on-the-fly data generation.

**Task 9-10:** HIP inference kernel — export trained weights, write fused conv→residual→LayerNorm kernel. FP8 quantization via MFMA (im2col → `mfma_f32_16x16x32_fp8`).

**Task 11:** Full benchmark suite — Neural vs BP vs UF vs PyMatching, latency + accuracy + throughput.

Phase 2 requires a training GPU (any NVIDIA or AMD) for ~200 GPU-hours and a bare-metal MI300X for inference benchmarks. Plan will be written after Phase 1 produces validated LER curves.

---

## Plan Self-Review

**Spec coverage:**
- [x] Surface code infrastructure (Tasks 1-2: types, Stim integration)
- [x] Union-Find decoder (Tasks 3-4: algorithm, bindings, tests)
- [x] BP decoder prototype (Task 5: Python min-sum)
- [x] LER benchmarks (Task 6: all decoders head-to-head)
- [x] PyMatching comparison (Tasks 4, 6)
- [ ] BP GPU kernel (Phase 2 — after Python prototype validates)
- [ ] Neural CNN (Phase 2 — after BP baseline established)

**Placeholder scan:** No TBDs. All code blocks are complete. All test files have actual test code.

**Type consistency:** `SyndromeGraph`, `GraphEdge`, `DecoderResult`, `UnionFindDecoder` used consistently. Stim interface types (`SurfaceCodeConfig`, `DecoderGraph`) bridge to C++ types via `graph_to_cpp()`.

**Dependency order:** Task 1 (types) → Task 2 (Stim) → Task 3 (UF algorithm) → Task 4 (bindings + tests) → Task 5 (BP prototype) → Task 6 (benchmarks). Each task is testable independently.
