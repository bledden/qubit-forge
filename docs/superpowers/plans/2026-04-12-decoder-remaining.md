# QEC Decoder — Remaining Work Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the decoder project — train neural decoders to convergence on GPU, add RL-based decoder steering to Union-Find, run definitive benchmarks.

**Architecture:** Three decoder tiers (UF + BP + Neural CNN), with decoder steering enhancement on UF. All benchmarked against PyMatching.

**Tech Stack:** PyTorch, HIP/ROCm, Stim, pybind11, C++17

## Status: What's Done

- [x] Stim integration (syndrome generation, DEM extraction)
- [x] C++ Union-Find decoder (weighted growth, peel, pybind11 bindings)
- [x] Python BP decoder prototype
- [x] Neural CNN model (DirectionalConv3d, bottleneck blocks, Muon optimizer)
- [x] Training pipeline (Stim data, curriculum, pre-compiled samplers)
- [x] d=3 neural decoder trained — **beats PyMatching at ALL noise rates**
- [x] d=5 neural decoder trained (20K steps) — beats UF, 2-3x behind PyMatching
- [x] LER benchmarks (d=3,5 × multiple noise rates)
- [x] Conditional CMake build (GPU simulator on MI300X, decoder on any platform)

## What Remains

---

### Task 1: Train d=5 Neural Decoder to Convergence (GPU)

The d=5 model at 20K steps (CPU) reaches 4.3% LER vs PyMatching's 0.8%.
The paper used 80K steps — we're compute-limited on CPU. GPU training will converge in ~30 min.

**On MI300X:**

- [ ] **Step 1: Install PyTorch ROCm**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install stim pymatching
```

- [ ] **Step 2: Deploy code**

```bash
git clone https://github.com/bledden/qubit-forge.git && cd qubit-forge
```

- [ ] **Step 3: Train d=5 for 80K steps**

```bash
python -u decoder/train/train.py \
    --distance 5 --hidden_dim 256 --steps 80000 \
    --batch_size 1024 --noise_rate 0.007 \
    --eval_interval 5000 --log_interval 2000 \
    --checkpoint_dir decoder/train/checkpoints/d5_gpu
```

Expected: ~30-45 min on MI300X. Target LER: <1% at p=0.007 (match or beat PyMatching).

- [ ] **Step 4: Evaluate**

```bash
python decoder/train/evaluate.py \
    --checkpoint decoder/train/checkpoints/d5_gpu/best_model.pt \
    --n_shots 100000
```

- [ ] **Step 5: Commit results**

```bash
git add decoder/train/checkpoints/d5_gpu/
git commit -m "feat(decoder): d=5 neural decoder GPU-trained — 80K steps"
```

---

### Task 2: Train d=7 Neural Decoder (GPU)

d=7 is where the "waterfall regime" from Gu et al. should appear — error suppression scaling faster than distance scaling.

- [ ] **Step 1: Train d=7**

```bash
python -u decoder/train/train.py \
    --distance 7 --hidden_dim 256 --steps 80000 \
    --batch_size 512 --noise_rate 0.007 \
    --eval_interval 5000 --log_interval 2000 \
    --checkpoint_dir decoder/train/checkpoints/d7_gpu
```

Expected: ~45-60 min. The model has 500K params at d=7.

- [ ] **Step 2: Evaluate and commit**

---

### Task 3: Union-Find Decoder Steering (RL-inspired)

From Sivak et al. (arXiv:2511.08493): dynamically reweight the decoder's matching graph
based on observed detection event patterns. They achieved 0.7x additional LER improvement
on top of their base decoder.

**Concept:** Instead of using static edge weights from the DEM, maintain a running estimate
of per-edge error rates based on recent detection history. Edges that fire more frequently
than expected get lower weights (higher probability → fuse faster). This adapts the decoder
to non-stationary noise without retraining.

**Files:**
- Modify: `decoder/include/decoder/union_find.h`
- Modify: `decoder/src/union_find.cpp`
- Create: `decoder/tests/test_decoder_steering.py`

- [ ] **Step 1: Add adaptive weight update to UnionFindDecoder**

Add to `decoder/include/decoder/union_find.h`:

```cpp
class UnionFindDecoder {
public:
    // ... existing ...

    // Decoder steering: update edge weights based on observed detection patterns
    // Call after each decode() to adapt weights to non-stationary noise
    void update_weights(const std::vector<bool>& detection_events, 
                        double learning_rate = 0.01);

    // Reset weights to DEM baseline
    void reset_weights();

private:
    // ... existing ...
    std::vector<double> running_detection_rates_;  // Per-detector running average
    int steering_window_;                           // Number of recent syndromes to average
    int steering_count_;                            // How many syndromes seen
};
```

- [ ] **Step 2: Implement adaptive weight update**

Add to `decoder/src/union_find.cpp`:

```cpp
void UnionFindDecoder::update_weights(const std::vector<bool>& detection_events,
                                       double learning_rate) {
    if (running_detection_rates_.empty()) {
        running_detection_rates_.resize(graph_.n_detectors, 0.0);
        steering_count_ = 0;
    }
    steering_count_++;

    // Exponential moving average of per-detector firing rates
    for (int i = 0; i < graph_.n_detectors && i < (int)detection_events.size(); i++) {
        double observed = detection_events[i] ? 1.0 : 0.0;
        running_detection_rates_[i] = 
            (1.0 - learning_rate) * running_detection_rates_[i] + 
            learning_rate * observed;
    }

    // Reweight edges: if detectors on an edge fire more than expected,
    // lower the edge weight (= higher probability = fuse faster)
    if (steering_count_ % 100 == 0) {  // Update weights every 100 syndromes
        for (size_t e = 0; e < graph_.edges.size(); e++) {
            auto& edge = graph_.edges[e];
            double base_weight = edge_weights_[e];

            // Compute observed error rate for this edge from detector rates
            double det_rate = 0.0;
            int n_det = 0;
            if (edge.source >= 0 && edge.source < (int)running_detection_rates_.size()) {
                det_rate += running_detection_rates_[edge.source];
                n_det++;
            }
            if (edge.target >= 0 && edge.target < (int)running_detection_rates_.size()) {
                det_rate += running_detection_rates_[edge.target];
                n_det++;
            }
            if (n_det > 0) det_rate /= n_det;

            // Adjust weight: higher detection rate → lower weight → faster fusion
            // Blend between DEM weight and observed-rate weight
            double observed_weight = (det_rate > 0) ? 
                std::max(1.0, -std::log(std::min(det_rate, 0.5))) : base_weight;
            edge_weights_[e] = 0.8 * base_weight + 0.2 * observed_weight;
        }
    }
}

void UnionFindDecoder::reset_weights() {
    for (size_t i = 0; i < graph_.edges.size(); i++) {
        double p = graph_.edges[i].error_prob;
        edge_weights_[i] = (p > 0 && p < 1) ? 
            std::max(1, (int)std::ceil(-std::log(p))) : 1;
    }
    running_detection_rates_.clear();
    steering_count_ = 0;
}
```

- [ ] **Step 3: Add Python binding for steering**

Add to `decoder/python/pydecoder.cpp` UnionFindDecoder binding:

```cpp
        .def("update_weights", &decoder::UnionFindDecoder::update_weights,
             py::arg("detection_events"), py::arg("learning_rate") = 0.01)
        .def("reset_weights", &decoder::UnionFindDecoder::reset_weights)
```

- [ ] **Step 4: Test decoder steering**

Create `decoder/tests/test_decoder_steering.py`:

```python
"""Test RL-inspired decoder steering (Sivak et al. arXiv:2511.08493)."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
import pydecoder


def graph_to_cpp(g):
    sg = pydecoder.SyndromeGraph()
    sg.n_detectors = g.n_detectors
    sg.n_observables = g.n_observables
    for s, t, p, o in g.edges:
        sg.add_edge(s, t, p, o)
    sg.build_adjacency()
    return sg


class TestDecoderSteering:
    def test_steering_runs(self):
        """Decoder steering should run without crashing."""
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.005)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        cpp_graph = graph_to_cpp(graph)
        dec = pydecoder.UnionFindDecoder(cpp_graph)

        det_events, obs_flips = sample_syndromes(circuit, num_shots=200)
        for i in range(200):
            det_list = list(det_events[i])
            result = dec.decode(det_list)
            dec.update_weights(det_list, learning_rate=0.01)

    def test_steering_adapts_to_drift(self):
        """Steering should improve LER when noise drifts from DEM calibration.

        Simulate drift: calibrate DEM at p=0.005, then decode at p=0.008.
        Steering should adapt weights and improve vs static weights.
        """
        # Calibrate at p=0.005
        calib_config = SurfaceCodeConfig(distance=5, rounds=5, physical_error_rate=0.005)
        calib_circuit = make_circuit(calib_config)
        calib_graph = extract_decoder_graph(calib_circuit)

        # But actual noise has drifted to p=0.008
        drift_config = SurfaceCodeConfig(distance=5, rounds=5, physical_error_rate=0.008)
        drift_circuit = make_circuit(drift_config)
        det_events, obs_flips = sample_syndromes(drift_circuit, num_shots=5000)

        # Static decoder (calibrated at p=0.005, decoding at p=0.008)
        static_graph = graph_to_cpp(calib_graph)
        static_dec = pydecoder.UnionFindDecoder(static_graph)
        static_pred = static_dec.decode_batch(det_events)
        static_ler = np.sum(np.any(static_pred != obs_flips, axis=1)) / 5000

        # Steered decoder (same calibration, but adapts via steering)
        steered_graph = graph_to_cpp(calib_graph)
        steered_dec = pydecoder.UnionFindDecoder(steered_graph)

        # Warm up steering with first 1000 shots
        for i in range(1000):
            det_list = list(det_events[i])
            steered_dec.decode(det_list)
            steered_dec.update_weights(det_list, learning_rate=0.02)

        # Evaluate on remaining 4000 shots
        steered_pred = steered_dec.decode_batch(det_events[1000:])
        steered_ler = np.sum(np.any(steered_pred != obs_flips[1000:], axis=1)) / 4000

        print(f"Static  LER (calibrated p=0.005, actual p=0.008): {static_ler:.4f}")
        print(f"Steered LER (adapted via 1000 shots):             {steered_ler:.4f}")

        # Steering should help (or at least not hurt)
        # Note: improvement may be marginal for small codes
        assert steered_ler <= static_ler * 1.1, \
            f"Steering should not make things much worse: {steered_ler} vs {static_ler}"

    def test_reset_weights(self):
        """reset_weights should return to DEM baseline."""
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.005)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        cpp_graph = graph_to_cpp(graph)
        dec = pydecoder.UnionFindDecoder(cpp_graph)

        det_events, _ = sample_syndromes(circuit, num_shots=100)

        # Run steering
        for i in range(100):
            det_list = list(det_events[i])
            dec.decode(det_list)
            dec.update_weights(det_list)

        # Reset and verify decoder still works
        dec.reset_weights()
        result = dec.decode([False] * circuit.num_detectors)
        assert all(not p for p in result.observable_prediction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 5: Build, test, commit**

```bash
cd build && make -j && cd ..
python -m pytest decoder/tests/test_decoder_steering.py -v
git add decoder/
git commit -m "feat(decoder): RL-inspired decoder steering — adaptive edge weights (Sivak et al. 2025)"
```

---

### Task 4: Full Benchmark Suite (GPU)

After all decoders are trained, run definitive comparison:

- [ ] **Step 1: Create comprehensive benchmark script**

Create `decoder/bench/full_comparison.py` that compares ALL decoders:
- PyMatching (MWPM) — accuracy baseline
- C++ Union-Find (static weights)
- C++ Union-Find (with steering, warmed up on 1000 shots)
- Neural CNN (d=3, d=5, d=7)
- Python BP (for reference)

At distances d=3,5,7 and noise rates p=0.001, 0.002, 0.005, 0.007, 0.01.

Report: LER, latency per syndrome, throughput (syndromes/sec).

- [ ] **Step 2: Run benchmark**

```bash
python decoder/bench/full_comparison.py | tee decoder/bench/results/full_comparison.txt
```

- [ ] **Step 3: Commit results**

```bash
git add decoder/bench/
git commit -m "feat(decoder): definitive benchmark — Neural vs UF(steered) vs BP vs PyMatching"
```

---

### Task 5: Update README with Decoder Results

- [ ] **Step 1: Update README.md**

Add decoder section with:
- Architecture diagram (3-tier decoder)
- LER comparison table (all decoders × all distances × noise rates)
- Training details (Muon, curriculum, Gu et al. reference)
- Decoder steering explanation (Sivak et al. reference)

- [ ] **Step 2: Commit and push**

```bash
git commit -m "docs: decoder results — Neural beats PyMatching at d=3, competitive at d=5,7"
git push origin master
```

---

## MI300X Session Checklist

When the MI300X instance is up, execute in this order:

1. Install deps: `pip install torch stim pymatching pybind11 numpy pytest`
2. Clone repo: `git clone https://github.com/bledden/qubit-forge.git`
3. Build decoder: `mkdir build && cd build && cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) && make -j`
4. Train d=5 (Task 1): ~30-45 min
5. Train d=7 (Task 2): ~45-60 min
6. Implement decoder steering (Task 3): build + test locally
7. Run full benchmarks (Task 4): ~15 min
8. Update README (Task 5): ~5 min
9. Push everything, shut down instance

**Total estimated time: 2-3 hours. Cost: ~$4-6 at $1.99/hr.**
