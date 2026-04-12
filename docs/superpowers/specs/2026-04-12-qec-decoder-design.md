# GPU-Accelerated QEC Decoder — Design Spec

**Date:** 2026-04-12
**Author:** Blake Ledden
**Target Hardware:** AMD MI300X (bare metal for final benchmarks)
**Goal:** Build three decoder tiers — Union-Find (CPU baseline), Belief Propagation (GPU), and Neural CNN (GPU, state-of-the-art) — benchmarked against PyMatching/Sparse Blossom, targeting sub-100μs latency on MI300X.

**Key reference:** Gu et al., "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation" (arXiv:2604.08358, April 9, 2026) — demonstrates CNN decoder achieving 17x better logical error rates than BP+OSD with ~40μs latency on H200.

## 1. What This Builds

A quantum error correction (QEC) decoder that takes noisy syndrome measurements from a surface code and determines the most likely error pattern. This is THE bottleneck for fault-tolerant quantum computing — decoders must run faster than errors accumulate (~1μs per round for superconducting qubits).

Three decoder tiers, each building on the previous:

1. **Union-Find (CPU)** — near-linear time baseline, validates correctness, builds the surface code infrastructure
2. **Belief Propagation + OSD (GPU)** — massively parallel message-passing, establishes the GPU decoding pipeline
3. **Neural CNN Decoder (GPU)** — convolutional neural network following Gu et al. 2026, targets state-of-the-art accuracy and competitive latency on MI300X

All integrated with Stim for syndrome generation and PyMatching for accuracy comparison.

### Why Three Tiers

- **Union-Find** builds the syndrome extraction infrastructure that all decoders share
- **BP+OSD** establishes the GPU kernel pipeline and gives us a comparison point
- **Neural CNN** is the frontier — Gu et al. showed it achieves 17x lower logical error rates than BP+OSD by learning flexible message-passing rules that avoid trapping sets. CNN inference is also more GPU-friendly than iterative BP (fixed computation graph, no convergence check, FP8-quantizable)

The story: "Union-Find at X μs (CPU), BP+OSD at Y μs (GPU), Neural at Z μs (GPU) with 17x better accuracy."

## 2. Background: Surface Code Decoding

### The Surface Code

A distance-d surface code encodes 1 logical qubit in d² physical qubits arranged on a 2D grid. d²-1 stabilizer measurements detect errors without disturbing the logical state.

For distance d=5: 25 data qubits, 24 stabilizers (12 X-type, 12 Z-type).

### Syndrome Extraction

Each round of error correction:
1. Apply stabilizer measurement circuits (CNOT chains between data qubits and ancilla qubits)
2. Measure ancillas → raw measurement bits
3. Compute detection events: `s_i^t = m_i^t XOR m_i^{t-1}` (difference between consecutive rounds)
4. A detection event (s=1) indicates an error occurred nearby

Run d rounds for temporal redundancy. The full syndrome is a 3D binary array: (stabilizer_index, time_round).

### Decoder's Job

Input: set of detection events (defect locations in the 3D syndrome graph)
Output: predicted correction (which logical observable was flipped)
Metric: logical error rate (LER) — fraction of decodings that produce wrong corrections

## 3. Architecture

```
qubit-forge/
├── decoder/
│   ├── include/decoder/
│   │   ├── types.h             # Syndrome, Correction, DecoderResult
│   │   ├── surface_code.h      # Surface code graph construction
│   │   ├── union_find.h        # Union-Find decoder
│   │   ├── bp_decoder.h        # Belief Propagation decoder
│   │   ├── bp_osd.h            # OSD post-processing
│   │   └── neural_decoder.h    # CNN decoder (inference)
│   ├── src/
│   │   ├── surface_code.cpp    # Syndrome graph from Stim DEM
│   │   ├── union_find.cpp      # Union-Find: growth, merge, peel
│   │   ├── bp_decoder.hip      # BP message-passing GPU kernel
│   │   ├── bp_osd.hip          # OSD Gaussian elimination on GPU
│   │   └── neural_decoder.hip  # CNN inference kernel
│   ├── train/
│   │   ├── model.py            # CNN architecture (PyTorch)
│   │   ├── train.py            # Training loop with Stim data gen
│   │   ├── export.py           # Export trained weights for HIP inference
│   │   └── configs/            # Per-code training configs
│   ├── python/
│   │   └── pydecoder.cpp       # pybind11 bindings
│   ├── bench/
│   │   ├── latency_bench.py    # Per-syndrome decoding latency
│   │   ├── throughput_bench.py # Batch throughput (syndromes/sec)
│   │   ├── ler_bench.py        # Logical error rate curves
│   │   └── compare_all.py      # Head-to-head: UF vs BP vs Neural vs PyMatching
│   └── tests/
│       └── test_decoder.py
```

## 4. Data Flow

```
Stim (syndrome generation)
    ↓
stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, ...)
    ↓
sampler.sample(shots=N) → detection_events (bool[N, n_detectors]), observable_flips (bool[N, n_obs])
    ↓
stim.DetectorErrorModel → weighted graph (edge list with error probabilities)
    ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Union-Find   │   │ BP+OSD       │   │ Neural CNN       │
│ (CPU)        │   │ (GPU)        │   │ (GPU)            │
│              │   │              │   │                  │
│ Input:       │   │ Input:       │   │ Input:           │
│  defect list │   │  syndrome    │   │  syndrome tensor │
│ Graph: DEM   │   │  H matrix    │   │  (3D: space×time)│
│              │   │              │   │                  │
│ Output:      │   │ Output:      │   │ Output:          │
│  correction  │   │  correction  │   │  P(logical flip) │
│              │   │              │   │  + confidence    │
│ ~10-100μs    │   │ <100μs       │   │ <50μs target     │
└──────────────┘   └──────────────┘   └──────────────────┘
    ↓                    ↓                    ↓
         Compare predicted vs actual observable_flips
                         ↓
                  Logical Error Rate (LER)
```

## 5. Union-Find Decoder (CPU)

### Data Structures

```cpp
struct Node {
    int parent;         // Union-Find parent (index, not pointer — cache-friendly)
    int rank;           // Union-by-rank
    int cluster_size;   // Number of defects in cluster
    bool is_boundary;   // Virtual boundary node
};

struct Edge {
    int u, v;           // Node indices
    double weight;      // -log(error_probability) from DEM
    int growth;         // Current growth counter
    bool fully_grown;
};

struct SyndromeGraph {
    std::vector<Node> nodes;       // Defect nodes + boundary nodes
    std::vector<Edge> edges;       // Weighted edges from DEM
    std::vector<int> defects;      // Indices of active defect nodes
    int n_detectors;
    int n_observables;
};
```

### Algorithm (Delfosse & Nickerson 2021)

1. **Initialize**: Create node per defect. Add boundary nodes. Build edge list from Stim's DetectorErrorModel.

2. **Growth**: For each odd cluster (odd number of defects), increment growth on all boundary edges by 1. Edge becomes fully grown when total growth from both sides ≥ weight.

3. **Merge**: When edge is fully grown, union the two endpoint clusters. Odd + odd = even (stop growing). Odd + boundary = even.

4. **Peel** (correction extraction): Build spanning tree of fully-grown edges. Process leaves first. For each leaf edge: if subtree has odd defect count, include edge in correction.

5. **Output**: The correction predicts which observables were flipped.

### Complexity

O(n · α(n)) where n = number of edges in syndrome graph. For distance d with d rounds: n ≈ O(d³). In practice: ~10-100μs per syndrome at d=5-15.

## 6. Belief Propagation Decoder (GPU)

### Parity-Check Matrix

The surface code (or any QLDPC code) is defined by parity-check matrices H_X and H_Z. Each row = one stabilizer, each column = one qubit. For CSS codes, decode X and Z errors independently.

Store H in CSR format for check-node updates, CSC for variable-node updates.

### Min-Sum Message Passing

Avoids transcendental functions (no atanh/tanh) — pure multiply/compare, GPU-friendly.

**Variable-to-check message:**
```
m_{v→c} = channel_LLR[v] + sum_{c' ≠ c} m_{c'→v}
```

**Check-to-variable message (min-sum):**
```
m_{c→v} = (product of signs of m_{v'→c} for v' ≠ v) × min_{v' ≠ v} |m_{v'→c}|
```

**Hard decision after T iterations:**
```
LLR[v] = channel_LLR[v] + sum_c m_{c→v}
error[v] = 1 if LLR[v] < 0
```

### GPU Kernel Design

```cpp
// One thread block per check node (or batch of check nodes)
// Shared memory holds messages for current check row
__global__ __launch_bounds__(256, 4)
void bp_check_update(
    float* __restrict__ messages_vc,
    float* __restrict__ messages_cv,
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_idx,
    int n_checks)
{
    // Each check node reads its neighbor variable messages
    // Computes product-of-signs and min-of-abs
    // Writes back check→variable messages
}

__global__ __launch_bounds__(256, 4)
void bp_variable_update(
    float* __restrict__ messages_cv,
    float* __restrict__ messages_vc,
    const float* __restrict__ channel_llr,
    const int* __restrict__ csc_col_ptr,
    const int* __restrict__ csc_row_idx,
    int n_variables)
{
    // Each variable node sums incoming check messages + channel prior
    // Writes variable→check messages
}
```

**Key optimizations from hackathon lessons:**
- FP16 messages (half bandwidth vs FP32)
- Fuse check + variable update into single kernel (avoid global memory round-trip)
- Batch decode: outer dimension = syndrome index, inner loop = BP iterations
- `__launch_bounds__(256, 4)` for bandwidth-bound sweet spot
- Non-temporal stores for message arrays

### OSD Post-Processing

When BP fails to converge (syndrome not satisfied after max iterations):
1. Sort columns of H by LLR magnitude (most reliable first)
2. Gaussian elimination to find information set
3. OSD-0: use most-likely consistent solution
4. OSD-w (w=5-10): enumerate weight-w perturbations

OSD is sequential but only runs on ~5-20% of syndromes (BP converges for the rest).

## 7. Neural CNN Decoder (GPU)

### Why Neural Over BP

Gu et al. (arXiv:2604.08358, April 2026) demonstrated that a CNN decoder achieves:
- **17x lower logical error rates** than BP+OSD on [144,12,12] Gross code
- **~40μs single-shot latency** on H200
- **No accuracy loss** at FP8 quantization
- **"Waterfall regime"**: error suppression scaling as p^11 (vs p^5.4 for BP+OSD) because the CNN learns to correct high-weight failure modes that BP misses due to trapping sets

BP is fundamentally limited by short cycles in the Tanner graph. The CNN learns flexible message-passing rules that transcend these limitations.

### Architecture (following Gu et al.)

```
Input: Binary syndrome tensor [R rounds × d × d spatial]
  ↓
Embedding: syndrome bits → H-dimensional representations
  ↓
L convolutional blocks (L ~ d):
  Each block:
    - 1×1 conv (reduce H → H/4)
    - 3D conv with direction-specific weights (message passing)
    - 1×1 conv (restore H/4 → H)
    - Residual connection
    - LayerNorm
  ↓
Final conv: scatter check representations → data qubit positions
  ↓
Average pooling → per-logical-observable aggregation
  ↓
2-layer MLP → logits for each logical observable + confidence score
```

**Key design choices from the paper:**
- **Direction-specific weights**: convolution weights depend on relative position between stabilizers, not absolute position (translation equivariance)
- **Hidden dimension H**: 256-512 (larger = more accurate, more compute)
- **Depth L ~ d**: scales with code distance
- **Bottleneck residual**: reduce dimension 4x before message-passing convolution to save compute

### Training Pipeline

```python
# train/train.py
import stim
import torch

# Generate training data on-the-fly with Stim
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=d, rounds=d,
    after_clifford_depolarization=p_train,
)
sampler = circuit.compile_detector_sampler()

# Training loop
for step in range(80_000):
    # Sample batch of syndromes
    det_events, obs_flips = sampler.sample(batch_size, separate_observables=True)

    # Forward pass
    syndrome_tensor = reshape_to_3d(det_events, d, rounds)  # [B, R, d, d]
    logits = model(syndrome_tensor)                          # [B, n_observables]

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(logits, obs_flips.float())
    loss.backward()
    optimizer.step()
```

**Training details from the paper:**
- Optimizer: Muon (Newton-Schulz orthogonalization) + Lion
- Schedule: cosine decay, 1000-step warmup, 80K steps
- Batch size: 3,328
- Curriculum: 3-stage noise annealing (low → target error rate)
- Training noise: p=0.7% for surface codes
- Mixed precision: bfloat16
- Training cost: ~200 GPU-hours for largest model (d=13 surface code)

### HIP Inference Kernel

After training in PyTorch, export weights and run inference in raw HIP for minimum latency:

```cpp
// decoder/src/neural_decoder.hip

struct ConvBlock {
    // Pre-loaded weights in device memory
    half* w_reduce;     // [H/4, H, 1, 1, 1] — 1x1 reduce
    half* w_message;    // [H/4, H/4, 3, 3, 3] — 3D message passing
    half* w_restore;    // [H, H/4, 1, 1, 1] — 1x1 restore
    half* ln_scale;     // [H] — LayerNorm
    half* ln_bias;      // [H]
};

struct NeuralDecoder {
    int n_blocks;           // L ~ d
    int hidden_dim;         // H = 256 or 512
    ConvBlock* blocks;      // Pre-loaded on GPU
    half* w_embed;          // Embedding layer
    half* w_scatter;        // Final scatter conv
    half* w_mlp;            // MLP weights
};

// Inference: syndrome → logits
// Each block: 1x1 reduce → 3D conv → 1x1 restore → residual → LayerNorm
// Use FP16 (or FP8 if MI300X supports it via MFMA) for maximum throughput
__global__ __launch_bounds__(256, 4)
void neural_decode_batch(
    const int8_t* __restrict__ syndromes,   // [batch, R, d, d] packed bits
    half* __restrict__ logits,              // [batch, n_observables]
    const NeuralDecoder decoder,
    int batch_size);
```

**FP8 quantization opportunity:** The paper shows no accuracy loss at FP8. MI300X (gfx942) has FP8 MFMA instructions (`mfma_f32_16x16x32_fp8`). If we express the 3D convolution as GEMM (im2col approach), we can use MFMA for the compute-intensive convolution layers. This is where the CUTLASS/CuTe experience directly transfers.

### Model Size Estimates

For d=5 surface code, H=256, L=5 blocks:
- Per block: ~256K parameters (reduce + message + restore)
- Total: ~1.3M parameters = ~2.6 MB at FP16, ~1.3 MB at FP8
- Fits entirely in L2 cache (256 MB on MI300X) — inference is compute-bound, not bandwidth-bound

For d=11, H=512, L=11 blocks:
- Total: ~30M parameters = ~60 MB at FP16
- Still fits in L2 cache

## 8. Stim Integration

```python
import stim

# Generate surface code circuit with noise
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=d,
    rounds=d,
    after_clifford_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
)

# Get detector error model (weighted graph for decoder initialization)
dem = circuit.detector_error_model(decompose_errors=True)

# Sample syndromes
sampler = circuit.compile_detector_sampler()
detection_events, observable_flips = sampler.sample(
    shots=num_shots, separate_observables=True
)
```

Stim generates syndromes at ~1 billion Clifford gates/sec. This is our data source for both training and benchmarking.

## 9. Testing Strategy

### Correctness Tests
1. **Trivial syndrome**: No defects → no correction needed
2. **Single error**: One defect pair → decoder finds the connecting path
3. **Known error pattern**: Inject specific errors, verify decoder recovers
4. **Match PyMatching**: For small codes (d=3,5), verify our decoder agrees with PyMatching on >99% of syndromes
5. **Neural matches BP on easy syndromes**: For low-noise syndromes, neural and BP should agree

### Accuracy Tests (LER curves)
For each distance d in [3, 5, 7, 9, 11]:
- Sweep physical error rate p from 0.001 to 0.02
- Run 100,000+ decoding shots per (d, p) point
- Plot LER vs p for all four decoders (UF, BP+OSD, Neural, PyMatching)
- Verify: LER decreases with increasing d (below threshold)
- Verify: Neural < BP+OSD < UF in LER (accuracy ordering)
- Look for "waterfall regime" (p^11 scaling) in neural decoder

### Latency Tests
- Single-syndrome decoding latency at d=5, 7, 11, 15, 21
- Batch throughput: syndromes/sec at batch sizes 100, 1000, 10000
- Compare: Union-Find (CPU) vs BP (GPU) vs Neural (GPU) vs PyMatching (CPU)

## 10. Benchmarks

### Target Numbers
| Metric | Target | Reference |
|--------|--------|-----------|
| Union-Find latency (d=11) | <100μs | Helios FPGA: 11.5ns |
| BP+OSD latency (single) | <100μs | GPU-LDPC paper: sub-63μs |
| Neural latency (single) | <50μs | Gu et al.: ~40μs on H200 |
| Neural batch throughput | >500K syndromes/sec | Gu et al.: 3000-100000x faster than CPU |
| Neural accuracy | 17x better LER than BP+OSD | Gu et al. on [144,12,12] Gross code |
| UF accuracy vs MWPM | Within 10x LER | Literature: UF ~50% of MWPM gap |

### Comparison Baselines
- **PyMatching** (Sparse Blossom): gold standard for accuracy + reasonable speed
- **GPU-LDPC paper**: speed target for BP decoder
- **Gu et al. 2026**: accuracy and speed target for neural decoder
- **Helios FPGA**: theoretical speed ceiling (11.5ns, FPGA-only)

### The Publishable Result

If the neural decoder on MI300X matches or beats the H200 results from Gu et al.:
- First MI300X neural decoder benchmark for QEC
- Direct comparison: MI300X vs H200 for real-time decoding
- Demonstrates that AMD hardware is viable for quantum computing infrastructure

## 11. Build Integration

Add to existing CMakeLists.txt:

```cmake
# Decoder library
add_library(decoder SHARED
    decoder/src/surface_code.cpp
    decoder/src/union_find.cpp
    decoder/src/bp_decoder.hip
    decoder/src/bp_osd.hip
    decoder/src/neural_decoder.hip
)
target_include_directories(decoder PUBLIC decoder/include)
target_link_libraries(decoder hip::device)

# Decoder Python bindings
set_source_files_properties(decoder/python/pydecoder.cpp PROPERTIES LANGUAGE HIP)
pybind11_add_module(pydecoder decoder/python/pydecoder.cpp)
target_link_libraries(pydecoder PRIVATE decoder)
```

Python deps: `pip install stim pymatching numpy pytest torch`

## 12. Implementation Order

1. **Surface code infrastructure** — Stim integration, syndrome graph construction, DEM parsing
2. **Union-Find decoder** — CPU baseline, correctness validation
3. **BP+OSD decoder** — GPU message-passing kernels, establish GPU pipeline
4. **Neural CNN training** — PyTorch model, Stim data generation, training loop
5. **Neural CNN HIP inference** — Export weights, write inference kernel, FP8 quantization
6. **Benchmarks** — Latency, throughput, LER curves, head-to-head comparison

## 13. Non-Goals

- Real-time decoder integration with quantum hardware (we simulate, not deploy)
- FPGA implementation (GPU only)
- Codes beyond surface code and bivariate bicycle/QLDPC (extensible but not in scope)
- Custom noise models beyond Stim's circuit-level depolarizing
- Multi-GPU decoding
- Training infrastructure optimization (use standard PyTorch, not custom training kernels)
