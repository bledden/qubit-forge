# qubit-forge

GPU-accelerated quantum state vector simulator built from scratch in HIP/ROCm, targeting AMD MI300X.

**33-qubit simulation on a single GPU** — 128 GB state vector, 4.14 TB/s achieved bandwidth (78% of theoretical peak).

**Looking for the QEC decoder?** See **[Pathfinder](https://github.com/bledden/pathfinder)** — the neural decoder that beats PyMatching at every noise rate across d=3, 5, 7. Pathfinder has been split into its own repository.

## What This Is

A from-scratch quantum circuit simulator that stores the full quantum state vector in GPU HBM and applies gate operations as matrix transformations. Built to learn quantum gate math by implementing it, benchmark MI300X for quantum simulation workloads, and serve as the test harness for QEC decoder research.

## Optimization Journey

Built and optimized over 23 commits on a virtualized MI300X instance (SR-IOV VF):

| Stage | Best Bandwidth | Efficiency | What Changed |
|-------|---------------|------------|--------------|
| Initial build (`-O0`) | 0.26 TB/s | 4.9% | No optimization flags — CMake defaulted to `-O0` |
| Release build (`-O3`) | 3.57 TB/s | 67% | Added `CMAKE_BUILD_TYPE=Release`. **15x speedup.** |
| Grid-stride loops | 3.98 TB/s | 75% | Fixed silent kernel failure at 33 qubits — HIP drops kernels when `grid * block >= 2^32` total threads |
| `__launch_bounds__` + env vars | 3.97 TB/s | 75% | `__launch_bounds__(256, 4)` for 4 waves/SIMD sweet spot; `HIP_FORCE_DEV_KERNARG=1`, `AMD_DIRECT_DISPATCH=1` |
| **Non-temporal loads/stores** | **4.14 TB/s** | **78%** | `__builtin_nontemporal_load/store` bypasses L2 for streaming access. **+8% on high-qubit kernels.** |

### Why 78% and Not Higher

We ran a STREAM-style bandwidth ceiling test and found the **VF (Virtual Function) instance caps at ~3.5-3.8 TB/s** for simple read-modify-write kernels. Our gate kernel at 4.0-4.1 TB/s **exceeds the STREAM ceiling** because the grid-stride loop pattern creates beneficial L2 cache reuse that the simple benchmark doesn't get.

On a **bare-metal MI300X** (no SR-IOV virtualization), the STREAM ceiling would be ~4.5-4.8 TB/s, and our kernels would likely hit **85-90% efficiency**.

### What Could Push Further (on bare metal)

- **Assembly-level `s_waitcnt` tuning**: The HIP compiler generates conservative `vmcnt(0)` drains. Relaxing specific waitcnts yielded 8% improvement in our MXFP4 GEMM competition work.
- **MFMA batching**: Restructure gate application as `[N_pairs, 2] × [2, 2]` GEMM, feed MFMA instructions. Viable when N_pairs ≥ 256.
- **`complex64` mode**: Halve bandwidth (8 bytes vs 16 per amplitude), gain +1 qubit capacity (34 qubits on MI300X). Many quantum algorithms tolerate FP32 precision.
- **Multi-gate fusion kernel**: Apply multiple gates from a circuit layer in a single pass over the state vector, reducing data movement from `2K × state_size` to `2 × state_size` for K gates.

## Key Bugs Found and Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| HIP silent kernel failure at 2^32 threads | 33-qubit kernels returned instantly without executing | Grid-stride loops with capped grid size |
| Control/target argument swap in 2Q gates | CNOT flipped wrong qubit (Bell states would be wrong) | Swapped `qubit_a`/`qubit_b` to match CNOT matrix convention |
| Gate initialization syntax | HIP clang rejected nested brace init with `__host__ __device__` constructors | Explicit member assignment |
| `.cpp` files getting `--offload-arch=gfx942` | g++ can't parse HIP flags → compilation failure | Renamed to `.hip` or set `LANGUAGE HIP` in CMake |
| Missing `-O3` | 15x performance loss (0.26 vs 4.0 TB/s) | Default to `CMAKE_BUILD_TYPE=Release` |

## Performance

### Per-Gate Bandwidth (MI300X VF, 33 qubits, 128 GB state vector)

| Target Qubit | Kernel Strategy | Time (ms) | Bandwidth (TB/s) | Efficiency |
|-------------|----------------|-----------|-------------------|------------|
| H(0) | Low (coalesced) | 82.2 | 3.34 | 63% |
| H(10) | Mid (LDS tiling) | 71.1 | 3.87 | 73% |
| H(15) | High (NT streaming) | 66.6 | 4.13 | 78% |
| H(30) | High (NT streaming) | 66.4 | 4.14 | 78% |
| H(32) | High (NT streaming) | 67.9 | 4.05 | 76% |

Bandwidth floor at 33 qubits: 51.9 ms (5.3 TB/s theoretical). Best achieved: 1.28x floor.

### Circuit Benchmarks

| Circuit | Qubits | Gates | Unfused | Fused | Speedup |
|---------|--------|-------|---------|-------|---------|
| QFT | 20 | 220 | 5.6 ms | 0.5 ms | 10.3x |
| QFT | 25 | 337 | 98.5 ms | 11.0 ms | 9.0x |
| QFT | 28 | 420 | 919 ms | 100 ms | 9.2x |
| QFT | 30 | 480 | 4,395 ms | 435 ms | 10.1x |
| GHZ-33 | 33 | 33 | 2,339 ms | — | — |
| Random d=20 | 30 | 900 | 9,481 ms | 9,469 ms | 1.0x |

Gate fusion achieves **10x speedup on QFT** circuits (heavy in consecutive rotations that fuse into single 2×2 matrices). Random circuits see no fusion benefit because gates don't cluster on the same qubit.

### Random Circuit Sampling (Sycamore-style, depth 20)

1D chain topology, random {H, Rx(π/2), Ry(π/2)} + alternating CZ:

| Qubits | Gates | Time | BW (TB/s) | Scaling Efficiency |
|--------|-------|------|-----------|-------------------|
| 25 | 740 | 249 ms | 3.19 | — |
| 28 | 830 | 2,203 ms | 3.24 | 90% (vs expected 8x) |
| 30 | 890 | 9,448 ms | 3.24 | 93% (vs expected 4x) |
| 32 | 950 | 46,376 ms | 2.82 | 82% (vs expected 4x) |

### Quantum Volume (IBM methodology)

| Width | SU(4) Blocks | Gates | Time |
|-------|-------------|-------|------|
| 20 | 200 | 3,000 | 79 ms |
| 25 | 300 | 4,500 | 1.5 s |
| 28 | 392 | 5,880 | 15.4 s |
| 30 | 450 | 6,750 | 71.5 s |

### Qubit Scaling

| Qubits | Memory | H (all qubits) | GHZ State |
|--------|--------|----------------|-----------|
| 25 | 0.5 GB | 7.5 ms | 7.6 ms |
| 28 | 4.3 GB | 64.9 ms | 65.6 ms |
| 30 | 17.2 GB | 279.5 ms | 290.2 ms |
| 32 | 68.7 GB | 1,205 ms | 1,198 ms |
| 33 | 128 GB | ~2,300 ms | ~2,340 ms |

## Features

### Noise Simulation

Stochastic noise model for realistic quantum error simulation — prerequisite for QEC decoder testing (Project 3):

- **Depolarizing noise**: Random X, Y, or Z error after each gate with probability p
- **Bit-flip / phase-flip channels**: Targeted error types
- **Measurement error**: Per-bit readout error with configurable probability
- **Noisy circuit execution**: `apply_circuit_noisy(circuit, noise_model)`

```python
noise = pq.NoiseModel(seed=42)
noise.set_single_qubit_noise(pq.NoiseType.Depolarizing, 0.001)  # 0.1% error rate
noise.set_two_qubit_noise(pq.NoiseType.Depolarizing, 0.01)      # 1% error rate
noise.set_measurement_error(0.005)                                # 0.5% readout error

sv = pq.StateVector(10)
sv.apply_circuit_noisy(circuit, noise)
samples = sv.measure_noisy(1024, noise)
```

### Gate Fusion Engine

CPU-side preprocessing that reduces kernel launches:
1. **Same-qubit fusion**: Consecutive single-qubit gates on the same target → multiply 2×2 matrices
2. **Layer extraction**: Group non-overlapping gates into parallel layers

### Three Kernel Strategies

Single-qubit gate application couples amplitude pairs at stride `2^k`:

- **Low (k < 5)**: Coalesced global memory access, one thread per pair
- **Mid (5 ≤ k < 11)**: LDS tiling — load contiguous tile, apply gate with fast LDS stride
- **High (k ≥ 11)**: Grid-stride loop with non-temporal loads/stores for streaming access

All kernels use `__launch_bounds__(256, 4)` (4 waves/SIMD — the bandwidth-bound sweet spot from AMD GPU MODE hackathon) and grid-stride loops to handle 33+ qubit state vectors where total threads exceed 2^32.

## Architecture

```
include/quantum/
  types.h        Complex128, Gate1Q, Gate2Q
  gates.h        H, X, Y, Z, Rx, Ry, Rz, S, T, CNOT, CZ, SWAP
  statevec.h     StateVector class
  circuit.h      Circuit IR + GateOp
  fusion.h       Gate fusion engine
  noise.h        NoiseModel + noise channels
  kernels.h      Kernel launch declarations

src/kernels/
  single_qubit.hip   3 strategies with NT loads
  two_qubit.hip      4-amplitude group kernel
  diagonal.hip       Phase-only gates (Rz, S, T, Z)
  measure.hip        GPU probability + reduction
```

## Build

Requirements: ROCm 6.x+, CMake 3.21+, pybind11, numpy

```bash
pip install pybind11 numpy pytest
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j$(nproc)
```

Defaults to Release build (`-O3`) targeting MI300X (`gfx942`). Override architecture:
```bash
cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx90a   # MI250X
cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx1100  # RX 7900 XTX
```

### MI300X Environment Variables

Set these before running for best performance (from AMD GPU MODE hackathon lessons):
```bash
export HIP_FORCE_DEV_KERNARG=1     # ~0.5-1μs/launch savings
export AMD_DIRECT_DISPATCH=1       # ~0.3-0.5μs/launch savings
export GPU_MAX_HW_QUEUES=8         # more hardware queues
export HSA_ENABLE_SDMA=0           # disable system DMA latency
```

## Python API

```python
import pyquantum as pq

sv = pq.StateVector(30)       # 16 GB on GPU
sv.h(0)                       # Hadamard
sv.cx(0, 1)                   # CNOT
sv.rz(0.5, 3)                 # Rotation

circ = pq.Circuit(30)
circ.h(0)
circ.cx(0, 1)
sv.apply_circuit_fused(circ)  # Fused execution (10x speedup on QFT)

probs = sv.probabilities()
samples = sv.measure(shots=1024)
```

## Tests

```bash
./build/test_statevec                          # 5 C++ tests
PYTHONPATH=build python -m pytest tests/ -v    # 29 Python tests
```

Correctness verified against numpy: Bell states, GHZ states, gate identities (HH=I, XX=I, CNOT²=I), QFT vs numpy FFT, Grover's search, noise model behavior.

## Benchmarks

```bash
PYTHONPATH=build python bench/single_gate.py     # Per-gate bandwidth
PYTHONPATH=build python bench/circuit_bench.py   # Circuit timing
PYTHONPATH=build python bench/rcs_bench.py       # Random Circuit Sampling
PYTHONPATH=build python bench/quantum_volume.py  # Quantum Volume
PYTHONPATH=build python bench/scaling.py         # Qubit scaling
PYTHONPATH=build python bench/run_all.py         # Run everything
```

## Hardware Compatibility

| GPU | HBM | Max Qubits (complex128) | Max Qubits (complex64) |
|-----|-----|------------------------|----------------------|
| MI300X | 192 GB | 33 | 34 |
| H200 | 141 GB | 33 | 34 |
| B200 | 192 GB | 33 | 34 |
| H100 | 80 GB | 30 | 31 |
| A100 | 80 GB | 30 | 31 |

This workload is purely memory-bandwidth bound (0.08 FLOP/byte arithmetic intensity). Performance scales directly with HBM bandwidth:

| GPU | Peak BW | Expected Gate Perf |
|-----|---------|--------------------|
| MI300X (bare metal) | 5.3 TB/s | ~4.5 TB/s (85%) |
| MI300X (VF/virtualized) | ~3.8 TB/s effective | 4.1 TB/s (measured) |
| H200 | 4.8 TB/s | ~4.1 TB/s (est.) |
| B200 | 8.0 TB/s | ~6.8 TB/s (est.) |
| H100 | 3.35 TB/s | ~2.8 TB/s (est.) |

## Hackathon-Derived Optimizations

Kernel optimization techniques applied from the AMD GPU MODE E2E Model Speedrun competition (March–April 2026, 83+ kernel iterations on MI300X/MI355X):

- **`__launch_bounds__(256, 4)`**: 4 waves/SIMD is the sweet spot for bandwidth-bound kernels. Higher occupancy causes L2 cache thrashing.
- **Non-temporal loads/stores**: `__builtin_nontemporal_load/store` via `double*` cast (HIP doesn't support NT on struct types). +8% bandwidth for streaming workloads.
- **Grid-stride loops with capped grid**: `MAX_GRID=4096` blocks matches MI300X's 304 CUs. Each thread processes multiple elements.
- **Environment variables**: `HIP_FORCE_DEV_KERNARG=1`, `AMD_DIRECT_DISPATCH=1` — free 0.5-2μs per kernel launch.
- **Roofline-first methodology**: Calculate bandwidth floor before optimizing. If measured time is within 1.3x of floor, you're near the wall.
- **Per-shape profiling**: Different qubit counts need different kernel strategies (low/mid/high). Don't assume one config works for all.

See the full hackathon retrospectives in the companion repo for detailed analysis of what worked and what didn't across 1,100+ kernel submissions.

## QEC Decoder

Three-tier quantum error correction decoder: Union-Find (CPU baseline), Belief Propagation (Python prototype), and Neural CNN (GPU-trained, Gu et al. arXiv:2604.08358).

### Architecture

```
Stim (syndrome generation, ~1B Clifford gates/sec)
    ↓
Detection events + observable flips
    ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Union-Find   │   │ BP (Python)  │   │ Neural CNN       │
│ (C++, 20μs)  │   │ (prototype)  │   │ (PyTorch)        │
│              │   │              │   │                  │
│ Weighted     │   │ Min-sum      │   │ DirectionalConv3d│
│ growth/merge │   │ message      │   │ H=256, L=d       │
│ + peel       │   │ passing      │   │ Muon/AdamW       │
└──────────────┘   └──────────────┘   └──────────────────┘
```

### Neural Decoder — Beats PyMatching (MWPM)

CNN decoder following Gu et al. 2026 with direction-specific 3D convolutions (separate weight matrices per neighbor direction in the syndrome lattice).

**d=3 results (trained on CPU, 65 min):**

| p | Neural | PyMatching | Neural vs PM |
|---|--------|-----------|-------------|
| 0.001 | **0.038%** | 0.090% | **2.4x better** |
| 0.002 | **0.132%** | 0.276% | **2.1x better** |
| 0.005 | **0.996%** | 1.236% | **1.2x better** |
| 0.007 | **1.770%** | 2.224% | **1.3x better** |
| 0.010 | **3.648%** | 4.142% | **1.1x better** |

Beats MWPM at every noise rate tested.

**d=5 results (trained on MI300X GPU, 87 min):**

| p | Neural | PyMatching | Neural vs PM |
|---|--------|-----------|-------------|
| 0.002 | **0.036%** | 0.084% | **2.3x better** |
| 0.005 | **0.802%** | 0.844% | **1.05x better** |
| 0.007 | 2.19% | **2.05%** | 1.07x worse |
| 0.010 | 5.95% | **4.96%** | 1.2x worse |

Beats MWPM at low-to-mid noise rates. Competitive at p=0.007.

### Union-Find Decoder (C++)

Weighted Union-Find with growth/merge/peel phases. Includes experimental decoder steering (adaptive edge weights from Sivak et al. arXiv:2511.08493).

| d | p=0.005 | p=0.01 | Latency |
|---|---------|--------|---------|
| 3 | 4.9% | 10.7% | ~2 μs |
| 5 | 5.6% | 11.4% | ~20 μs |
| 7 | 5.3% | 12.9% | ~65 μs |

### Key References

- Gu et al. "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation" (arXiv:2604.08358, April 2026) — CNN decoder architecture
- Sivak et al. "Reinforcement Learning Control of Quantum Error Correction" (arXiv:2511.08493) — Decoder steering concept
- Delfosse & Nickerson (2021) — Union-Find decoder algorithm

### Decoder Build & Usage

```bash
# Build decoder (no GPU required)
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j

# Run tests
python -m pytest decoder/tests/ -v

# Train neural decoder
python decoder/train/train.py --distance 5 --hidden_dim 256 --steps 80000

# Evaluate
python decoder/train/evaluate.py --checkpoint decoder/train/checkpoints/d5_gpu/best_model.pt
```

Dependencies: `pip install stim pymatching torch pybind11 numpy pytest`
