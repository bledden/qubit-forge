# GPU State Vector Quantum Simulator — Design Spec

**Date:** 2026-04-10
**Author:** Blake Ledden
**Target Hardware:** AMD MI300X (192GB HBM3, 5.3 TB/s, ROCm/HIP)
**Goal:** Build a from-scratch GPU-accelerated quantum state vector simulator that benchmarks MI300X against published cuQuantum/A100/H100 numbers, while learning quantum gate math by implementing it.

## 1. Computational Model

A quantum state of `n` qubits is a vector of `2^n` complex amplitudes stored in GPU HBM. Gates are applied in-place.

**Single-qubit gate on qubit k:** For each index `i` where bit `k` is 0, load the pair `(state[i], state[i + 2^k])`, multiply by a 2x2 complex unitary matrix, write back. The stride between coupled amplitudes is `2^k`.

**Two-qubit gate on qubits (c, t):** For each index where bits `c` and `t` are both 0, load the group of 4 amplitudes at offsets `{0, 2^t, 2^c, 2^t + 2^c}`, multiply by a 4x4 unitary, write back.

**Memory requirements:**
| Qubits | Amplitudes | complex128 | complex64 |
|--------|-----------|------------|-----------|
| 25     | 33M       | 512 MB     | 256 MB    |
| 28     | 268M      | 4 GB       | 2 GB      |
| 30     | 1B        | 16 GB      | 8 GB      |
| 32     | 4B        | 64 GB      | 32 GB     |
| 33     | 8B        | 128 GB     | 64 GB     |
| 34     | 16B       | —          | 128 GB    |

MI300X (192GB) supports: 33 qubits at complex128, 34 qubits at complex64.

## 2. Project Structure

```
quantum/
├── src/
│   ├── kernels/
│   │   ├── single_qubit.hip    # 1-qubit gate: 3 strategies by qubit index
│   │   ├── two_qubit.hip       # 2-qubit gate: CNOT, CZ, arbitrary
│   │   ├── diagonal.hip        # Diagonal gates (Rz, Phase, T, S)
│   │   └── measure.hip         # Measurement sampling
│   ├── statevec.hip            # State vector alloc/init/destroy
│   ├── fusion.cpp              # Gate fusion engine (CPU-side DAG)
│   ├── circuit.cpp             # Circuit IR and gate queue
│   └── gates.h                 # Gate definitions (2x2/4x4 unitaries)
├── include/
│   └── quantum/
│       ├── statevec.h
│       ├── circuit.h
│       ├── fusion.h
│       └── types.h             # complex types, gate enum
├── python/
│   └── pyquantum.cpp           # pybind11 interface
├── bench/
│   ├── single_gate.py          # Per-gate bandwidth benchmark
│   ├── circuit_bench.py        # QFT, Grover, random circuits
│   └── scaling.py              # 20->33 qubit scaling curves
├── tests/
│   └── test_gates.py           # Correctness vs analytic/numpy
├── CMakeLists.txt
└── README.md
```

## 3. Kernel Design

### 3.1 Single-Qubit Gate — Three Strategies

The stride `2^k` determines memory access pattern. Three kernel variants selected at dispatch time:

**Low qubits (k < 5):**
- Paired amplitudes are within 32 elements of each other (within a wavefront)
- One thread per pair, contiguous workgroup access
- Gate matrix loaded into registers (4 complex values = 64 bytes)
- Expected: >85% of peak HBM bandwidth (5.3 TB/s)

**Mid qubits (5 <= k < 10):**
- Load tiles of `2^(k+1)` contiguous elements into LDS (64KB per CU on MI300X)
- Apply gate using LDS-local stride — fast random access
- Coalesced global loads and stores
- Expected: >70% of peak HBM bandwidth

**High qubits (k >= 10):**
- Pairs are >= 16KB apart in memory — no shared-memory tiling possible
- Each workgroup processes multiple pairs, issuing two coalesced reads from distant memory regions
- Prefetch via `__builtin_amdgcn_s_prefetch_data` where beneficial
- Gate fusion is critical to amortize the two-region access pattern
- Expected: ~50-60% of peak bandwidth (memory latency limited)

### 3.2 Two-Qubit Gate

- Identify groups of 4 amplitudes by masking bits `c` and `t`
- Strategy depends on `min(c, t)` and `|c - t|`:
  - Both low: coalesced access, one thread per group
  - One high: hybrid — coalesced for one qubit, strided for the other
  - Both high: dual-region access, fusion critical
- The 4x4 unitary is loaded into registers (32 complex values)

### 3.3 Diagonal Gates

- Rz, Phase, T, S gates only modify phase: `state[i] *= exp(i * angle)` when bit k is 1
- No cross-amplitude coupling — embarrassingly parallel
- Batch multiple diagonal gates into a single kernel with a phase lookup table indexed by relevant bits
- Expected: near-peak bandwidth (simple element-wise multiply)

### 3.4 Measurement

- **Probability computation:** Reduce `|state[i]|^2` across all amplitudes using hipCUB block-reduce
- **Single-shot:** Binary search on cumulative distribution
- **Multi-shot (batched):** Compute full probability vector once, generate `shots` random indices via GPU RNG (hipRAND), parallel binary search

## 4. Gate Fusion Engine

CPU-side preprocessing before kernel dispatch. Three passes:

**Pass 1 — Same-qubit fusion:**
Scan gate queue. Consecutive single-qubit gates on the same target qubit: multiply their 2x2 matrices. `Rz(0.3) → Rx(0.5) → Rz(0.7)` becomes a single 2x2 unitary. Cost: one matrix multiply per fusion (negligible).

**Pass 2 — Layer extraction:**
Build DAG from gate queue (edges = qubit-wire dependencies). Topological sort, group gates into layers of non-overlapping qubit sets. Each layer dispatches as a single kernel pass over the state vector, applying all gates in the layer.

**Pass 3 — Diagonal batching:**
Within a layer, separate diagonal gates from non-diagonal. All diagonal gates in a layer merge into a single phase-table kernel.

## 5. Standard Gate Library

Built-in gates with precomputed 2x2/4x4 matrices:

| Gate | Type | Matrix |
|------|------|--------|
| H (Hadamard) | Single, dense | `1/sqrt(2) * [[1,1],[1,-1]]` |
| X (Pauli-X) | Single, permutation | `[[0,1],[1,0]]` |
| Y (Pauli-Y) | Single, dense | `[[0,-i],[i,0]]` |
| Z (Pauli-Z) | Single, diagonal | `[[1,0],[0,-1]]` |
| Rx(theta) | Single, dense | `[[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]` |
| Ry(theta) | Single, dense | `[[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]` |
| Rz(theta) | Single, diagonal | `[[exp(-it/2), 0], [0, exp(it/2)]]` |
| S, T | Single, diagonal | Phase gates |
| CNOT | Two-qubit | Permutation of |10⟩ ↔ |11⟩ |
| CZ | Two-qubit, diagonal | Phase flip on |11⟩ |
| SWAP | Two-qubit | Permutation |
| Toffoli | Three-qubit | CCNOT (decomposed into 1- and 2-qubit gates) |

## 6. Python Interface

```python
import pyquantum as pq

# Create state vector on GPU
sv = pq.StateVector(n_qubits=30, dtype="complex128")  # 16 GB on MI300X

# Apply individual gates
sv.h(0)           # Hadamard on qubit 0
sv.cx(0, 1)       # CNOT: control=0, target=1
sv.rz(0.5, 3)     # Rz(0.5) on qubit 3

# Build and run a fused circuit
circ = pq.Circuit(n_qubits=30)
circ.h(0)
circ.cx(0, 1)
circ.rz(0.5, 3)
sv.apply_circuit(circ)  # Fused execution

# Measure
probs = sv.probabilities()       # Full probability vector
samples = sv.measure(shots=1024) # Batched sampling
amp = sv.amplitude(0b101)        # Single amplitude query
```

No external quantum framework dependencies. Standalone.

## 7. Benchmarks

### 7.1 Per-Gate Bandwidth

For each qubit index k (0 through n-1), at qubit counts 20, 25, 28, 30, 32, 33:
- Apply a single Hadamard gate
- Measure wall-clock time
- Compute achieved bandwidth: `(2^n * 16 * 3) / time` (read pair + write pair = 3x state size)
- Report as percentage of MI300X theoretical peak (5.3 TB/s)

Produces a heatmap: qubit index vs. qubit count vs. bandwidth efficiency.

### 7.2 Standard Circuits

| Circuit | Qubits | What It Tests |
|---------|--------|---------------|
| QFT | 20, 25, 30, 33 | Mixed H + controlled-phase, deep circuit |
| Grover (single marked) | 20, 25, 30 | Oracle + diffusion, repeated iterations |
| Random (depth 20) | 20, 25, 30, 33 | Worst-case fusion, random gate placement |
| Bell/GHZ | 10, 20, 30 | CNOT chain, entanglement |

Compare against published cuQuantum numbers on A100-80GB and H100-80GB.

### 7.3 MI300X Headline

- **33-qubit QFT** on a single GPU — requires 128GB HBM, which rules out A100-80GB and H100-80GB (the GPUs with the most published cuQuantum benchmarks). H200 (141GB) and B200 (192GB) can also fit 33 qubits but have less published simulation data.
- **34-qubit** at complex64 if accuracy is acceptable (128GB, fits on MI300X/H200/B200)
- Report total time, per-gate breakdown, and bandwidth utilization
- **Bandwidth comparison**: MI300X (5.3 TB/s) vs H200 (4.8 TB/s) vs H100 (3.35 TB/s) vs A100 (2.0 TB/s) — this workload is purely memory-bandwidth bound, so these ratios predict relative performance directly

### 7.4 Profiling

- `rocprof` for kernel timing and HW counters
- `omniperf` for occupancy, LDS bank conflicts, memory channel utilization
- Compare achieved vs. roofline model

## 8. Testing Strategy

Correctness tests at small qubit counts (3-10 qubits) where numpy can compute exact results:

1. **Bell state:** H(0) + CNOT(0,1) → verify amplitudes [0.707, 0, 0, 0.707]
2. **GHZ state:** H(0) + CNOT chain → verify equal |00...0⟩ and |11...1⟩
3. **QFT:** Compare output amplitudes against `numpy.fft`
4. **Grover:** Verify marked state has >90% probability after optimal iterations
5. **Gate identities:** HH=I, XX=I, CNOT*CNOT=I, verify state unchanged
6. **Random circuits:** Generate small random circuits, compare GPU result against numpy matrix multiplication (construct full 2^n × 2^n unitary, multiply)

Tolerance: `|gpu - numpy| < 1e-10` for complex128, `< 1e-5` for complex64.

## 9. Build System

CMake with ROCm/HIP toolchain:
- `hipcc` for `.hip` files
- Standard C++17 for CPU code
- pybind11 for Python bindings
- No other dependencies

Build targets:
- `libquantum.so` — core library
- `pyquantum` — Python module
- `test_quantum` — C++ unit tests
- `bench_quantum` — C++ microbenchmarks

## 10. Non-Goals (Explicit Exclusions)

- Multi-GPU / distributed simulation (future project)
- Noise simulation / density matrix
- Qiskit/Cirq backend compatibility (can be added as a thin adapter later)
- Tensor network simulation (separate project B)
- NVIDIA/CUDA support (HIP is source-compatible; trivial to port later)
- Visualization or UI
