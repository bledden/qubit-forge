# Quantum State Vector Simulator

GPU-accelerated quantum circuit simulator built from scratch in HIP/ROCm, targeting AMD MI300X.

**33-qubit simulation on a single GPU** (128 GB state vector) — exceeds the capacity of A100-80GB and H100-80GB.

## Performance

Benchmarked on AMD Instinct MI300X (192 GB HBM3, 5.3 TB/s theoretical peak):

### Per-Gate Bandwidth

| Qubits | State Vector | Best BW (TB/s) | Peak Efficiency |
|--------|-------------|----------------|-----------------|
| 25     | 0.5 GB      | 3.78           | 71%             |
| 28     | 4.3 GB      | 3.98           | 75%             |
| 30     | 17.2 GB     | 3.89           | 73%             |
| 32     | 68.7 GB     | 3.96           | 75%             |
| 33     | 128 GB      | 4.02           | 76%             |

### Circuit Benchmarks

| Circuit         | Qubits | Gates | Unfused   | Fused     | Speedup |
|-----------------|--------|-------|-----------|-----------|---------|
| QFT             | 25     | 337   | 98.5 ms   | 11.0 ms   | 9.0x    |
| QFT             | 30     | 480   | 4,395 ms  | 435 ms    | 10.1x   |
| Random (d=20)   | 28     | 840   | 2,155 ms  | 2,146 ms  | 1.0x    |
| GHZ             | 30     | 30    | 292 ms    | —         | —       |

### Random Circuit Sampling (Sycamore-style)

1D chain topology, alternating CZ layers, depth 20:

| Qubits | Gates | Time      | BW (TB/s) | 2x Scaling |
|--------|-------|-----------|-----------|------------|
| 20     | 590   | 15.8 ms   | 1.26      | —          |
| 25     | 740   | 249 ms    | 3.19      | 15.8x (exp 32x) |
| 28     | 830   | 2,203 ms  | 3.24      | 8.9x (exp 8x)   |
| 30     | 890   | 9,448 ms  | 3.24      | 4.3x (exp 4x)   |
| 32     | 950   | 46,376 ms | 2.82      | 4.9x (exp 4x)   |

Scaling efficiency 90-93% at 25-32 qubits.

### Quantum Volume

| Width | SU(4) Blocks | Gates | Time      |
|-------|-------------|-------|-----------|
| 20    | 200         | 3,000 | 572 ms    |
| 25    | 300         | 4,500 | 10.3 s    |
| 28    | 392         | 5,880 | 15.4 s    |
| 30    | 450         | 6,750 | 75.2 s    |

### Qubit Scaling (H on all qubits + GHZ state)

| Qubits | Memory  | H_all     | GHZ       |
|--------|---------|-----------|-----------|
| 25     | 0.5 GB  | 7.5 ms    | 7.6 ms    |
| 28     | 4.3 GB  | 64.9 ms   | 65.6 ms   |
| 30     | 17.2 GB | 279.5 ms  | 290.2 ms  |
| 32     | 68.7 GB | 1,205 ms  | 1,198 ms  |
| 33     | 128 GB  | ~2,300 ms | ~2,340 ms |

## Architecture

```
include/quantum/
  types.h        Complex128, Gate1Q, Gate2Q
  gates.h        H, X, Y, Z, Rx, Ry, Rz, S, T, CNOT, CZ, SWAP
  statevec.h     StateVector class
  circuit.h      Circuit IR + GateOp
  fusion.h       Gate fusion engine
  kernels.h      Kernel launch declarations

src/kernels/
  single_qubit.hip   3 strategies: low (coalesced), mid (LDS tiling), high (grid-stride)
  two_qubit.hip      4-amplitude group kernel with grid-stride loop
  diagonal.hip       Phase-only gates (Rz, S, T, Z)
  measure.hip        GPU probability computation + reduction
```

### Kernel Strategies

Single-qubit gate application couples pairs of amplitudes at stride `2^k` (where k = target qubit index):

- **Low (k < 5):** Adjacent amplitudes, coalesced global memory access
- **Mid (5 <= k < 11):** Tiles loaded into LDS (Local Data Share), gate applied with local stride
- **High (k >= 11):** Grid-stride loop over distant amplitude pairs

All kernels use grid-stride loops to handle state vectors exceeding 2^32 total threads (HIP limitation).

### Gate Fusion

CPU-side preprocessing before kernel dispatch:
1. **Same-qubit fusion:** Consecutive single-qubit gates on the same target are multiplied into a single 2x2 unitary
2. **Layer extraction:** Non-overlapping gates grouped into parallel layers, reducing synchronization points

Achieves 10x speedup on QFT circuits (heavy in consecutive rotations).

## Build

Requirements: ROCm 6.x+, CMake 3.21+, pybind11, numpy

```bash
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j$(nproc)
```

Defaults to Release build (`-O3`) targeting MI300X (`gfx942`). Override architecture:
```bash
cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx90a  # MI250X
```

## Python API

```python
import pyquantum as pq

# 30-qubit state vector (16 GB on GPU)
sv = pq.StateVector(30)

# Apply gates
sv.h(0)          # Hadamard
sv.cx(0, 1)      # CNOT
sv.rz(0.5, 3)    # Rotation

# Build and run circuits with fusion
circ = pq.Circuit(30)
circ.h(0)
circ.cx(0, 1)
sv.apply_circuit_fused(circ)

# Measure
probs = sv.probabilities()       # numpy array
samples = sv.measure(shots=1024) # sampled basis states
```

## Tests

```bash
./build/test_statevec                          # C++ tests
PYTHONPATH=build python -m pytest tests/ -v    # 25 Python tests
```

Correctness verified against numpy at small qubit counts: Bell states, GHZ states, gate identities, QFT vs numpy FFT, Grover's search.

## Benchmarks

```bash
PYTHONPATH=build python bench/single_gate.py     # Per-gate bandwidth
PYTHONPATH=build python bench/circuit_bench.py   # Circuit timing
PYTHONPATH=build python bench/rcs_bench.py       # Random Circuit Sampling
PYTHONPATH=build python bench/quantum_volume.py  # Quantum Volume
PYTHONPATH=build python bench/scaling.py         # Qubit scaling
```

## Hardware Notes

- **MI300X (192 GB):** 33 qubits at complex128, 34 at complex64
- **H200 (141 GB):** 33 qubits at complex128
- **H100 (80 GB):** 30 qubits at complex128
- **A100 (80 GB):** 30 qubits at complex128

Bandwidth comparison (this workload is purely memory-bandwidth bound):

| GPU    | Peak BW   | Achieved BW | Efficiency |
|--------|-----------|-------------|------------|
| MI300X | 5.3 TB/s  | 3.7-4.0 TB/s | 70-76%    |
| H200   | 4.8 TB/s  | —           | —          |
| H100   | 3.35 TB/s | —           | —          |
| A100   | 2.0 TB/s  | —           | —          |
