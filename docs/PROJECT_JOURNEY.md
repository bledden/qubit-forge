# qubit-forge: Project Journey

## Origin

This project started with an NVIDIA lecture attended by Blake Ledden (April 2026) that explained how traditional computing chips are approaching quantum mechanical limits — electrons teleporting as transistors shrink — and how quantum computers are currently bottlenecked by classical computing infrastructure for error correction. The lecturer's framing: quantum computers fail frequently, and classical high-performance computers must error-check them in real time.

Blake's background as a GPU kernel optimization engineer (48+ PRs across FlashInfer, vLLM, SGLang, Triton, CUTLASS, Flash-Attention; AMD GPU MODE Hackathon competitor with 83+ kernel iterations on MI300X) positioned him uniquely at the intersection of HPC and quantum computing — a gap where "most quantum physicists don't know cache hierarchies, and most kernel engineers don't know Hamiltonians."

A conversation with a quantum computing LLM specialist mapped out a 4-project roadmap:

1. **GPU State Vector Simulator** — learn quantum gate math by implementing it
2. **Tensor Network Contraction Engine** — contraction-as-GEMM optimization
3. **Real-Time QEC Decoder** — the high-impact, hireable deliverable
4. **Hybrid Variational Runtime** — VQE/QAOA classical optimizer loops

Projects 1 and 3 were built. Project 2 was deferred (solved problem with cuTensorNet). Project 3 was prioritized over 2 because decoders are the unsolved engineering problem.

---

## Project 1: GPU State Vector Simulator

### What It Does

Simulates quantum circuits by storing the full quantum state vector (2^n complex amplitudes) in GPU HBM and applying gate operations as matrix transformations. Targeting AMD MI300X with 192 GB HBM3.

### Build Timeline

**Day 1 (April 10-11, 2026):**

- Designed the architecture: three kernel strategies by qubit index (low=coalesced, mid=LDS tiling, high=grid-stride streaming)
- Implemented 13 tasks from scaffold to benchmarks using subagent-driven development
- Pre-flight review caught a **control/target argument swap** in the two-qubit gate kernel that would have produced wrong Bell states
- First deployment to MI300X VF instance: **25/25 tests passed** on first try after pre-flight fixes

**Key bugs found and fixed:**
- HIP silently drops kernels when `grid × block ≥ 2^32` total threads — fixed with grid-stride loops and capped grid size
- `CMAKE_BUILD_TYPE` defaulted to empty (no optimization) — the `-O0` → `-O3` fix was a **15x speedup** (0.26 → 3.9 TB/s)
- Gate initialization: HIP's clang rejected nested brace init with `__host__ __device__` constructors
- `.cpp` files receiving `--offload-arch=gfx942` from g++ — renamed to `.hip` or set `LANGUAGE HIP` in CMake

**Optimization journey (applied lessons from AMD GPU MODE Hackathon):**

| Stage | Bandwidth | Efficiency | What Changed |
|-------|-----------|------------|--------------|
| Initial (`-O0`) | 0.26 TB/s | 4.9% | No optimization flags |
| Release (`-O3`) | 3.57 TB/s | 67% | `CMAKE_BUILD_TYPE=Release` |
| Grid-stride fix | 3.98 TB/s | 75% | 33-qubit kernels working |
| `__launch_bounds__` + env vars | 3.97 TB/s | 75% | `__launch_bounds__(256, 4)`, `HIP_FORCE_DEV_KERNARG=1` |
| **Non-temporal loads** | **4.14 TB/s** | **78%** | `__builtin_nontemporal_load/store` via `double*` cast |

**STREAM ceiling analysis** proved the VF (SR-IOV virtualized) instance caps at ~3.8 TB/s for simple kernels. Our gate kernels at 4.0-4.1 TB/s **exceed the STREAM ceiling** via L2 cache reuse in grid-stride loops.

**Noise simulation** was added as a bridge to the decoder project — depolarizing, bit-flip, phase-flip, and measurement error channels enable generating realistic noisy syndrome data for QEC decoder testing.

### Benchmark Results

- **33-qubit simulation**: 128 GB state vector on single MI300X GPU
- **Peak bandwidth**: 4.14 TB/s (78% of 5.3 TB/s theoretical)
- **QFT fusion**: 10x speedup (fused vs unfused)
- **RCS scaling**: 90-93% efficiency at 25-32 qubits
- **Quantum Volume**: QV-30 in 71.5 seconds
- **Total MI300X cost**: ~$10

### Hackathon Lessons Applied

From the AMD GPU MODE E2E Model Speedrun competition (March-April 2026, $1.1M prize pool):

- `__launch_bounds__(256, 4)` — 4 waves/SIMD is the sweet spot for bandwidth-bound kernels; higher occupancy causes L2 cache thrashing
- Non-temporal loads/stores — `__builtin_nontemporal_load/store` for streaming access patterns (+8% bandwidth)
- Environment variables — `HIP_FORCE_DEV_KERNARG=1`, `AMD_DIRECT_DISPATCH=1` (free 0.5-2μs per kernel launch)
- Roofline-first methodology — calculate bandwidth floor before optimizing; if within 1.3x of floor, you're near the wall
- Per-shape profiling — different qubit counts need different kernel strategies

---

## Project 3: QEC Decoder Suite

### What It Does

Decodes quantum error correction syndromes — the core bottleneck for fault-tolerant quantum computing. Takes noisy parity-check measurements from a surface code and determines the most likely error pattern. Must run faster than errors accumulate (~1μs for superconducting qubits).

### Three-Tier Architecture

1. **Union-Find (C++)** — near-linear time CPU decoder, correctness baseline
2. **Belief Propagation (Python)** — min-sum message passing prototype
3. **Neural CNN (PyTorch)** — convolutional neural network following Gu et al. [1]

### Build Timeline

**Day 2 (April 12, 2026):**

- Integrated Stim [2] for syndrome generation (~1B Clifford gates/sec)
- Built C++ Union-Find decoder with weighted growth/merge/peel algorithm [3]
- **Critical pybind11 bug**: `def_readwrite` for `std::vector` returns a copy — `sg.edges.append(e)` was appending to a temporary copy. The decoder ran with an **empty graph** until this was fixed. Solution: `add_edge()` method that modifies the C++ vector directly.
- Python BP decoder prototype validated min-sum message passing

**Day 2-3 (April 12-13):**

- Designed neural CNN decoder architecture following Gu et al. [1] (arXiv:2604.08358, published April 9, 2026 — 3 days before our implementation)
- Key architecture: `DirectionalConv3d` — 7 separate weight matrices per neighbor direction in the 3D syndrome lattice, NOT standard `nn.Conv3d` which shares weights across directions
- First training on CPU (Apple M4 Mac): **d=3 model beats PyMatching at ALL noise rates** after 65 minutes of training

**Day 3-4 (April 13-14):**

- Deployed to MI300X for GPU training
- **Muon optimizer discovery**: PyTorch 2.5.1+ROCm didn't include `torch.optim.Muon` (added in 2.9). The standalone `muon-pytorch` package required distributed init (`torch.distributed`). Solution: found `SingleDeviceMuonWithAuxAdam` class already in the package — handles both Muon and Adam params without distributed.
- **Muon vs AdamW at d=7**: 2.5x improvement in accuracy (4.10% → 1.71% LER). This was the single most impactful optimization in the entire decoder project.
- **Compressed curriculum**: Original 3-stage curriculum wasted 20% of training on trivially easy examples (p=0.0007). Compressed to start at p=0.002, saving significant training time.
- **Decoder steering** (experimental): Implemented RL-inspired adaptive edge weights from Sivak et al. [4]. Simple EMA approach didn't improve accuracy — needs per-edge correlation tracking for meaningful impact.

### The Neural Decoder Results

**d=3 (trained on CPU, 65 min, 252K params):**

Beats PyMatching (MWPM) at every noise rate tested.

| p | Neural | PyMatching | vs PM |
|---|--------|-----------|-------|
| 0.001 | 0.038% | 0.090% | **2.4x better** |
| 0.005 | 0.996% | 1.236% | **1.2x better** |
| 0.007 | 1.770% | 2.224% | **1.3x better** |
| 0.010 | 3.648% | 4.142% | **1.1x better** |

**d=5 (AdamW, MI300X GPU, 87 min, 376K params):**

Beats PyMatching at low-to-mid noise.

| p | Neural | PyMatching | vs PM |
|---|--------|-----------|-------|
| 0.002 | 0.036% | 0.084% | **2.3x better** |
| 0.005 | 0.802% | 0.844% | **1.05x better** |
| 0.007 | 2.19% | 2.05% | 1.07x worse |

**d=7 (Muon optimizer, MI300X GPU, 5.7 hours, 500K params):**

Beats PyMatching at low-to-mid noise. Within 5% at p=0.007.

| p | Neural | PyMatching | vs PM |
|---|--------|-----------|-------|
| 0.002 | 0.003% | 0.013% | **4x better** |
| 0.005 | 0.370% | 0.453% | **1.2x better** |
| 0.007 | 1.66% | 1.56% | 1.06x worse |

**d=5 and d=7 retraining with Muon + target p=0.01 is running overnight to close the remaining gaps.**

### Optimizer Impact: Muon vs AdamW

The switch from AdamW to Muon (Newton-Schulz orthogonalization) was the most impactful single change:

| Distance | AdamW Best | Muon Best | Improvement |
|----------|-----------|-----------|-------------|
| d=7 p=0.005 | 1.19% | 0.37% | **3.2x** |
| d=7 p=0.007 | 4.12% | 1.66% | **2.5x** |
| d=7 p=0.010 | 12.2% | 6.91% | **1.8x** |

Muon's Newton-Schulz iteration orthogonalizes weight matrix updates, preventing the kind of weight degeneration that standard optimizers like Adam allow. This is particularly effective for the directional convolution weights in our architecture, where each direction-specific weight matrix benefits from staying well-conditioned.

---

## Key Technical Decisions

### Why DirectionalConv3d Instead of Standard Conv3d

The surface code syndrome lattice has specific geometric structure: each stabilizer has exactly 4 data-qubit neighbors in specific spatial directions, plus temporal neighbors from consecutive measurement rounds. Standard `nn.Conv3d` uses the same 3×3×3 kernel everywhere — it can't distinguish between "this neighbor is to the left" and "this neighbor is in the future."

`DirectionalConv3d` uses 7 separate `nn.Linear` weight matrices — one for self-connection and one for each of the 6 neighbor directions (±t, ±row, ±col). This preserves the lattice structure that standard convolution would blur.

### Why Stim for Data Generation

Stim [2] (Google's stabilizer circuit simulator) generates detection events at ~1 billion Clifford gates per second. Training data is generated on-the-fly — no pre-generated datasets needed. This eliminates the data pipeline bottleneck that would otherwise dominate training time.

Stim also provides exact detector coordinates via `circuit.get_detector_coordinates()`, which we use for the spatial mapping from flat detector indices to 3D tensor positions.

### Why Muon Over AdamW

The Gu et al. paper specified "Muon (Newton-Schulz orthogonalization)" but didn't explain why it matters for decoder training specifically. Our experiments showed: Muon produces **2.5x better accuracy** at d=7 compared to AdamW. The orthogonalization step keeps the directional convolution weight matrices well-conditioned, which is critical for the message-passing interpretation of the CNN — each layer propagates information through the syndrome graph, and degenerate weights would corrupt the message-passing dynamics.

### Why Compressed Curriculum

The original 3-stage curriculum (Gu et al.) spends 20% of training at very low noise (p_target × 0.1). At low noise, syndromes are almost always trivial (no defects), so the model learns nothing useful. Our compressed curriculum starts at p_target × 0.3 and reaches target noise by 40% through training (vs 60% in the original). This gave the model 50% more steps of useful training at challenging noise levels.

---

## Infrastructure

### Conditional Build System

The CMakeLists.txt auto-detects ROCm/HIP: builds the full GPU simulator when HIP is available, builds the decoder-only (pure CPU C++) when it isn't. This allows development on any platform (Mac, Linux) without requiring a GPU.

### GPU Instances Used

| Instance | Provider | Type | GPU | Cost | Used For |
|----------|----------|------|-----|------|----------|
| #1 | DigitalOcean | VF (virtualized) | MI300X | ~$10 | Simulator benchmarks |
| #2 | DigitalOcean | VF (virtualized) | MI300X | ~$10 | d=5 GPU training, d=7 v1 training |
| #3 | Shadeform | VF | MI300X | ~$15 est. | d=7 Muon training, d=5/d=7 final |

Total GPU cost: ~$35

### Repository Structure

```
qubit-forge/
├── src/kernels/           # HIP gate-application kernels (simulator)
├── include/quantum/       # Simulator headers
├── decoder/
│   ├── src/               # C++ Union-Find decoder
│   ├── train/             # PyTorch neural decoder
│   │   ├── model.py       # DirectionalConv3d + BottleneckBlock architecture
│   │   ├── data.py        # Stim data pipeline + curriculum
│   │   ├── train.py       # Training loop with Muon
│   │   └── evaluate.py    # Multi-decoder comparison
│   ├── python/            # Stim interface + pybind11 bindings
│   └── tests/             # 20+ decoder tests
├── bench/                 # Simulator benchmarks (RCS, QV, scaling)
└── docs/                  # Specs, plans, this document
```

---

## References

[1] Gu, A., Bonilla Ataides, J.P., Lukin, M.D., & Yelin, S.F. (2026). "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation." arXiv:2604.08358. — CNN decoder architecture, waterfall regime, 17x improvement over BP+OSD.

[2] Gidney, C. (2021). "Stim: a fast stabilizer circuit simulator." Quantum 5, 497. — Stabilizer simulation at ~1B gates/sec, used for syndrome data generation.

[3] Delfosse, N. & Nickerson, N. (2021). "Almost-linear time decoding algorithm for topological codes." Quantum 5, 595. — Union-Find decoder algorithm (growth, merge, peel).

[4] Sivak, V.V., et al. (2025). "Reinforcement Learning Control of Quantum Error Correction." arXiv:2511.08493. — RL-based decoder steering concept, 3.5x stability improvement on Willow processor.

[5] Fowler, A.G., Mariantoni, M., Martinis, J.M., & Cleland, A.N. (2012). "Surface codes: Towards practical large-scale quantum computation." Physical Review A 86, 032324. — The surface code reference.

[6] Google Quantum AI (2024). "Quantum error correction below the surface code threshold." Nature 636, 923-929. — Willow chip, first demonstration of exponential error suppression with code distance.

[7] Higgott, O. & Gidney, C. (2023). "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." arXiv:2303.15933. — PyMatching/Sparse Blossom decoder (our comparison baseline).

---

## What's Next

- **Overnight training results** (d=5 Muon + d=7 at target p=0.01) — targeting PyMatching-beating performance across all noise rates at all distances
- **GPU inference kernel** — HIP/CUDA kernel for neural decoder inference, targeting <50μs latency
- **FP8 quantization** — Gu et al. showed no accuracy loss at FP8; MI300X has MFMA FP8 instructions
- **Bare-metal bandwidth optimization** — push simulator from 78% (VF ceiling) to 85-90% on bare-metal MI300X
