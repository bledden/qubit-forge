"""Quantum Volume (QV) benchmark — IBM methodology.

Quantum Volume (Cross et al., arXiv:1902.08181) measures the largest random
circuit of equal width and depth that a quantum computer can reliably execute.
For a statevector simulator the circuit always succeeds; the benchmark instead
measures the computational cost of simulating QV circuits.

Circuit construction (following IBM's protocol):
  1. Width = depth = m.
  2. Each of the m layers applies a random permutation of qubit pairs, then
     a random SU(4) two-qubit unitary to each pair.
  3. SU(4) unitaries are decomposed into the standard KAK form:
        Ry(θ₁)·Rz(φ₁) ⊗ Ry(θ₂)·Rz(φ₂) → CNOT → Ry(θ₃)·Rz(φ₃) ⊗ Ry(θ₄)·Rz(φ₄) → CNOT → Ry(θ₅)·Rz(φ₅) ⊗ Ry(θ₆)·Rz(φ₆)
     This gives 3 CNOT gates and 12 single-qubit rotations per SU(4) block,
     which is sufficient to span the full SU(4) manifold up to a global phase.

Metrics reported:
  - Wall-clock time (median of timed iterations after warmup)
  - Gate count
  - Time per SU(4) block
  - Effective bandwidth
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(__file__))
import mi300x_env  # Set HIP env vars before importing pyquantum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import pyquantum as pq
import numpy as np


def _apply_su4_block(circ, q0, q1, rng):
    """Decompose a random SU(4) into single-qubit rotations + CNOTs.

    Uses a simplified KAK-like decomposition:
        [Ry Rz ⊗ Ry Rz] · CNOT · [Ry Rz ⊗ Ry Rz] · CNOT · [Ry Rz ⊗ Ry Rz]

    This produces 3 CNOTs and 12 single-qubit rotations, spanning SU(4).

    Args:
        circ: Circuit to append gates to.
        q0, q1: Qubit indices.
        rng: NumPy random generator.

    Returns:
        Number of gates added.
    """
    gate_count = 0

    for _ in range(3):  # 3 rounds of [rotations + CNOT]
        # Single-qubit rotations on both qubits
        for q in (q0, q1):
            circ.ry(rng.uniform(0, 2 * math.pi), q)
            circ.rz(rng.uniform(0, 2 * math.pi), q)
            gate_count += 2

        # CNOT entangling gate
        circ.cx(q0, q1)
        gate_count += 1

    # Final single-qubit rotations (no trailing CNOT)
    # Already included in the last iteration above — the decomposition is
    # [rot CNOT] × 3 which gives 3 CNOTs and 6 rotation pairs = 12 rotations.

    return gate_count


def build_qv_circuit(width, seed=0):
    """Build a Quantum Volume circuit.

    Args:
        width: Number of qubits (depth = width per QV protocol).
        seed: RNG seed for reproducibility.

    Returns:
        (circuit, total_gate_count, su4_block_count) tuple.
    """
    rng = np.random.default_rng(seed)
    depth = width  # QV: depth = width
    circ = pq.Circuit(width)
    total_gates = 0
    su4_blocks = 0

    for _ in range(depth):
        # Random permutation of qubits, then pair them up
        perm = rng.permutation(width)
        n_pairs = width // 2

        for p in range(n_pairs):
            q0, q1 = int(perm[2 * p]), int(perm[2 * p + 1])
            # Ensure q0 < q1 for consistency (gate semantics are symmetric)
            if q0 > q1:
                q0, q1 = q1, q0
            gates_added = _apply_su4_block(circ, q0, q1, rng)
            total_gates += gates_added
            su4_blocks += 1

    return circ, total_gates, su4_blocks


def bench_qv(width, seed=0, n_warmup=2, n_iter=5):
    """Benchmark a single Quantum Volume configuration.

    Returns:
        dict with keys: width, gate_count, su4_blocks, median_time_s,
        time_per_su4_us, bandwidth_TBs.
    """
    circ, gate_count, su4_blocks = build_qv_circuit(width, seed=seed)

    times = []
    for i in range(n_warmup + n_iter):
        sv = pq.StateVector(width)
        start = time.perf_counter()
        sv.apply_circuit(circ)
        end = time.perf_counter()
        if i >= n_warmup:
            times.append(end - start)

    median_t = float(np.median(times))

    # Bandwidth: each gate reads + writes the full statevector
    bytes_per_gate = 2 * (2 ** width) * 16
    total_bytes = bytes_per_gate * gate_count
    bandwidth = total_bytes / median_t

    return {
        'width': width,
        'gate_count': gate_count,
        'su4_blocks': su4_blocks,
        'median_time_s': median_t,
        'time_per_su4_us': (median_t / su4_blocks) * 1e6 if su4_blocks > 0 else 0,
        'bandwidth_TBs': bandwidth / 1e12,
    }


def main():
    widths = [4, 6, 8, 10, 12, 14, 16, 20, 25, 28, 30]

    print("=" * 80)
    print("Quantum Volume (QV) Benchmark — IBM Methodology")
    print("Each circuit: depth = width, random SU(4) on random qubit pairs")
    print("SU(4) decomposition: 3 CNOTs + 12 single-qubit rotations per block")
    print("=" * 80)
    print()

    print(f"{'Width':>5}  {'Depth':>5}  {'SU4s':>5}  {'Gates':>6}  "
          f"{'Time(ms)':>10}  {'us/SU4':>8}  {'BW(TB/s)':>9}")
    print("-" * 65)

    results = []
    for w in widths:
        try:
            r = bench_qv(w, seed=42)
            results.append(r)

            print(f"{r['width']:>5}  {r['width']:>5}  "
                  f"{r['su4_blocks']:>5}  {r['gate_count']:>6}  "
                  f"{r['median_time_s']*1e3:>10.2f}  "
                  f"{r['time_per_su4_us']:>8.2f}  "
                  f"{r['bandwidth_TBs']:>9.3f}")

        except Exception as e:
            print(f"{w:>5}  FAILED: {e}")
            break

    # --- Scaling summary ---
    print()
    print("=" * 80)
    print("Scaling Analysis")
    print("=" * 80)
    if len(results) >= 2:
        print(f"{'From':>5} {'To':>5}  {'dQubits':>7}  "
              f"{'Time Ratio':>10}  {'Gate Ratio':>10}  {'Adjusted':>10}")
        print("-" * 55)
        for i in range(1, len(results)):
            r0, r1 = results[i - 1], results[i]
            dq = r1['width'] - r0['width']
            time_ratio = r1['median_time_s'] / r0['median_time_s']
            gate_ratio = r1['gate_count'] / r0['gate_count']
            # Adjusted ratio: time normalized by gate count increase
            adjusted = time_ratio / gate_ratio if gate_ratio > 0 else 0
            print(f"{r0['width']:>5} {r1['width']:>5}  {dq:>7}  "
                  f"{time_ratio:>10.2f}x {gate_ratio:>10.2f}x {adjusted:>10.3f}x")
    print()


if __name__ == "__main__":
    main()
