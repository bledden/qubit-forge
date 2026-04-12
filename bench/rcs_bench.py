"""Random Circuit Sampling (RCS) benchmark — Sycamore/quantum-supremacy methodology.

This benchmark follows the structure used in Google's 2019 quantum supremacy
experiment (Arute et al., Nature 574, 505–510):

  1. Each cycle consists of a layer of random single-qubit gates drawn uniformly
     from {H, sqrt(X) ≈ Rx(π/2), sqrt(Y) ≈ Ry(π/2)} applied to every qubit,
     followed by a layer of CZ gates on alternating nearest-neighbor pairs
     (even-odd on even cycles, odd-even on odd cycles) in a 1D chain topology.

  2. The full circuit is: (single-qubit layer + entangling layer) × depth.

  3. We sweep over qubit counts and depths, measuring wall-clock time, time per
     gate, and effective HBM bandwidth (read + write of the full statevector
     per gate application).

  4. We verify the expected ~2× slowdown per additional qubit (statevector
     doubles in size).

Metrics reported:
  - Wall-clock time (median of timed iterations after warmup)
  - Time per gate (wall-clock / total gate count)
  - Effective bandwidth (bytes moved / wall-clock time)
  - Scaling ratio vs. previous qubit count
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


def build_rcs_circuit(n_qubits, depth, seed=0):
    """Build a Random Circuit Sampling circuit on a 1D chain.

    Args:
        n_qubits: Number of qubits.
        depth: Number of cycles (each cycle = single-qubit layer + CZ layer).
        seed: RNG seed for reproducibility.

    Returns:
        (circuit, gate_count) tuple.
    """
    rng = np.random.default_rng(seed)
    circ = pq.Circuit(n_qubits)
    gate_count = 0

    for d in range(depth):
        # --- Single-qubit layer: random gate from {H, sqrt(X), sqrt(Y)} ---
        for q in range(n_qubits):
            choice = rng.integers(0, 3)
            if choice == 0:
                circ.h(q)
            elif choice == 1:
                circ.rx(math.pi / 2, q)   # sqrt(X)
            else:
                circ.ry(math.pi / 2, q)   # sqrt(Y)
            gate_count += 1

        # --- Entangling layer: CZ on alternating pairs ---
        # Even cycles: pairs (0,1), (2,3), ...
        # Odd cycles:  pairs (1,2), (3,4), ...
        start = 0 if d % 2 == 0 else 1
        for q in range(start, n_qubits - 1, 2):
            circ.cz(q, q + 1)
            gate_count += 1

    return circ, gate_count


def bench_rcs(n_qubits, depth, seed=0, n_warmup=2, n_iter=5):
    """Benchmark a single RCS configuration.

    Returns:
        dict with keys: n_qubits, depth, gate_count, median_time_s,
        time_per_gate_us, bandwidth_TBs.
    """
    circ, gate_count = build_rcs_circuit(n_qubits, depth, seed=seed)

    times = []
    for i in range(n_warmup + n_iter):
        sv = pq.StateVector(n_qubits)
        start = time.perf_counter()
        sv.apply_circuit(circ)
        end = time.perf_counter()
        if i >= n_warmup:
            times.append(end - start)

    median_t = float(np.median(times))

    # Each gate touches the full statevector: read + write = 2 × 2^n × 16 bytes
    bytes_per_gate = 2 * (2 ** n_qubits) * 16
    total_bytes = bytes_per_gate * gate_count
    bandwidth = total_bytes / median_t

    return {
        'n_qubits': n_qubits,
        'depth': depth,
        'gate_count': gate_count,
        'median_time_s': median_t,
        'time_per_gate_us': (median_t / gate_count) * 1e6,
        'bandwidth_TBs': bandwidth / 1e12,
    }


def main():
    qubit_counts = [10, 15, 20, 25, 28, 30, 32]
    depths = [10, 14, 20]

    print("=" * 85)
    print("Random Circuit Sampling (RCS) Benchmark — 1D Chain Topology")
    print("Methodology: Sycamore-style cycles (random 1Q + alternating CZ)")
    print("=" * 85)
    print()

    results = []

    for depth in depths:
        print(f"--- depth = {depth} ---")
        print(f"{'Qubits':>6}  {'Gates':>6}  {'Time(ms)':>10}  "
              f"{'us/gate':>8}  {'BW(TB/s)':>9}  {'2x scaling':>10}")
        print("-" * 65)

        prev_time = None
        for n in qubit_counts:
            try:
                r = bench_rcs(n, depth, seed=42)
                results.append(r)

                # Compute scaling ratio vs. previous qubit count
                if prev_time is not None:
                    qubit_diff = n - prev_n
                    expected_ratio = 2.0 ** qubit_diff
                    actual_ratio = r['median_time_s'] / prev_time
                    scaling_str = f"{actual_ratio:.2f}x (exp {expected_ratio:.0f}x)"
                else:
                    scaling_str = "—"

                print(f"{r['n_qubits']:>6}  {r['gate_count']:>6}  "
                      f"{r['median_time_s']*1e3:>10.2f}  "
                      f"{r['time_per_gate_us']:>8.2f}  "
                      f"{r['bandwidth_TBs']:>9.3f}  "
                      f"{scaling_str:>10}")

                prev_time = r['median_time_s']
                prev_n = n

            except Exception as e:
                print(f"{n:>6}  FAILED: {e}")
                break

        print()

    # --- Summary: scaling analysis ---
    print("=" * 85)
    print("Scaling Analysis (depth=20)")
    print("=" * 85)
    depth20 = [r for r in results if r['depth'] == 20]
    if len(depth20) >= 2:
        print(f"{'From':>6} {'To':>6}  {'dQubits':>7}  "
              f"{'Expected':>8}  {'Actual':>8}  {'Efficiency':>10}")
        print("-" * 55)
        for i in range(1, len(depth20)):
            dq = depth20[i]['n_qubits'] - depth20[i - 1]['n_qubits']
            expected = 2.0 ** dq
            actual = depth20[i]['median_time_s'] / depth20[i - 1]['median_time_s']
            eff = (expected / actual) * 100 if actual > 0 else 0
            print(f"{depth20[i-1]['n_qubits']:>6} {depth20[i]['n_qubits']:>6}  "
                  f"{dq:>7}  {expected:>8.1f}x {actual:>8.2f}x {eff:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
