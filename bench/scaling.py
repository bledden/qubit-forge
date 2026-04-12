"""Qubit scaling benchmark: measure time for standard operations as qubits increase."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mi300x_env  # Set HIP env vars before importing pyquantum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import pyquantum as pq
import numpy as np

def main():
    print(f"{'Qubits':>6} {'StateVec(GB)':>12} {'Alloc(ms)':>10} {'H_all(ms)':>10} {'GHZ(ms)':>10}")
    print("-" * 55)

    for n in range(10, 34):
        mem_gb = (2**n * 16) / 1e9

        try:
            t0 = time.perf_counter()
            sv = pq.StateVector(n)
            t_alloc = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            for q in range(n):
                sv.h(q)
            t_h = (time.perf_counter() - t0) * 1e3

            sv.init_zero()
            t0 = time.perf_counter()
            sv.h(0)
            for q in range(n - 1):
                sv.cx(q, q + 1)
            t_ghz = (time.perf_counter() - t0) * 1e3

            print(f"{n:>6} {mem_gb:>12.3f} {t_alloc:>10.1f} {t_h:>10.1f} {t_ghz:>10.1f}")

        except Exception as e:
            print(f"{n:>6} {mem_gb:>12.3f}  FAILED: {e}")
            break

if __name__ == "__main__":
    main()
