"""Per-gate bandwidth benchmark on MI300X.

Measures achieved HBM bandwidth for single-qubit gate application
across all qubit indices at various qubit counts.

33 qubits (128GB) exceeds A100-80GB and H100-80GB capacity.
H200 (141GB) and B200 (192GB) can also fit 33 qubits.
MI300X advantage: 5.3 TB/s bandwidth (vs H200 4.8, H100 3.35, A100 2.0).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import pyquantum as pq
import numpy as np

MI300X_PEAK_BW = 5.3e12  # 5.3 TB/s

def benchmark_gate(n_qubits, target_qubit, n_warmup=3, n_iter=10):
    sv = pq.StateVector(n_qubits)

    for _ in range(n_warmup):
        sv.h(target_qubit)

    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        sv.h(target_qubit)
        end = time.perf_counter()
        times.append(end - start)

    median_time = np.median(times)
    bytes_moved = 2 * (2**n_qubits) * 16
    bandwidth = bytes_moved / median_time
    efficiency = bandwidth / MI300X_PEAK_BW * 100

    return median_time, bandwidth, efficiency

def main():
    qubit_counts = [20, 25, 28, 30]
    try:
        _ = pq.StateVector(32)
        qubit_counts.append(32)
        del _
    except:
        pass
    try:
        _ = pq.StateVector(33)
        qubit_counts.append(33)
        del _
    except:
        pass

    print(f"{'Qubits':>6} {'Target':>6} {'Time(us)':>10} {'BW(TB/s)':>10} {'Eff%':>6}")
    print("-" * 45)

    for n in qubit_counts:
        for k in range(min(n, 20)):
            if k < 5 or (5 <= k < 11 and k % 2 == 0) or k >= 10:
                t, bw, eff = benchmark_gate(n, k)
                print(f"{n:>6} {k:>6} {t*1e6:>10.1f} {bw/1e12:>10.3f} {eff:>6.1f}")
        print()

if __name__ == "__main__":
    main()
