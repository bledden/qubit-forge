"""Standard circuit benchmarks: QFT, Grover, random circuits."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import mi300x_env  # Set HIP env vars before importing pyquantum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import pyquantum as pq
import numpy as np

def build_qft(n):
    circ = pq.Circuit(n)
    for target in range(n):
        circ.h(target)
        for control in range(target + 1, n):
            angle = np.pi / (2 ** (control - target))
            circ.rz(angle, target)
    for i in range(n // 2):
        circ.swap(i, n - i - 1)
    return circ

def build_ghz(n):
    circ = pq.Circuit(n)
    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    return circ

def build_random(n, depth, seed=42):
    rng = np.random.default_rng(seed)
    circ = pq.Circuit(n)
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(['h', 'rx', 'rz'])
            if gate == 'h':
                circ.h(q)
            elif gate == 'rx':
                circ.rx(rng.uniform(0, 2 * np.pi), q)
            else:
                circ.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(0, n - 1, 2):
            circ.cx(q, q + 1)
    return circ

def bench_circuit(name, n, circ, fused=True, n_warmup=2, n_iter=5):
    times = []
    for i in range(n_warmup + n_iter):
        sv = pq.StateVector(n)
        start = time.perf_counter()
        if fused:
            sv.apply_circuit_fused(circ)
        else:
            sv.apply_circuit(circ)
        end = time.perf_counter()
        if i >= n_warmup:
            times.append(end - start)

    median = np.median(times)
    print(f"{name:>20} n={n:>2}  gates={circ.size:>5}  "
          f"{'fused' if fused else 'plain':>5}  {median*1e3:>8.2f} ms")

def main():
    for n in [10, 20, 25, 28, 30]:
        print(f"\n=== {n} qubits ===")
        qft = build_qft(n)
        ghz = build_ghz(n)
        rand = build_random(n, 20)

        bench_circuit("QFT", n, qft, fused=False)
        bench_circuit("QFT (fused)", n, qft, fused=True)
        bench_circuit("GHZ", n, ghz, fused=False)
        bench_circuit("Random d=20", n, rand, fused=False)
        bench_circuit("Random d=20 (fused)", n, rand, fused=True)

if __name__ == "__main__":
    main()
