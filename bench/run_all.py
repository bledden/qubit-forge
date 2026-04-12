"""Master benchmark runner — executes all benchmarks and captures output.

Runs each benchmark script as a subprocess, captures stdout to both the
terminal and a results file under bench/results/.

Benchmarks executed:
  1. single_gate.py   — Per-gate bandwidth (HBM utilization)
  2. circuit_bench.py — Standard circuits (QFT, GHZ, random)
  3. rcs_bench.py     — Random Circuit Sampling (Sycamore methodology)
  4. quantum_volume.py — Quantum Volume (IBM methodology)
  5. scaling.py       — Qubit scaling (allocation + gate timing)
"""
import subprocess
import sys
import os
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, 'results')
PYTHON = sys.executable

BENCHMARKS = [
    ('single_gate.py',    'single_gate.txt',   'Single-Gate Bandwidth'),
    ('circuit_bench.py',  'circuit_bench.txt',  'Circuit Benchmarks'),
    ('rcs_bench.py',      'rcs_bench.txt',      'Random Circuit Sampling'),
    ('quantum_volume.py', 'qv_bench.txt',       'Quantum Volume'),
    ('scaling.py',        'scaling.txt',         'Qubit Scaling'),
]


def run_benchmark(script, output_file, label):
    """Run a single benchmark script and capture output.

    Args:
        script: Filename of the benchmark script (in BENCH_DIR).
        output_file: Filename for the results (in RESULTS_DIR).
        label: Human-readable label for printing.

    Returns:
        (success: bool, elapsed_s: float, output_path: str)
    """
    script_path = os.path.join(BENCH_DIR, script)
    output_path = os.path.join(RESULTS_DIR, output_file)

    if not os.path.exists(script_path):
        print(f"  SKIP — {script} not found")
        return False, 0.0, output_path

    print(f"  Running {script} ...", flush=True)
    start = time.perf_counter()

    try:
        result = subprocess.run(
            [PYTHON, script_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per benchmark
        )
        elapsed = time.perf_counter() - start

        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr

        # Write to file
        with open(output_path, 'w') as f:
            f.write(output)

        # Echo to terminal
        print(output)

        if result.returncode != 0:
            print(f"  WARNING: {script} exited with code {result.returncode}")
            return False, elapsed, output_path

        return True, elapsed, output_path

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print(f"  TIMEOUT after {elapsed:.0f}s")
        with open(output_path, 'w') as f:
            f.write(f"TIMEOUT after {elapsed:.0f}s\n")
        return False, elapsed, output_path

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  ERROR: {e}")
        with open(output_path, 'w') as f:
            f.write(f"ERROR: {e}\n")
        return False, elapsed, output_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("GPU Quantum Simulator — Full Benchmark Suite")
    print("=" * 70)
    print(f"Python:     {sys.executable}")
    print(f"Results:    {RESULTS_DIR}")
    print(f"Benchmarks: {len(BENCHMARKS)}")
    print()

    total_start = time.perf_counter()
    summary = []

    for script, output_file, label in BENCHMARKS:
        print(f"\n{'─' * 70}")
        print(f"[{len(summary)+1}/{len(BENCHMARKS)}] {label}")
        print(f"{'─' * 70}")

        success, elapsed, output_path = run_benchmark(script, output_file, label)
        summary.append((label, success, elapsed, output_path))

    total_elapsed = time.perf_counter() - total_start

    # --- Summary ---
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'#':>2}  {'Benchmark':<30}  {'Status':>8}  {'Time':>10}  {'Output'}")
    print("-" * 70)

    n_pass = 0
    for i, (label, success, elapsed, output_path) in enumerate(summary):
        status = "PASS" if success else "FAIL"
        n_pass += int(success)
        rel_path = os.path.relpath(output_path, BENCH_DIR)
        print(f"{i+1:>2}  {label:<30}  {status:>8}  {elapsed:>8.1f}s  {rel_path}")

    print("-" * 70)
    print(f"    {n_pass}/{len(summary)} passed, total time: {total_elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
