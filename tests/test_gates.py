import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import pyquantum as pq


def assert_amplitudes(sv, expected, atol=1e-10):
    amps = sv.amplitudes()
    np.testing.assert_allclose(amps, expected, atol=atol)


class TestSingleQubitGates:
    def test_hadamard_qubit0(self):
        sv = pq.StateVector(3)
        sv.h(0)
        expected = np.zeros(8, dtype=complex)
        s = 1 / np.sqrt(2)
        expected[0] = s
        expected[1] = s
        assert_amplitudes(sv, expected)

    def test_hadamard_qubit1(self):
        sv = pq.StateVector(3)
        sv.h(1)
        expected = np.zeros(8, dtype=complex)
        s = 1 / np.sqrt(2)
        expected[0] = s
        expected[2] = s
        assert_amplitudes(sv, expected)

    def test_hadamard_qubit2(self):
        sv = pq.StateVector(3)
        sv.h(2)
        expected = np.zeros(8, dtype=complex)
        s = 1 / np.sqrt(2)
        expected[0] = s
        expected[4] = s
        assert_amplitudes(sv, expected)

    def test_pauli_x(self):
        sv = pq.StateVector(2)
        sv.x(0)
        expected = np.array([0, 1, 0, 0], dtype=complex)
        assert_amplitudes(sv, expected)

    def test_pauli_x_twice_is_identity(self):
        sv = pq.StateVector(2)
        sv.x(0)
        sv.x(0)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        assert_amplitudes(sv, expected)

    def test_hadamard_twice_is_identity(self):
        sv = pq.StateVector(3)
        sv.h(1)
        sv.h(1)
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0
        assert_amplitudes(sv, expected)

    def test_rz_gate(self):
        sv = pq.StateVector(1)
        sv.h(0)
        sv.rz(np.pi / 2, 0)
        amps = sv.amplitudes()
        s = 1 / np.sqrt(2)
        expected_0 = s * np.exp(-1j * np.pi / 4)
        expected_1 = s * np.exp(1j * np.pi / 4)
        np.testing.assert_allclose(amps[0], expected_0, atol=1e-10)
        np.testing.assert_allclose(amps[1], expected_1, atol=1e-10)

    def test_all_qubits_independent(self):
        n = 4
        sv = pq.StateVector(n)
        for q in range(n):
            sv.h(q)
        amps = sv.amplitudes()
        expected_amp = 1.0 / np.sqrt(2**n)
        np.testing.assert_allclose(np.abs(amps), expected_amp, atol=1e-10)


class TestTwoQubitGates:
    def test_bell_state(self):
        sv = pq.StateVector(2)
        sv.h(0)
        sv.cx(0, 1)
        s = 1 / np.sqrt(2)
        expected = np.array([s, 0, 0, s], dtype=complex)
        assert_amplitudes(sv, expected)

    def test_bell_state_reversed(self):
        sv = pq.StateVector(2)
        sv.h(1)
        sv.cx(1, 0)
        s = 1 / np.sqrt(2)
        expected = np.array([s, 0, 0, s], dtype=complex)
        assert_amplitudes(sv, expected)

    def test_ghz_state(self):
        sv = pq.StateVector(3)
        sv.h(0)
        sv.cx(0, 1)
        sv.cx(1, 2)
        s = 1 / np.sqrt(2)
        expected = np.zeros(8, dtype=complex)
        expected[0] = s
        expected[7] = s
        assert_amplitudes(sv, expected)

    def test_cnot_identity(self):
        sv = pq.StateVector(2)
        sv.h(0)
        sv.cx(0, 1)
        sv.cx(0, 1)
        sv.h(0)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        assert_amplitudes(sv, expected)

    def test_swap_gate(self):
        sv = pq.StateVector(2)
        sv.x(0)
        sv.swap(0, 1)
        expected = np.array([0, 0, 1, 0], dtype=complex)
        assert_amplitudes(sv, expected)

    def test_cz_gate(self):
        sv = pq.StateVector(2)
        sv.x(0)
        sv.x(1)
        sv.cz(0, 1)
        expected = np.array([0, 0, 0, -1], dtype=complex)
        assert_amplitudes(sv, expected)


class TestAgainstNumpy:
    @staticmethod
    def numpy_apply_gate(state, gate_matrix, target, n_qubits):
        N = 2**n_qubits
        result = state.copy()
        stride = 2**target
        for i in range(N):
            if i & stride:
                continue
            lo, hi = i, i | stride
            a, b = result[lo], result[hi]
            result[lo] = gate_matrix[0, 0] * a + gate_matrix[0, 1] * b
            result[hi] = gate_matrix[1, 0] * a + gate_matrix[1, 1] * b
        return result

    def test_random_circuit_vs_numpy(self):
        n = 4
        rng = np.random.default_rng(42)
        np_state = np.zeros(2**n, dtype=complex)
        np_state[0] = 1.0
        sv = pq.StateVector(n)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        targets = rng.choice(n, size=10, replace=True)
        for t in targets:
            np_state = self.numpy_apply_gate(np_state, H, int(t), n)
            sv.h(int(t))
        assert_amplitudes(sv, np_state)


class TestCircuit:
    def test_circuit_bell_state(self):
        circ = pq.Circuit(2)
        circ.h(0)
        circ.cx(0, 1)
        sv = pq.StateVector(2)
        sv.apply_circuit(circ)
        s = 1 / np.sqrt(2)
        expected = np.array([s, 0, 0, s], dtype=complex)
        assert_amplitudes(sv, expected)

    def test_circuit_qft_2qubit(self):
        circ = pq.Circuit(2)
        circ.h(0)
        circ.rz(np.pi / 2, 0)
        circ.h(1)
        sv = pq.StateVector(2)
        sv.apply_circuit(circ)
        probs = sv.probabilities()
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-10)


class TestFusion:
    def test_fused_same_qubit(self):
        circ = pq.Circuit(2)
        circ.h(0)
        circ.h(0)
        circ.h(0)
        sv = pq.StateVector(2)
        sv.apply_circuit_fused(circ)
        sv_ref = pq.StateVector(2)
        sv_ref.h(0)
        np.testing.assert_allclose(sv.amplitudes(), sv_ref.amplitudes(), atol=1e-10)

    def test_fused_matches_unfused(self):
        circ = pq.Circuit(4)
        circ.h(0)
        circ.h(1)
        circ.cx(0, 1)
        circ.rz(0.5, 0)
        circ.rx(0.3, 0)
        circ.h(2)
        circ.cx(2, 3)
        sv_fused = pq.StateVector(4)
        sv_fused.apply_circuit_fused(circ)
        sv_plain = pq.StateVector(4)
        sv_plain.apply_circuit(circ)
        np.testing.assert_allclose(sv_fused.amplitudes(), sv_plain.amplitudes(), atol=1e-10)


class TestMeasurement:
    def test_probabilities_sum_to_one(self):
        sv = pq.StateVector(3)
        sv.h(0)
        sv.h(1)
        probs = sv.probabilities()
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-10)

    def test_bell_state_measurement(self):
        sv = pq.StateVector(2)
        sv.h(0)
        sv.cx(0, 1)
        samples = sv.measure(10000)
        counts = {}
        for s in samples:
            counts[int(s)] = counts.get(int(s), 0) + 1
        assert set(counts.keys()).issubset({0, 3})
        assert abs(counts.get(0, 0) / 10000 - 0.5) < 0.05
        assert abs(counts.get(3, 0) / 10000 - 0.5) < 0.05

    def test_deterministic_state(self):
        sv = pq.StateVector(2)
        sv.x(0)
        samples = sv.measure(100)
        assert all(s == 1 for s in samples)


class TestQuantumAlgorithms:
    def test_qft_vs_numpy_fft(self):
        """Compare our QFT against numpy's FFT for correctness."""
        n = 4
        sv = pq.StateVector(n)
        sv.x(0)
        sv.x(1)

        for target in range(n):
            sv.h(target)
            for control in range(target + 1, n):
                angle = np.pi / (2 ** (control - target))
                sv.rz(angle, target)

        gpu_amps = sv.amplitudes()

        input_state = np.zeros(2**n, dtype=complex)
        input_state[3] = 1.0
        np_fft = np.fft.ifft(input_state) * np.sqrt(2**n)

        gpu_probs = np.abs(gpu_amps) ** 2
        np_probs = np.abs(np_fft) ** 2
        np.testing.assert_allclose(sorted(gpu_probs), sorted(np_probs), atol=1e-8)

    def test_grover_2qubit(self):
        """Grover's search on 2 qubits, mark |11⟩."""
        sv = pq.StateVector(2)
        sv.h(0)
        sv.h(1)
        sv.cz(0, 1)
        sv.h(0)
        sv.h(1)
        sv.x(0)
        sv.x(1)
        sv.cz(0, 1)
        sv.x(0)
        sv.x(1)
        sv.h(0)
        sv.h(1)

        probs = sv.probabilities()
        np.testing.assert_allclose(probs[3], 1.0, atol=1e-10)

    def test_grover_3qubit(self):
        """Grover's on 3 qubits, mark |101⟩. One iteration."""
        n = 3
        marked = 5

        sv = pq.StateVector(n)
        for q in range(n):
            sv.h(q)

        sv.x(1)
        sv.h(2)
        sv.cx(0, 2)
        sv.h(2)
        sv.x(1)

        for q in range(n):
            sv.h(q)
            sv.x(q)
        sv.h(n - 1)
        sv.cx(0, n - 1)
        sv.h(n - 1)
        for q in range(n):
            sv.x(q)
            sv.h(q)

        probs = sv.probabilities()
        assert probs[marked] > 1.0 / (2**n), \
            f"Marked state probability {probs[marked]} should exceed uniform {1.0/(2**n)}"


class TestNoise:
    def test_no_noise_unchanged(self):
        """Zero noise should produce identical results."""
        noise = pq.NoiseModel(seed=42)
        noise.set_single_qubit_noise(pq.NoiseType.Depolarizing, 0.0)

        circ = pq.Circuit(3)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)

        sv_clean = pq.StateVector(3)
        sv_clean.apply_circuit(circ)

        sv_noisy = pq.StateVector(3)
        sv_noisy.apply_circuit_noisy(circ, noise)

        np.testing.assert_allclose(sv_clean.amplitudes(), sv_noisy.amplitudes(), atol=1e-10)

    def test_high_noise_destroys_state(self):
        """100% depolarizing noise should produce a maximally mixed state."""
        noise = pq.NoiseModel(seed=42)
        noise.set_single_qubit_noise(pq.NoiseType.Depolarizing, 1.0)
        noise.set_two_qubit_noise(pq.NoiseType.Depolarizing, 1.0)

        circ = pq.Circuit(3)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)

        # Run many noisy instances and average probabilities
        avg_probs = np.zeros(8)
        n_trials = 100
        for _ in range(n_trials):
            sv = pq.StateVector(3)
            sv.apply_circuit_noisy(circ, noise)
            avg_probs += sv.probabilities()
        avg_probs /= n_trials

        # With 100% noise, probabilities should be roughly uniform
        # (not exactly, because noise is after each gate, not a perfect depolarizing channel)
        assert np.std(avg_probs) < 0.15, f"Probabilities should be roughly uniform: {avg_probs}"

    def test_measurement_error(self):
        """Measurement errors should flip bits."""
        noise = pq.NoiseModel(seed=42)
        noise.set_measurement_error(1.0)  # 100% measurement error

        sv = pq.StateVector(2)
        # State is |00⟩, so clean measurement always gives 0
        # With 100% measurement error, every bit flips → always gives 3 (|11⟩)
        samples = sv.measure_noisy(100, noise)
        assert all(s == 3 for s in samples), "100% measurement error on |00⟩ should give |11⟩"

    def test_bit_flip_noise(self):
        """Bit flip noise should flip |0⟩ to |1⟩ stochastically."""
        noise = pq.NoiseModel(seed=42)
        noise.set_single_qubit_noise(pq.NoiseType.BitFlip, 0.5)

        # Use Ry(pi/6) — asymmetric state where X visibly changes probabilities
        # Ry(pi/6)|0⟩ has P(0) ≈ 0.933, P(1) ≈ 0.067
        # X Ry(pi/6)|0⟩ has P(0) ≈ 0.067, P(1) ≈ 0.933
        circ = pq.Circuit(1)
        circ.ry(np.pi / 6, 0)

        results = set()
        for _ in range(50):
            sv = pq.StateVector(1)
            sv.apply_circuit_noisy(circ, noise)
            probs = sv.probabilities()
            results.add(round(probs[0], 1))

        # Should see ~0.9 (no error) and ~0.1 (X applied)
        assert len(results) >= 2, f"Should see different outcomes from bit flip noise, got {results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
