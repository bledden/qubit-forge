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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
