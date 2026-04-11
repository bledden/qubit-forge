#pragma once

#include "quantum/types.h"
#include "quantum/circuit.h"
#include "quantum/fusion.h"
#include <cstdint>
#include <vector>

namespace quantum {

class StateVector {
public:
    StateVector(int n_qubits);
    ~StateVector();

    StateVector(const StateVector&) = delete;
    StateVector& operator=(const StateVector&) = delete;

    int n_qubits() const { return n_qubits_; }
    int64_t size() const { return 1LL << n_qubits_; }

    void init_zero();
    std::vector<Complex128> download() const;

    void apply_gate_1q(const Gate1Q& gate, int target);
    void apply_gate_2q(const Gate2Q& gate, int qubit_a, int qubit_b);

    void apply_circuit(const Circuit& circuit);
    void apply_circuit_fused(const Circuit& circuit);

    Complex128* data() { return d_state_; }
    const Complex128* data() const { return d_state_; }

private:
    int n_qubits_;
    Complex128* d_state_;
};

} // namespace quantum
