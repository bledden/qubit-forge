#pragma once

#include "quantum/types.h"
#include "quantum/gates.h"
#include <vector>
#include <string>

namespace quantum {

enum class GateType {
    H, X, Y, Z, Rx, Ry, Rz, S, T,
    CNOT, CZ, SWAP,
    Custom1Q, Custom2Q
};

struct GateOp {
    GateType type;
    int target;
    int control;      // -1 for single-qubit gates
    double param;
    Gate1Q custom_1q;
    Gate2Q custom_2q;

    bool is_single_qubit() const { return control == -1; }
    bool is_diagonal() const;

    Gate1Q to_gate1q() const;
    Gate2Q to_gate2q() const;
};

class Circuit {
public:
    Circuit(int n_qubits);

    int n_qubits() const { return n_qubits_; }
    const std::vector<GateOp>& ops() const { return ops_; }
    size_t size() const { return ops_.size(); }

    void h(int target);
    void x(int target);
    void y(int target);
    void z(int target);
    void rx(double theta, int target);
    void ry(double theta, int target);
    void rz(double theta, int target);
    void s(int target);
    void t(int target);

    void cx(int control, int target);
    void cz(int control, int target);
    void swap(int a, int b);

private:
    int n_qubits_;
    std::vector<GateOp> ops_;
};

} // namespace quantum
