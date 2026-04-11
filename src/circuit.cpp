#include "quantum/circuit.h"
#include "quantum/kernels.h"
#include <stdexcept>
#include <cmath>

namespace quantum {

bool GateOp::is_diagonal() const {
    switch (type) {
        case GateType::Z: case GateType::Rz: case GateType::S: case GateType::T:
        case GateType::CZ:
            return true;
        case GateType::Custom1Q:
            return gate_is_diagonal(custom_1q);
        default:
            return false;
    }
}

Gate1Q GateOp::to_gate1q() const {
    switch (type) {
        case GateType::H:  return gates::H();
        case GateType::X:  return gates::X();
        case GateType::Y:  return gates::Y();
        case GateType::Z:  return gates::Z();
        case GateType::Rx: return gates::Rx(param);
        case GateType::Ry: return gates::Ry(param);
        case GateType::Rz: return gates::Rz(param);
        case GateType::S:  return gates::S();
        case GateType::T:  return gates::T();
        case GateType::Custom1Q: return custom_1q;
        default: throw std::runtime_error("not a 1Q gate");
    }
}

Gate2Q GateOp::to_gate2q() const {
    switch (type) {
        case GateType::CNOT: return gates::CNOT();
        case GateType::CZ:   return gates::CZ();
        case GateType::SWAP: return gates::SWAP();
        case GateType::Custom2Q: return custom_2q;
        default: throw std::runtime_error("not a 2Q gate");
    }
}

Circuit::Circuit(int n_qubits) : n_qubits_(n_qubits) {
    if (n_qubits < 1) throw std::invalid_argument("n_qubits must be >= 1");
}

void Circuit::h(int t)  { ops_.push_back({GateType::H,  t, -1, 0, {}, {}}); }
void Circuit::x(int t)  { ops_.push_back({GateType::X,  t, -1, 0, {}, {}}); }
void Circuit::y(int t)  { ops_.push_back({GateType::Y,  t, -1, 0, {}, {}}); }
void Circuit::z(int t)  { ops_.push_back({GateType::Z,  t, -1, 0, {}, {}}); }
void Circuit::rx(double theta, int t) { ops_.push_back({GateType::Rx, t, -1, theta, {}, {}}); }
void Circuit::ry(double theta, int t) { ops_.push_back({GateType::Ry, t, -1, theta, {}, {}}); }
void Circuit::rz(double theta, int t) { ops_.push_back({GateType::Rz, t, -1, theta, {}, {}}); }
void Circuit::s(int t)  { ops_.push_back({GateType::S,  t, -1, 0, {}, {}}); }
void Circuit::t(int t)  { ops_.push_back({GateType::T,  t, -1, 0, {}, {}}); }

void Circuit::cx(int c, int t)   { ops_.push_back({GateType::CNOT, t, c, 0, {}, {}}); }
void Circuit::cz(int c, int t)   { ops_.push_back({GateType::CZ,   t, c, 0, {}, {}}); }
void Circuit::swap(int a, int b) { ops_.push_back({GateType::SWAP, b, a, 0, {}, {}}); }

} // namespace quantum
