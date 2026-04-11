#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>

#include "quantum/statevec.h"
#include "quantum/gates.h"
#include "quantum/circuit.h"

namespace py = pybind11;

py::array_t<std::complex<double>> amplitudes_to_numpy(const std::vector<quantum::Complex128>& amps) {
    auto result = py::array_t<std::complex<double>>(amps.size());
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < amps.size(); i++) {
        buf(i) = std::complex<double>(amps[i].x, amps[i].y);
    }
    return result;
}

PYBIND11_MODULE(pyquantum, m) {
    m.doc() = "GPU-accelerated quantum state vector simulator";

    py::class_<quantum::StateVector>(m, "StateVector")
        .def(py::init<int>(), py::arg("n_qubits"))
        .def_property_readonly("n_qubits", &quantum::StateVector::n_qubits)
        .def_property_readonly("size", &quantum::StateVector::size)
        .def("init_zero", &quantum::StateVector::init_zero)
        .def("amplitudes", [](const quantum::StateVector& sv) {
            return amplitudes_to_numpy(sv.download());
        })
        .def("probability", [](const quantum::StateVector& sv, int64_t index) {
            auto amps = sv.download();
            return amps[index].norm2();
        })
        .def("probabilities", [](const quantum::StateVector& sv) {
            auto probs = sv.probabilities_vec();
            auto result = py::array_t<double>(probs.size());
            auto buf = result.mutable_unchecked<1>();
            for (size_t i = 0; i < probs.size(); i++) {
                buf(i) = probs[i];
            }
            return result;
        })
        .def("h", [](quantum::StateVector& sv, int t) {
            sv.apply_gate_1q(quantum::gates::H(), t);
        }, py::arg("target"))
        .def("x", [](quantum::StateVector& sv, int t) {
            sv.apply_gate_1q(quantum::gates::X(), t);
        }, py::arg("target"))
        .def("y", [](quantum::StateVector& sv, int t) {
            sv.apply_gate_1q(quantum::gates::Y(), t);
        }, py::arg("target"))
        .def("z", [](quantum::StateVector& sv, int t) {
            sv.apply_gate_1q(quantum::gates::Z(), t);
        }, py::arg("target"))
        .def("rx", [](quantum::StateVector& sv, double theta, int t) {
            sv.apply_gate_1q(quantum::gates::Rx(theta), t);
        }, py::arg("theta"), py::arg("target"))
        .def("ry", [](quantum::StateVector& sv, double theta, int t) {
            sv.apply_gate_1q(quantum::gates::Ry(theta), t);
        }, py::arg("theta"), py::arg("target"))
        .def("rz", [](quantum::StateVector& sv, double theta, int t) {
            sv.apply_gate_1q(quantum::gates::Rz(theta), t);
        }, py::arg("theta"), py::arg("target"))
        .def("s", [](quantum::StateVector& sv, int t) {
            sv.apply_gate_1q(quantum::gates::S(), t);
        }, py::arg("target"))
        .def("t", [](quantum::StateVector& sv, int t) {
            sv.apply_gate_1q(quantum::gates::T(), t);
        }, py::arg("target"))
        .def("cx", [](quantum::StateVector& sv, int c, int t) {
            sv.apply_gate_2q(quantum::gates::CNOT(), t, c);
        }, py::arg("control"), py::arg("target"))
        .def("cz", [](quantum::StateVector& sv, int c, int t) {
            sv.apply_gate_2q(quantum::gates::CZ(), t, c);
        }, py::arg("control"), py::arg("target"))
        .def("swap", [](quantum::StateVector& sv, int a, int b) {
            sv.apply_gate_2q(quantum::gates::SWAP(), a, b);
        }, py::arg("qubit_a"), py::arg("qubit_b"))
        .def("apply_circuit", &quantum::StateVector::apply_circuit)
        .def("apply_circuit_fused", &quantum::StateVector::apply_circuit_fused)
        .def("measure", [](const quantum::StateVector& sv, int shots) {
            auto results = sv.measure(shots);
            auto arr = py::array_t<int64_t>(results.size());
            auto buf = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < results.size(); i++) {
                buf(i) = results[i];
            }
            return arr;
        }, py::arg("shots") = 1024)
    ;

    py::class_<quantum::Circuit>(m, "Circuit")
        .def(py::init<int>(), py::arg("n_qubits"))
        .def_property_readonly("n_qubits", &quantum::Circuit::n_qubits)
        .def_property_readonly("size", &quantum::Circuit::size)
        .def("h", &quantum::Circuit::h, py::arg("target"))
        .def("x", &quantum::Circuit::x, py::arg("target"))
        .def("y", &quantum::Circuit::y, py::arg("target"))
        .def("z", &quantum::Circuit::z, py::arg("target"))
        .def("rx", &quantum::Circuit::rx, py::arg("theta"), py::arg("target"))
        .def("ry", &quantum::Circuit::ry, py::arg("theta"), py::arg("target"))
        .def("rz", &quantum::Circuit::rz, py::arg("theta"), py::arg("target"))
        .def("s", &quantum::Circuit::s, py::arg("target"))
        .def("t", &quantum::Circuit::t, py::arg("target"))
        .def("cx", &quantum::Circuit::cx, py::arg("control"), py::arg("target"))
        .def("cz", &quantum::Circuit::cz, py::arg("control"), py::arg("target"))
        .def("swap", &quantum::Circuit::swap, py::arg("qubit_a"), py::arg("qubit_b"))
    ;
}
