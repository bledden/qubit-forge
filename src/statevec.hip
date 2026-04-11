#include "quantum/statevec.h"
#include "quantum/kernels.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

namespace quantum {

StateVector::StateVector(int n_qubits) : n_qubits_(n_qubits), d_state_(nullptr) {
    if (n_qubits < 1 || n_qubits > 34) {
        throw std::invalid_argument("n_qubits must be between 1 and 34");
    }
    int64_t bytes = size() * sizeof(Complex128);
    HIP_CHECK(hipMalloc(&d_state_, bytes));
    init_zero();
}

StateVector::~StateVector() {
    if (d_state_) {
        hipFree(d_state_);
    }
}

void StateVector::init_zero() {
    int64_t bytes = size() * sizeof(Complex128);
    HIP_CHECK(hipMemset(d_state_, 0, bytes));
    Complex128 one(1.0, 0.0);
    HIP_CHECK(hipMemcpy(d_state_, &one, sizeof(Complex128), hipMemcpyHostToDevice));
}

std::vector<Complex128> StateVector::download() const {
    std::vector<Complex128> host(size());
    int64_t bytes = size() * sizeof(Complex128);
    HIP_CHECK(hipMemcpy(host.data(), d_state_, bytes, hipMemcpyDeviceToHost));
    return host;
}

bool gate_is_diagonal(const Gate1Q& gate) {
    return (std::abs(gate.m[0][1].x) < 1e-15 && std::abs(gate.m[0][1].y) < 1e-15 &&
            std::abs(gate.m[1][0].x) < 1e-15 && std::abs(gate.m[1][0].y) < 1e-15);
}

void StateVector::apply_gate_1q(const Gate1Q& gate, int target) {
    if (target < 0 || target >= n_qubits_) {
        throw std::invalid_argument("target qubit out of range");
    }
    if (gate_is_diagonal(gate)) {
        launch_apply_diagonal(d_state_, n_qubits_, target, gate);
    } else {
        launch_apply_1q(d_state_, n_qubits_, target, gate);
    }
    HIP_CHECK(hipDeviceSynchronize());
}

void StateVector::apply_gate_2q(const Gate2Q& gate, int qubit_a, int qubit_b) {
    if (qubit_a < 0 || qubit_a >= n_qubits_ || qubit_b < 0 || qubit_b >= n_qubits_) {
        throw std::invalid_argument("qubit index out of range");
    }
    if (qubit_a == qubit_b) {
        throw std::invalid_argument("two-qubit gate requires distinct qubits");
    }
    launch_apply_2q(d_state_, n_qubits_, qubit_a, qubit_b, gate);
    HIP_CHECK(hipDeviceSynchronize());
}

void StateVector::apply_circuit(const Circuit& circuit) {
    for (const auto& op : circuit.ops()) {
        if (op.is_single_qubit()) {
            apply_gate_1q(op.to_gate1q(), op.target);
        } else {
            apply_gate_2q(op.to_gate2q(), op.control, op.target);
        }
    }
}

void StateVector::apply_circuit_fused(const Circuit& circuit) {
    auto layers = fuse_circuit(circuit);
    for (const auto& layer : layers) {
        for (const auto& op : layer.ops) {
            if (op.is_single_qubit()) {
                Gate1Q g = op.to_gate1q();
                if (gate_is_diagonal(g)) {
                    launch_apply_diagonal(d_state_, n_qubits_, op.target, g);
                } else {
                    launch_apply_1q(d_state_, n_qubits_, op.target, g);
                }
            } else {
                launch_apply_2q(d_state_, n_qubits_, op.control, op.target, op.to_gate2q());
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
    }
}

std::vector<double> StateVector::probabilities_vec() const {
    int64_t n = size();
    double* d_probs;
    HIP_CHECK(hipMalloc(&d_probs, n * sizeof(double)));
    launch_probabilities(d_state_, d_probs, n);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> result(n);
    HIP_CHECK(hipMemcpy(result.data(), d_probs, n * sizeof(double), hipMemcpyDeviceToHost));
    hipFree(d_probs);
    return result;
}

std::vector<int64_t> StateVector::measure(int shots) const {
    auto probs = probabilities_vec();
    int64_t n = size();

    std::vector<double> cdf(n);
    cdf[0] = probs[0];
    for (int64_t i = 1; i < n; i++) {
        cdf[i] = cdf[i - 1] + probs[i];
    }

    std::vector<int64_t> results(shots);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int s = 0; s < shots; s++) {
        double r = dist(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        results[s] = std::distance(cdf.begin(), it);
    }
    return results;
}

} // namespace quantum
