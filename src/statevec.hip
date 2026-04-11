#include "quantum/statevec.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstring>

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

// Forward declarations for kernel launchers
void launch_apply_1q(Complex128* state, int n_qubits, int target, const Gate1Q& gate, hipStream_t stream = 0);
void launch_apply_2q(Complex128* state, int n_qubits, int qubit_a, int qubit_b, const Gate2Q& gate, hipStream_t stream = 0);

void StateVector::apply_gate_1q(const Gate1Q& gate, int target) {
    if (target < 0 || target >= n_qubits_) {
        throw std::invalid_argument("target qubit out of range");
    }
    launch_apply_1q(d_state_, n_qubits_, target, gate);
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

} // namespace quantum
