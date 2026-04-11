#include "quantum/types.h"
#include <hip/hip_runtime.h>

namespace quantum {

__global__ void kernel_apply_2q(
    Complex128* __restrict__ state,
    int n_qubits,
    int qubit_a,
    int qubit_b,
    Complex128 g00, Complex128 g01, Complex128 g02, Complex128 g03,
    Complex128 g10, Complex128 g11, Complex128 g12, Complex128 g13,
    Complex128 g20, Complex128 g21, Complex128 g22, Complex128 g23,
    Complex128 g30, Complex128 g31, Complex128 g32, Complex128 g33)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_groups = 1LL << (n_qubits - 2);
    if (tid >= n_groups) return;

    int lo_bit = qubit_a < qubit_b ? qubit_a : qubit_b;
    int hi_bit = qubit_a < qubit_b ? qubit_b : qubit_a;

    int64_t idx = tid;
    idx = ((idx >> lo_bit) << (lo_bit + 1)) | (idx & ((1LL << lo_bit) - 1));
    idx = ((idx >> hi_bit) << (hi_bit + 1)) | (idx & ((1LL << hi_bit) - 1));

    int64_t stride_a = 1LL << qubit_a;
    int64_t stride_b = 1LL << qubit_b;

    int64_t i00 = idx;
    int64_t i01 = idx | stride_a;
    int64_t i10 = idx | stride_b;
    int64_t i11 = idx | stride_a | stride_b;

    Complex128 a0 = state[i00];
    Complex128 a1 = state[i01];
    Complex128 a2 = state[i10];
    Complex128 a3 = state[i11];

    state[i00] = g00 * a0 + g01 * a1 + g02 * a2 + g03 * a3;
    state[i01] = g10 * a0 + g11 * a1 + g12 * a2 + g13 * a3;
    state[i10] = g20 * a0 + g21 * a1 + g22 * a2 + g23 * a3;
    state[i11] = g30 * a0 + g31 * a1 + g32 * a2 + g33 * a3;
}

void launch_apply_2q(Complex128* state, int n_qubits, int qubit_a, int qubit_b,
                     const Gate2Q& gate, hipStream_t stream) {
    int64_t n_groups = 1LL << (n_qubits - 2);
    int block_size = 256;
    int64_t grid_size = (n_groups + block_size - 1) / block_size;

    kernel_apply_2q<<<dim3(grid_size), dim3(block_size), 0, stream>>>(
        state, n_qubits, qubit_a, qubit_b,
        gate.m[0][0], gate.m[0][1], gate.m[0][2], gate.m[0][3],
        gate.m[1][0], gate.m[1][1], gate.m[1][2], gate.m[1][3],
        gate.m[2][0], gate.m[2][1], gate.m[2][2], gate.m[2][3],
        gate.m[3][0], gate.m[3][1], gate.m[3][2], gate.m[3][3]);
}

} // namespace quantum
