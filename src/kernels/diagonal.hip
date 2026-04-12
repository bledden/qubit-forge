#include "quantum/types.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <algorithm>

namespace quantum {

static constexpr unsigned int MAX_GRID = 1u << 20;

__global__ void kernel_apply_diagonal(
    Complex128* __restrict__ state,
    int64_t n_elements,
    int target,
    Complex128 phase)
{
    for (int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n_elements;
         tid += (int64_t)gridDim.x * blockDim.x) {
        if (tid & (1LL << target)) {
            Complex128 a = state[tid];
            state[tid] = a * phase;
        }
    }
}

__global__ void kernel_apply_diagonal_full(
    Complex128* __restrict__ state,
    int64_t n_elements,
    int target,
    Complex128 phase0,
    Complex128 phase1)
{
    for (int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n_elements;
         tid += (int64_t)gridDim.x * blockDim.x) {
        Complex128 a = state[tid];
        if (tid & (1LL << target)) {
            state[tid] = a * phase1;
        } else {
            state[tid] = a * phase0;
        }
    }
}

void launch_apply_diagonal(Complex128* state, int n_qubits, int target, const Gate1Q& gate,
                           hipStream_t stream) {
    int64_t n = 1LL << n_qubits;
    int block_size = 256;
    unsigned int grid = (unsigned int)std::min((int64_t)MAX_GRID, (n + block_size - 1) / block_size);

    bool phase0_is_identity = (std::abs(gate.m[0][0].x - 1.0) < 1e-15 &&
                                std::abs(gate.m[0][0].y) < 1e-15);

    if (phase0_is_identity) {
        kernel_apply_diagonal<<<dim3(grid), dim3(block_size), 0, stream>>>(
            state, n, target, gate.m[1][1]);
    } else {
        kernel_apply_diagonal_full<<<dim3(grid), dim3(block_size), 0, stream>>>(
            state, n, target, gate.m[0][0], gate.m[1][1]);
    }
}

} // namespace quantum
