#include "quantum/types.h"
#include <hip/hip_runtime.h>

namespace quantum {

__global__ void kernel_apply_1q(
    Complex128* __restrict__ state,
    int n_qubits,
    int target,
    Complex128 g00, Complex128 g01,
    Complex128 g10, Complex128 g11)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_pairs = 1LL << (n_qubits - 1);
    if (tid >= n_pairs) return;

    int64_t stride = 1LL << target;

    // Map linear thread index to the state index with bit 'target' = 0
    int64_t lo = ((tid >> target) << (target + 1)) | (tid & (stride - 1));
    int64_t hi = lo | stride;

    Complex128 a = state[lo];
    Complex128 b = state[hi];

    state[lo] = g00 * a + g01 * b;
    state[hi] = g10 * a + g11 * b;
}

void launch_apply_1q(Complex128* state, int n_qubits, int target, const Gate1Q& gate,
                     hipStream_t stream) {
    int64_t n_pairs = 1LL << (n_qubits - 1);
    int block_size = 256;
    int64_t grid_size = (n_pairs + block_size - 1) / block_size;

    kernel_apply_1q<<<dim3(grid_size), dim3(block_size), 0, stream>>>(
        state, n_qubits, target,
        gate.m[0][0], gate.m[0][1],
        gate.m[1][0], gate.m[1][1]);
}

} // namespace quantum
