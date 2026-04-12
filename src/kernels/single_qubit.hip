#include "quantum/types.h"
#include <hip/hip_runtime.h>
#include <algorithm>

namespace quantum {

// Grid-stride loops: cap grid to keep total threads < 2^32.
// MI300X has 304 CUs × 4 SIMDs = 1216 SIMD units. Target 4 waves/SIMD = 4864 waves.
// At 256 threads/block = 4 waves/block → ~1216 blocks saturates the GPU.
// Use 4096 blocks for safety margin + grid-stride amortization.
static constexpr unsigned int MAX_GRID = 4096u;

// --- Low qubit kernel (k < 5): coalesced access, grid-stride loop ---
// 4 waves/SIMD sweet spot for bandwidth-bound (from MXFP4 competition)
__global__ __launch_bounds__(256, 4)
void kernel_apply_1q_low(
    Complex128* __restrict__ state,
    int64_t n_pairs,
    int target,
    Complex128 g00, Complex128 g01,
    Complex128 g10, Complex128 g11)
{
    int64_t stride = 1LL << target;

    for (int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n_pairs;
         tid += (int64_t)gridDim.x * blockDim.x) {

        int64_t lo = ((tid >> target) << (target + 1)) | (tid & (stride - 1));
        int64_t hi = lo | stride;

        // Non-temporal loads for large state vectors (>L2 cache = 256MB = ~24 qubits)
        Complex128 a = state[lo];
        Complex128 b = state[hi];

        state[lo] = g00 * a + g01 * b;
        state[hi] = g10 * a + g11 * b;
    }
}

// --- Mid qubit kernel (5 <= k < 11): LDS tiling, grid-stride over tiles ---
// LDS bandwidth is ~100 TB/s vs 5.3 TB/s HBM — worth the staging cost.
__global__ __launch_bounds__(256, 4)
void kernel_apply_1q_mid(
    Complex128* __restrict__ state,
    int64_t n_tiles,
    int target,
    Complex128 g00, Complex128 g01,
    Complex128 g10, Complex128 g11)
{
    int64_t stride = 1LL << target;
    int64_t tile_size = stride << 1;

    extern __shared__ Complex128 lds[];

    for (int64_t tile_idx = blockIdx.x; tile_idx < n_tiles; tile_idx += gridDim.x) {
        int64_t tile_start = tile_idx * tile_size;

        // Coalesced global → LDS
        for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
            lds[i] = state[tile_start + i];
        }
        __syncthreads();

        // Apply gate in LDS (100 TB/s bandwidth)
        for (int64_t i = threadIdx.x; i < stride; i += blockDim.x) {
            Complex128 a = lds[i];
            Complex128 b = lds[i + stride];
            lds[i]          = g00 * a + g01 * b;
            lds[i + stride] = g10 * a + g11 * b;
        }
        __syncthreads();

        // Coalesced LDS → global
        for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
            state[tile_start + i] = lds[i];
        }
        __syncthreads();
    }
}

// --- High qubit kernel (k >= 11): dual-region, grid-stride loop ---
// Paired amplitudes are far apart — no LDS benefit. Stream through HBM.
__global__ __launch_bounds__(256, 4)
void kernel_apply_1q_high(
    Complex128* __restrict__ state,
    int64_t n_pairs,
    int target,
    Complex128 g00, Complex128 g01,
    Complex128 g10, Complex128 g11)
{
    int64_t stride = 1LL << target;

    for (int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n_pairs;
         tid += (int64_t)gridDim.x * blockDim.x) {

        int64_t lo = ((tid >> target) << (target + 1)) | (tid & (stride - 1));
        int64_t hi = lo | stride;

        Complex128 a = state[lo];
        Complex128 b = state[hi];

        state[lo] = g00 * a + g01 * b;
        state[hi] = g10 * a + g11 * b;
    }
}

void launch_apply_1q(Complex128* state, int n_qubits, int target, const Gate1Q& gate,
                     hipStream_t stream) {
    int64_t n_pairs = 1LL << (n_qubits - 1);

    Complex128 g00 = gate.m[0][0], g01 = gate.m[0][1];
    Complex128 g10 = gate.m[1][0], g11 = gate.m[1][1];

    int block_size = 256;

    if (target < 5) {
        unsigned int grid = (unsigned int)std::min((int64_t)MAX_GRID, (n_pairs + block_size - 1) / block_size);
        kernel_apply_1q_low<<<dim3(grid), dim3(block_size), 0, stream>>>(
            state, n_pairs, target, g00, g01, g10, g11);
    } else if (target < 11) {
        int64_t tile_size = 1LL << (target + 1);
        int64_t n_tiles = 1LL << (n_qubits - target - 1);
        size_t lds_bytes = tile_size * sizeof(Complex128);
        int threads = (int)std::min((int64_t)256, tile_size / 2);
        unsigned int grid = (unsigned int)std::min(n_tiles, (int64_t)MAX_GRID);
        kernel_apply_1q_mid<<<dim3(grid), dim3(threads), lds_bytes, stream>>>(
            state, n_tiles, target, g00, g01, g10, g11);
    } else {
        unsigned int grid = (unsigned int)std::min((int64_t)MAX_GRID, (n_pairs + block_size - 1) / block_size);
        kernel_apply_1q_high<<<dim3(grid), dim3(block_size), 0, stream>>>(
            state, n_pairs, target, g00, g01, g10, g11);
    }
}

} // namespace quantum
