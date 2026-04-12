#include "quantum/types.h"
#include <hip/hip_runtime.h>
#include <algorithm>

namespace quantum {

static constexpr unsigned int MAX_GRID = 4096u;

// Non-temporal load/store helpers for Complex128
// HIP's __builtin_nontemporal_load only works on fundamental types,
// so we load/store the two doubles individually.
__device__ __forceinline__ Complex128 nt_load(const Complex128* ptr) {
    const double* p = reinterpret_cast<const double*>(ptr);
    return {__builtin_nontemporal_load(p), __builtin_nontemporal_load(p + 1)};
}

__device__ __forceinline__ void nt_store(Complex128* ptr, Complex128 val) {
    double* p = reinterpret_cast<double*>(ptr);
    __builtin_nontemporal_store(val.x, p);
    __builtin_nontemporal_store(val.y, p + 1);
}

// --- Low qubit kernel (k < 5): coalesced access, NT loads/stores ---
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

        Complex128 a = nt_load(&state[lo]);
        Complex128 b = nt_load(&state[hi]);

        nt_store(&state[lo], g00 * a + g01 * b);
        nt_store(&state[hi], g10 * a + g11 * b);
    }
}

// --- Mid qubit kernel (5 <= k < 11): LDS tiling, grid-stride over tiles ---
// Global loads/stores use NT hints; LDS access is naturally fast.
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

        // NT load: global → LDS
        for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
            lds[i] = nt_load(&state[tile_start + i]);
        }
        __syncthreads();

        // Gate in LDS
        for (int64_t i = threadIdx.x; i < stride; i += blockDim.x) {
            Complex128 a = lds[i];
            Complex128 b = lds[i + stride];
            lds[i]          = g00 * a + g01 * b;
            lds[i + stride] = g10 * a + g11 * b;
        }
        __syncthreads();

        // NT store: LDS → global
        for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
            nt_store(&state[tile_start + i], lds[i]);
        }
        __syncthreads();
    }
}

// --- High qubit kernel (k >= 11): dual-region, NT loads/stores ---
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

        Complex128 a = nt_load(&state[lo]);
        Complex128 b = nt_load(&state[hi]);

        nt_store(&state[lo], g00 * a + g01 * b);
        nt_store(&state[hi], g10 * a + g11 * b);
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
