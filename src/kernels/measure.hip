#include "quantum/types.h"
#include <hip/hip_runtime.h>

namespace quantum {

__global__ void kernel_probabilities(
    const Complex128* __restrict__ state,
    double* __restrict__ probs,
    int64_t n)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    Complex128 a = state[tid];
    probs[tid] = a.x * a.x + a.y * a.y;
}

__global__ void kernel_reduce_sum(
    const double* __restrict__ data,
    double* __restrict__ out,
    int64_t n)
{
    extern __shared__ double sdata[];
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (tid < n) ? data[tid] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void launch_probabilities(const Complex128* state, double* probs, int64_t n,
                          hipStream_t stream) {
    int block = 256;
    int64_t grid = (n + block - 1) / block;
    kernel_probabilities<<<dim3(grid), dim3(block), 0, stream>>>(state, probs, n);
}

double launch_total_probability(const double* probs, int64_t n, hipStream_t stream) {
    double* d_sum;
    hipMalloc(&d_sum, sizeof(double));
    hipMemset(d_sum, 0, sizeof(double));

    int block = 256;
    int64_t grid = (n + block - 1) / block;
    kernel_reduce_sum<<<dim3(grid), dim3(block), block * sizeof(double), stream>>>(
        probs, d_sum, n);

    double result;
    hipMemcpy(&result, d_sum, sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_sum);
    return result;
}

} // namespace quantum
