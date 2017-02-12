#include "Array2D_CUDA.h"
#include "Array2D.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#define BLOCK_SIZE 1024

template <typename T>
__global__ pow2(T* in, T* out, size_t N){
    idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < N)out[idx] = in[idx] * in[idx];
}

template <class T>
void ArrayPow2_CUDA(const Array2D<T>& in, Array2D<T>& result) {
    Array2D< Cutype<T> > in_d(in);
    size_t N = in.size();
    pow2 <<< (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>> (in_d.begin(), in_d.begin(), in_d.size());
    cudaMemcpy(in, in_d, sizeof(T) * N, cudaMemcpyDevicetoHost);
}
