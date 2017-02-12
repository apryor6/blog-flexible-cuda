#include "Array2D_CUDA.h"
#include "Array2D.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#define BLOCK_SIZE 1024

template <class T>
__global__ void pow2(T* in, T* out, size_t N){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < N)out[idx] = in[idx] * in[idx];
}

template <class T>
void ArrayPow2_CUDA(Array2D<T>& in, Array2D<T>& result) {
    Array2D< Cutype<T> > in_d(in);
    size_t N = in.size();
    pow2 <<< (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>> (in_d.begin(), in_d.begin(), in.size());
    //double* in2 = in_d.begin();
    //double* out2 = in2;
    //pow2 <<< (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>> (in2, out2, in.size());
    T* arr_h = in.begin();
    T* arr_d = in_d.begin();
    cudaMemcpy(arr_h, arr_d, sizeof(T) * N, cudaMemcpyDeviceToHost);
}

template void ArrayPow2_CUDA(Array2D<float>&, Array2D<float>&);
template __global__ void pow2(float*, float*, size_t);

/*
__global__ void pow2(double* in, double* out, size_t N){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < N)out[idx] = in[idx] * in[idx];
}

void ArrayPow2_CUDA(Array2D<double>& in, Array2D<double>& result) {
    Array2D< Cutype<double> > in_d(in);
    size_t N = in.size();
    double* arr_h = in.begin();
    double* arr_d = in_d.begin();
    pow2 <<< (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>> (arr_h, arr_h, N);
    cudaMemcpy(arr_h, arr_d, sizeof(double) * N, cudaMemcpyDeviceToHost);
}
*/
