#ifndef ARRAY2D_CUDA_H
#define ARRAY2D_CUDA_H
#include <iostream>
#include "Array2D.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
using namespace std;

template <class T>
struct Cutype{
    T val;
};


template <>
template <class U>
class Array2D< Cutype<U> > {
public:
    Array2D(U* _data,
            const size_t& _nrows,
            const size_t& _ncols);
    Array2D(const Array2D<U>&);
    ~Array2D();
    size_t get_nrows() const {return *this->nrows;}
    size_t get_ncols() const {return *this->ncols;}
    size_t size()      const {return *this->N;}
    U* begin()const;
    U* end()const;
private:
    U* data;
    size_t* nrows;
    size_t* ncols;
    size_t* N;

};

template <>
template <class U>
Array2D< Cutype<U> >::Array2D(U* _data,
                    const size_t& _nrows,
                    const size_t& _ncols):data(_data){
    size_t N_tmp = _nrows * _ncols;

    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N    , sizeof(size_t));
    cudaMalloc((void**)&data , sizeof(U) * N_tmp);

    cudaMemcpy(nrows, &_nrows, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &_ncols, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(N,     &N_tmp , sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(data,  &_data , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);

    size_t a = 0;
    size_t* b;
    *b = a;
    cudaMemcpy(b, ncols , sizeof(size_t), cudaMemcpyDeviceToHost);
    cout << "*b = " << *b << endl;
    cout << "Super double secret construction with GPU memory allocation and copying" << endl;

};

template <>
template <class U>
Array2D< Cutype<U> >::Array2D(const Array2D<U>& other){
    size_t N_tmp = other.size();

    cudaMalloc((void**)&nrows, sizeof(size_t));
    cudaMalloc((void**)&ncols, sizeof(size_t));
    cudaMalloc((void**)&N    , sizeof(size_t));
    cudaMalloc((void**)&data , sizeof(U) * N_tmp);

    const size_t other_nrows = other.get_nrows();
    const size_t other_ncols = other.get_ncols();
    const size_t other_N = other.size();
    U *other_data = other.begin();

    cudaMemcpy(nrows, &other_nrows, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(ncols, &other_ncols, sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(N,     &other_N    , sizeof(size_t) , cudaMemcpyHostToDevice);
    cudaMemcpy(data,  &other_data , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);

    size_t a = 0;
    size_t* b;
    *b = a;
    cudaMemcpy(b, ncols , sizeof(size_t), cudaMemcpyDeviceToHost);
    cout << "*b = " << *b << endl;
    cout << "Super double secret COPY construction with GPU memory allocation and copying" << endl;

}

template <>
template <class U>
Array2D< Cutype<U> >::~Array2D(){
    cout << "Cleaning up cuda data" << endl;
    cudaFree(nrows);
    cudaFree(ncols);
    cudaFree(N);
    cudaFree(data);
}


#endif //ARRAY2D_CUDA_H
