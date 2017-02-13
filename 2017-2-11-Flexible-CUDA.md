---
layout: dark-post
title: The CPU/GPU Switcheroo&#58; Flexible Extension of C++ Template Libraries with CUDA
description: "Functional Programming Techniques and Template Specialization in CUDA"
tags: [C++, CUDA, NVIDIA, GPU, Software Engineer, Functional Programming]
categories: [image]
---
<figure class="half center">
        <a href="/images/Flexible-CUDA/NVIDIA-CUDA.jpg"><img src="/images/Flexible-CUDA/NVIDIA-CUDA.jpg" alt=""></a> </figure class="half center">

Consider the following scenario. You are a developer for a large C++ template library that performs computationally intensive processing on custom, complex data types, and you want to accelerate some of the slower functions with GPUs through CUDA. However, you don't want to suddenly introduce the CUDA toolkit as a hard dependency because you expect many of your users will continue to use CPU-only implementations. You simply want to provide GPU acceleration to those who wish to leverage it. What's the best way to accomplish this from a software engineering perspective?  

<!-- more -->

A really bad way would be to maintain two separate 
projects. A better way would be to provide
compile-time options to either target the 
GPU or CPU implementations of each function;
however, that would force users to pick a 
version and stick with it, with recompilation 
required to change targets. Furthermore, a 
function might run faster on the CPU for small
array sizes, and faster on the GPU for large 
ones, so ideally the user would have access to both
implementations. Okay, so what if we just provide 
two functions `foo_cpu()` and `foo_gpu()` so that 
the developer can choose which version of `foo()` 
to use? This solution is close to a good answer, 
but without one additional improvement you're now
expecting somebody to go through their (potentially
huge) codebase and change function names and possibly
introduce extra logical statements. As we will 
see, there is a functional programming solution 
that allows for infectious runtime determination 
of implementation that requires minimal change to
    existing codebases.  

A second concern relates to the development of the CUDA 
implementation of the library itself. CUDA is a 
very low-level language, and if our library has 
complex data structures then it can be difficult 
to manage data allocation and memory transfers. 
On the host-side (CPU), C++ classes make development
easier by abstracting away this type of housekeeping,
and ideally we want to do the same on the 
device-side (GPU) so that we can push our
accelerated library out faster. With specialized template
metaprogramming, we can create a CUDA-interface 
to our existing classes that greatly simplifies 
GPU development by abstracting away the tedium of 
dealing with low-level GPU memory management 
with `cudaMalloc`, `cudaMemcpy`, etc.

In the rest of this article I'll walk through an example of how this problem might come up in practice and how to solve it. 

## Demonstration
To demonstrate solutions to both of these topics, let's consider a simple example of a template library that works on 2D arrays, 
and a single function that squares every element in the array.
The actual CUDA code for this is trivial, but is not the 
main focus. The overall software design concept is more important.  

*You can find the code for this article [here](https://github.com/apryor6/blog-flexible-cuda)*

### The C++ Code
First, we build a basic 2D array class.   

*Disclaimer: This is an example and not intended 
to be used in a real library without further
modification. You generally don't want to be 
passing/freeing raw pointers in 
constructors/destructors without some sort of 
reference counting mechanism. There's also no move 
constructor, no operator[] overload, error 
checking, etc.*  


~~~ c++
//Array2D.h

#ifndef ARRAY2D_H
#define ARRAY2D_H

template <class T>
class Array2D {
public:
    Array2D(T* _data,
            const size_t& _nrows,
            const size_t& _ncols);
    Array2D(const Array2D<T>& other);
    Array2D<T>& operator=(const Array2D<T>& other);
    ~Array2D(){delete[] this->data;}
    size_t get_nrows() const {return this->nrows;}
    size_t get_ncols() const {return this->ncols;}
    size_t size()      const {return this->N;}
    T* begin(){return data;}
    T* begin()const{return data;}
    T* end(){return data + this->size();}
    T* end()const{return data + this->size();}
private:
    T* data;
    size_t nrows;
    size_t ncols;
    size_t N;

};

template <class T>
Array2D<T>::Array2D(T* _data,
                    const size_t& _nrows,
                    const size_t& _ncols):data(_data), nrows(_nrows), ncols(_ncols){
    this->N = _nrows * _ncols;
};


template <class T>
Array2D<T>::Array2D(const Array2D<T>& other):nrows(other.nrows), ncols(other.ncols), N(other.N){
    data = new T[N];
    auto i = this->begin();
    for (auto& o:other)*i++=o;
};


template <class T>
Array2D<T>& Array2D<T>::operator=(const Array2D<T>& other){
    this->ncols = other.ncols;
    this->ncols = other.nrows;
    this->N     = other.N;

    // here should compare the sizes of the arrays and reallocate if necessary
    delete[] data;
    data = new T[N];
    auto i = this->begin();
    for (auto& o:other)*i++=o;
    return *this;
};

#endif //ARRAY2D_H
~~~

The class stores a pointer to the raw data as well as the number of rows and columns for the 2D array it represents. There's a copy and assignment overload, and I define `begin()` and `end()` methods so I can use range-based for syntax. The destructor deletes the pointer, so `Array2D` objects should be constructed with a pointer returned from `new` or similar.

Now that the class exists, the implementation for `ArrayPow2`, a function that squares every element in an `Array2D`, is trivial.

~~~ c++
//ArrayPow2.h

#include <algorithm>
#include "Array2D.h"
template <class T>
void ArrayPow2(Array2D<T>& in, Array2D<T>& result){
    std::transform(in.begin(), in.end(), result.begin(), [](const T& a){return a*a;});
}
~~~

`std::transform` applies the function-like object in the 4th argument to every element in the range from the first to second arguments, and stores the results in the 3rd. 
Here I used a lambda function to express the squaring operation.  

A simple driver program verifies that what we have 
so far is okay:

~~~c++
// driver1.cpp

#include <iostream>
#include "Array2D.h"
#include "ArrayPow2.h"

using namespace std;
int main() {
    Array2D<float> arr(new float[100], 10, 10);
    int a = 2;
    for (auto& i:arr)i=++a;
    Array2D<float> result(arr);
    ArrayPow2(arr, result);

    cout << "arr[0]   = " << *arr.begin() << endl;
    cout << "arr[0]^2 = " << *result.begin() << endl;
    return 0;
}
~~~ 
~~~
arr[0]   = 3
arr[0]^2 = 9
~~~

## CUDA Template Specialization


To make developing/transcribing the CUDA version of our code easier,
we first want to implement a CUDA version of our `Array2D` class. This can
be done by creating a specialization of `Array2D` for CUDA. To differentiate
between the CPU and GPU versions of `Array2D`, I introduce a trivial class 
that just contains a single value. 

~~~ c++
template <class T>
struct Cutype{
    T val;
};
~~~

The effect this has is subtle, but powerful. I can now create an `Array2D< Cutype<T> >` that will instantiate an entirely
different class than `Array2D<T>`, and in this case that specialization
will be used to abstract away calls to `cudaMalloc`, `cudaMemcpy`, etc. For my implementation, both types 
will contain data of type `float`, but you might also decide that
the CUDA versions store data as `Cutype <float>`. There is no additional overhead
for the extra layer of this datatype, it gets optimized away in the compiler
and replaced as `float` (you can check the outputted assembly if you don't believe me), so
you could always cast one type to the other.
An additional positive side effect of this is that `Cutype` then serves
as a compile-time error boundary to prevent accidental assignment or 
dereferencing of data on the device from the host, which results in a 
segmentation fault at runtime that can be hard to diagnose. Either choice here seems fine to me.
 
 Here is the full specialization (as before this class
 is incomplete, but contains the code relevant to this discussion):
 
 ~~~ c++
 // Array2D_CUDA.h
 
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


template <class U>
class Array2D< Cutype<U> > {
public:
    Array2D(U* _data,
            const size_t& _nrows,
            const size_t& _ncols);
    Array2D(const Array2D<U>&);
Array2D< Cutype<U> >& operator=(const Array2D<U>& other);
    ~Array2D();
    size_t get_nrows() const {return *this->nrows;}
    size_t get_ncols() const {return *this->ncols;}
    size_t size()      const {return *this->N;}
    U* begin()const{return data;}
    U* end()const{return data + this->size();}
    U* begin(){return data;}
    U* end(){return data + this->size();}
private:
    U* data;
    size_t* nrows;
    size_t* ncols;
    size_t* N;

};

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
    cudaMemcpy(data,  _data  , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
};

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
    cudaMemcpy(data,  other_data  , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
}


template <class U>
Array2D< Cutype<U> >& Array2D< Cutype<U> >::operator=(const Array2D<U>& other){
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
    cudaMemcpy(data,  other_data , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);

    return *this;
}


template <class U>
Array2D< Cutype<U> >::~Array2D(){
    cudaFree(nrows);
    cudaFree(ncols);
    cudaFree(N);
    cudaFree(data);
}


#endif //ARRAY2D_CUDA_H
~~~

Most of the interface should look familiar. A key difference
is that in place of the assignment and copy constructors
for `Array2D<T>` that took in another `Array2D<T>` I have
defined how an `Array2D< Cutype<T> >` is copied from an 
`Array2D<T>`. This way we can construct a 2D array object
on the GPU simply by passing in an existing host-side array.
The calls to `delete[]` and `new` are replaced by `cudaMalloc`
and `cudaFree`, and the data copying is handled with `cudaMemcpy`.
Other than the way this object is built, the way you interact with 
such a class is largely unchanged. Taking the time to build a class like 
this will save a lot of time in the long run, rather than having to manually deal
with memory copying for every new kernel.
 
## CUDA Implementation
 
 Although we have written a bunch of code for the GPU, we haven't actually
 written any CUDA yet -- we've just used the CUDA runtime API. To make a 
 version of `ArrayPow2` from earlier on the GPU, we create a C++ wrapper function
 with the same signature in a header file.
 
 ~~~ c++   
 //ArrayPow2_CUDA.cuh    
 
 #include "Array2D_CUDA.h"
 #include "Array2D.h"
 
 template <class T>
 void ArrayPow2_CUDA(Array2D<T>& in, Array2D<T>& result);
 ~~~
 
 And then the implementation in a .cu file
 
 ~~~ c++
 //ArrayPow2_CUDA.cu
 
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
  std::cout << "Using the GPU version\n";
  Array2D< Cutype<T> > in_d(in);
  size_t N = in.size();
  pow2 <<< (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>> (in_d.begin(), in_d.begin(), in.size());
  cudaMemcpy(in.begin(), in_d.begin(), sizeof(T) * N, cudaMemcpyDeviceToHost);
}
 
 template void ArrayPow2_CUDA(Array2D<float>&, Array2D<float>&);
 template __global__ void pow2(float*, float*, size_t);
~~~

 Now you can see where the template specialization really shines.
 The wrapper function is called in exactly the same way as our original CPU
 implementation, and a GPU copy of the array is created in just one line. 
 We run the kernel using the exact same `.begin()` and `.end()` calls that
 we used with `std::transform`, and then copy the result back. I explicitly
 used `cudaMemcpy` here because it's just one line, but you could always 
 also write a helper function to hide that if you please.  
 
 The one important, and somewhat painful, bit is the necessary addition of the last two lines.  
 If you have worked with templates much, you'll quickly find that it is a pain to separate
 prototypes and implementations into different files. The reason is because the template itself
 just defines how the compiler can construct a class *from a given type*. But if the template code
 exists in its own compilation unit, then it gets compiled before it knows which classes actually
 need to be instantiated. By the time the linker is trying to connect your `main()` with whatever templates
 it needs, it usually will yell at you for "undefined symbols", meaning that it is looking for an object that
 hasn't been defined.   In C++, there are some ways you can get around this, but usually the easiest solution is 
 to put the full template implementation in the header file. This way the classes that need to be instantiated 
 are guaranteed to be visible at compile time. With CUDA code this solution doesn't work 
 because the compilation of CUDA and C++ by `nvcc` are inherently decoupled, so there must be multiple files.
 
 The solution is to forcibly
 instantiate the template types that will be used, which I did in those last two lines by forward declaring a template specialization that I
 intend to use in my library.

## Putting it all together 

Now that we have implemented our CUDA template specialization and kernel, 
we want to tie all of this back into our original library in a way that doesn't add any CUDA dependencies to users who just want to continue using the CPU-only version 
and provides a way to choose between CPU or GPU implementations at runtime without demanding
 that developers go through their code and change function names everywhere. The solution
 I used for integrating the GPU code is the following: 
 
 1. Change the name of the existing C++ function. I append with `_CPU` for clarity.
 2. Create a pointer to a function with the same signature as the pure C++ version and 
 the GPU version wrapper (remember, they have the same signature). Name it whatever the C++ function was called before you changed it.
 3. Add a compiler directive, ENABLE_GPU. If it is not defined at compile time, don't include
 anything related to CUDA, and set the function pointer equal to the CPU version. If it is enabled, 
 then introduce an additional runtime check for a command line input and use that to set the function pointer
  to either the CPU or CUDA version, appropriately.  
  
All of this goes into the driver program, which follows

~~~ c++
// main.cpp

#include <iostream>
#include <cstring>
#include "Array2D.h"
#include "ArrayPow2_CPU.h"

#ifdef ENABLE_GPU
#include "Array2D_CUDA.h"
#include "ArrayPow2_CUDA.cuh"
#endif //ENABLE_GPU

template <class T>
using ArrayPow2_F = void(*)(Array2D<T>&, Array2D<T>&);
ArrayPow2_F<float> ArrayPow2;
using namespace std;


int main(int argc, char** argv) {
#ifdef ENABLE_GPU
    if (argc>2 && !strcmp(argv[1],"gpu")){
        if (!strcmp(argv[2],"1")){
            ArrayPow2 = ArrayPow2_CUDA;
        } else{
            ArrayPow2 = ArrayPow2_CPU;
	}
    } else
    {
        ArrayPow2 = ArrayPow2_CUDA;
    }
#else
    ArrayPow2 = ArrayPow2_CPU;
#endif //ENABLE_GPU

    Array2D<float> arr(new float[120], 60, 2);
    int a = 2;
    for (auto& i:arr)i=++a;
    cout << "arr[0]" << *arr.begin()<< endl;
    Array2D<float> result(arr);
    cout << "arr[0]" << *arr.begin()<< endl;
    cout << "result[0]" << *result.begin()<< endl;

    ArrayPow2(arr, result);

    cout << "arr[0]   = " << *arr.begin() << endl;
    cout << "arr[0]^2 = " << *result.begin() << endl;
    return 0;
}
~~~

I also add a short Makefile for convenience 

~~~
//Makefile

all: cpu
clean:
	-rm demo
cpu:
	g++ -std=c++11 main.cpp -o demo
gpu:
	nvcc -std=c++11 main.cpp -D ENABLE_GPU ArrayPow2_CUDA.cu -o demo
~~~

With this setup we now have three ways to compile and run the program:

1. Change nothing. Compile with `make`
2. Compile for the gpu with `make gpu`. The default behavior
is now to run on the GPU.
3. Run the program compiled in option 2) on the CPU by adding `gpu 0` to the command line call

Here's the output from each of these (I added a line to `ArrayPow2_CPU` and `ArrayPow2_GPU` to print
which version was triggered):

~~~ 
$  make
$ ./demo
Using the CPU version
arr[0]   = 3
arr[0]^2 = 9
~~~

~~~ 
$  make gpu
$ ./demo
Using the GPU version
arr[0]   = 0
arr[0]^2 = 3
~~~

~~~ 
$  make gpu
$ ./demo gpu 0
Using the CPU version
arr[0]   = 3
arr[0]^2 = 9
~~~

If you think about it, this is an incredibly powerful programming pattern. We still only have
to maintain a single code-base as we continue to add more and more GPU functions. Users
that don't have or want to use CUDA GPU support for any reason are not affected in any way. Most importantly,
for those who do use the GPU support, the technique of replacing each accelerated
function with a function pointer and modifying the name of the original C++ version is infectious. A project
could be millions of lines long, and contain many calls to a function like 
`ArrayPow2`, but once it is replaced with a function pointer that is set to the GPU implementation, the entire project would immediately use the GPU version 
 with no further changes. Now, in this example I only considered a single function, but the extension
 to many is trivial. All that is required is configuration and assignment of the function pointers at the beginning of the end-users program. 
 Furthermore, this could be done however the developer wanted.   
 
 For example, a check 
 could be added for the array size, and the CPU or GPU version alternatively used based
 on optimizaton/tuning results. Another use case is if code was being run on a large 
 cluster where some nodes have GPUs, but others do not. A simple query can be run to check if a valid
 GPU is found and to use it, and then if not to fall back to the CPU version. A data center could be concerned
 about power consumption and dynamically change where data is processed in real-time. There are a lot of
 possibilies, most of which I probably haven't thought of. But after all, writing 
 software that is flexible enough that it can be easily adapted to fit a particular user's
 needs without the library developer anticipating them is a sign of good software engineering. 
 
