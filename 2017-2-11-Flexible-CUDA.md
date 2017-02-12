---
layout: dark-post
title: The CPU/GPU Switcheroo: Flexible Extension of C+ Template Libraries with CUDA
description: "Functional Programming Techniques and Template Specialization in CUDA"
tags: [C++, CUDA, NVIDIA, GPU, Software Engineer, Functional Programming]
---

Consider the following scenario. You are a developer for a large C++ template library that performs computationally intensive processing on custom, complex data types and classes, and you want to accelerate some of the slower functions with GPUs through CUDA. However, you don't want to suddenly introduce the CUDA toolkit as a hard dependency because you expect many of your users will continue to use CPU-only implementations. You simply want to provide GPU acceleration to those who wish to leverage it. What's the best way to accomplish this from a software engineering perspective?  

A really bad way would be to maintain two separate projects. A better way would be to provide compile-time options to either target the GPU or CPU implementations of each function; however, that would force users to pick a version and stick with it, with recompilation required to change targets. Furthermore, a function might run faster on the CPU for small array sizes, and faster on the GPU for large ones, so ideally the user would have access to both implementations. Okay, so what if we just provide two functions `foo_cpu()` and `foo_gpu()` so that the developer can choose which version of `foo()` to use? This solution is close to a good answer, but without one additional improvement you're now expecting somebody to go through their (potentially huge) codebase and change function names and introduce extra logical statements. As we will see, there is a functional programming solution that allows for infectious runtime determination of implementation that requires minimal change to existing codebases.  

A second concern relates to the development of the CUDA implementation of the library itself. CUDA is a very low-level language, and if our library has complex data structures than it can be difficult to manage data allocation and memory transfers. On the host-side (CPU), C++ classes make development easier by abstracting away this type of housekeeping, and ideally we want to do the same on the device-side (GPU) so that we can push our accelerated library out faster. With template metaprogramming, we can create a CUDA-interface to our existing classes that greatly simplifies GPU development.  

In the rest of this article I'll walk through an example of how this problem might come up in practice and how to solve it. 

## Demonstration
To demonstrate solutions to both of these problems (implementing code for the GPU and template specialization for the GPU), let's consider a simple example of a template library that works on 2D arrays, and a function that squares every element in the array.

### The C++ Code
First, we build a basic 2D array class.   

*Disclaimer: This is an example and not intended to be used in a real library without further modification. You generally don't want to be passing/freeing raw pointers in constructors/destructors without some sort of reference counting mechanism. There's no move constructor, no operator[] overload, error checking, etc.*  


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

`std::transform` applies the function-like object in the 4th argument to every element from the first to second arguments, and stores the results in the 3rd. Here I used a lambda function.  

Here's a simple driver to test what we have so far. 

~~~c++
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

