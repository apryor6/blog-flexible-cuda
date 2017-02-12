#include <iostream>
#include "Array2D.h"
#include "ArrayPow2_CPU.h"
#include "Array2D_CUDA.h"
#include "ArrayPow2_CUDA.cuh"


template <class T>
using ArrayPow2_F = void(*)(Array2D<T>&, Array2D<T>&);
ArrayPow2_F<float> ArrayPow2;
using namespace std;


int main() {
    Array2D<float> arr(new float[120], 60, 2);
    int a = 2;
    for (auto& i:arr)i=++a;

    cout << *arr.begin() << endl;

    Array2D<float> result(arr);
    //result = arr;
    ArrayPow2 = ArrayPow2_CPU;
    ArrayPow2(arr, result);

    cout << *result.begin() << endl;

    cout << "GPU now" << endl;
    ArrayPow2 = ArrayPow2_CUDA;
    ArrayPow2(arr, result);

    cout << *result.begin() << endl;



    return 0;
}

