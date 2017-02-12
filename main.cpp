#include <iostream>
#include "Array2D.h"
#include "ArrayPow2.h"
#include "Array2D_CUDA.h"
#include "ArrayPow2_CUDA.cuh"
template <class T>
using ArrayP = void(*)(Array2D<T>&, Array2D<T>&);
//#include "Array2D_CUDA.h"
using namespace std;
int main() {
ArrayP<double> F;
    Array2D<double> arr(new double[120], 60, 2);
    int a = 2;
    for (auto& i:arr)i=++a;

    cout << *arr.begin() << endl;

    Array2D<double> result(arr);
    //result = arr;
    F = ArrayPow2;
    F(arr, result);

    cout << *result.begin() << endl;

    cout << "GPU now" << endl;
    F = ArrayPow2_CUDA;
    F(arr, result);

    cout << *result.begin() << endl;



    return 0;
}

