#include <iostream>
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


int main(int argc, const char** argv) {

#ifdef ENABLE_GPU
    if (argc>1){
        if (argv[1]=="-gpu"){
            ArrayPow2 = (argv[2]=='1') ? ArrayPow2_GPU : ArrayPow2_CPU;
        }
     } else
    {
        ArrayPow2 = ArrayPow2_GPU;
    }
#else
    ArrayPow2 = ArrayPow2_CPU;
#endif //ENABLE_GPU

    Array2D<float> arr(new float[120], 60, 2);
    int a = 2;
    for (auto& i:arr)i=++a;

    Array2D<float> result(arr);
    ArrayPow2(arr, result);

    cout << "arr[0]   = " << *arr.begin() << endl;
    cout << "arr[0]^2 = " << *result.begin() << endl;
    return 0;
}

