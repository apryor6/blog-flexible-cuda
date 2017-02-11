#include <iostream>
#include "Array2D.h"
#include "Array2D_CUDA.h"
using namespace std;
int main() {
    Array2D<double> arr(new double[120], 60, 2);
    //for (auto& i:arr)cout << i << '\n';


    #ifdef ENABLE_GPU
    Array2D< Cutype<double> > arr_c(new double[100], 10, 10);
    #endif

    Array2D< Cutype<double> > arr_other(arr);


    //for (auto& i:arr)i=++a;
    //for (auto& i:arr)cout << i << '\n';
    return 0;
}