#include <iostream>
#include "Array2D.h"
#include "ArrayPow2.h"
#include "Array2D_CUDA.h"

//#include "Array2D_CUDA.h"
using namespace std;
int main() {
    Array2D<double> arr(new double[120], 60, 2);
    int a = 2;
    for (auto& i:arr)i=++a;

    cout << *arr.begin() << endl;

    Array2D<double> result(arr);
    //result = arr;
    ArrayPow2(arr, result);

    cout << *result.begin() << endl;

    cout << "GPU now" << endl;
    ArrayPow2_CUDA(arr, result);

    cout << *result.begin() << endl;



    return 0;
}

