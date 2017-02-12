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

