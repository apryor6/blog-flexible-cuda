#include <algorithm>
#include "Array2D.h"
template <class T>
void ArrayPow2_CPU(Array2D<T>& in, Array2D<T>& result){
    std::cout << "Using the CPU version\n";
    std::transform(in.begin(), in.end(), result.begin(), [](const T& a){return a*a;});
}
