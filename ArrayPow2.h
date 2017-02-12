#include <algorithm>
#include "Array2D.h"
template <class T>
void ArrayPow2(Array2D<T>& in, Array2D<T>& result){
    std::transform(in.begin(), in.end(), result.begin(), [](const T& a){return a*a;});

}