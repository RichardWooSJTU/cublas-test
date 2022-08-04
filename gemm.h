#include <iostream>
#include <vector>

template<typename T>
T Elementwise(const T* src1, const T* src2, int l);

template<typename T>
void Gemm(const T* A, const T* Bt, T* C, int m, int n, int k);