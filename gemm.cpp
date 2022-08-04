#include "gemm.h"

template<typename T>
T Elementwise(const T* src1, const T* src2, int l) {
    T dst;
    for (int i = 0; i < l; ++i) {
        dst += src1[i] * src2[i];
    }
    return dst;
}


template<typename T>
void Gemm(const T* A, const T* Bt, T* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i*n + j] = Elementwise(&A[i * k], &Bt[j * k], k);
        }
    }
}