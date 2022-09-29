#include <iostream>
#include <vector>

#pragma once


template<typename T>
void Transpose(const T* a, T* b, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			b[j * rows + i] = a[i * cols + j];
		}
	}
	return;
}

template<typename InT, typename OutT>
OutT Elementwise(const InT* src1, const InT* src2, int l) {
    OutT dst = 0;
    for (int i = 0; i < l; ++i) {
        // std::cout << static_cast<OutT>(src2[i]) << std::endl;
        dst += static_cast<OutT>(src1[i]) * static_cast<OutT>(src2[i]);
    }
    return dst;
}


template<typename InT, typename OutT>
void Gemm(const InT* A, const InT* B, OutT* C, int m, int n, int k) {
    std::vector<InT> Bt_vec(k * n);
    // Transpose(B, Bt_vec.data(), k, n);
    // InT* Bt = Bt_vec.data();
    const InT* Bt = B;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i*n + j] = Elementwise<InT, OutT>(&A[i * k], &Bt[j * k], k);
        }
    }
}