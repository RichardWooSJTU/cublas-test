#include <iostream>
#include <sys/time.h>
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

inline double diffTime(timeval start, timeval end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}
