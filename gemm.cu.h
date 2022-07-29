#include "cublas.cu.h"
#include "find_algo.cu.h"
#include <ctime>
#include <sys/time.h>

#pragma once

void GemmInt8(int m, int k, int n, 
    cublasLtHandle_t ltHandle
    );

void GemmFp16(int m, int k, int n, 
    cublasLtHandle_t ltHandle);

void CublasGemmFp16(int m, int k, int n, cublasHandle_t handle);
void CublasGemmInt8(int m, int k, int n, cublasHandle_t handle);

inline double diffTime(timeval start, timeval end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}