
#include <iostream>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <vector>

#pragma once


__global__ void transpose_kernel(const int8_t* src,
    int8_t* dst,
    int row,
    int col);

void transpose_kernelLauncher(const int8_t* input,
 int8_t* output,
    int row,
    int col,
    cudaStream_t stream);

void checkCudaStatus(cudaError_t status);

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)
