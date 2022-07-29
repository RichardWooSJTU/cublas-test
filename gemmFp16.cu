#include "gemm.cu.h"
#include <fstream>
#include <iostream>

void GemmFp16(int m, int k, int n, 
    cublasLtHandle_t ltHandle) {
    cublasStatus_t status;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    half * A, * B, *C;
    cudaMalloc(&A, m * k * sizeof(half));
    cudaMalloc(&B, k * n * sizeof(half));
    cudaMalloc(&C, m * n * sizeof(half));

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int lda = m, ldb = k, ldc = m;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    half alpha = (half)1.0f, beta = (half)0.0f;
    void* workspace;
    size_t workspaceSize = 4 * (m * k + m * k + n * k);
    cudaMalloc(&workspace, workspaceSize);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    if (cudaEventCreate(&startEvent, cudaEventBlockingSync) != cudaSuccess) {
        std::cout << " cudaEventCreate GG with status " << std::endl;
    }
    if (cudaEventCreate(&stopEvent, cudaEventBlockingSync) != cudaSuccess) {
        std::cout << " cudaEventCreate GG with status "<< std::endl;
    }


    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
     cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
     cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
     cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
     cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
     cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
     cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc);
     ////////////
    // Select //
    ////////////
    auto results = FindAlgo(ltHandle, m, n, k, A, B, C, operationDesc, Adesc, Bdesc, Cdesc, CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F);
    // std::ofstream outfile;
    // outfile.open("res.csv", std::ios::app);
    // int i = 0;
    // while (results[i].time == 0) i++;
    // outfile << "fp16, " << m << ", "  << k << ", " << n << ", " << results[i].time << ", " << results[i].wavesCount << ", " << results[i].workspaceSize << std::endl;

    // outfile.close();
    std::cout << "finsh select" << std::endl;

    //  cudaError_t err;
    //  err = cudaEventRecord(startEvent, 0);
    //  status = cublasLtMatmul(ltHandle,
    //     operationDesc,
    //     &alpha,
    //     A,
    //     Adesc,
    //     B,
    //     Bdesc,
    //     &beta,
    //     C,
    //     Cdesc,
    //     C,
    //     Cdesc,
    //     NULL,
    //     workspace,
    //     workspaceSize,
    //     0);
    // if (status != CUBLAS_STATUS_SUCCESS) {
    //     std::cout << "cublasLtMatmul GG with status " << status << std::endl;
    // }
    // err = cudaEventRecord(stopEvent, 0);
    // err = cudaEventSynchronize(stopEvent);
    // float time;
    // err = cudaEventElapsedTime(&time, startEvent, stopEvent);
    // if (err != cudaSuccess) {
    //     std::cout << " cuda event elpsed time GG" << std::endl;
    // }
    // std::cout << "fp16 matmul " << time << " ms" << std::endl;;
}

void CublasGemmFp16(int m, int k, int n, cublasHandle_t handle) {
    cublasStatus_t status;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    const __half alpha = (__half)1.0f;
    const __half beta = (__half)0.0f;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    // int32_t alpha = 1;
    // int32_t beta = 0;
    int lda = m;
    int ldb = k;
    int ldc = m;

    __half *A, *B, *C;
    cudaMalloc(&A, sizeof(__half) * m * k);
    cudaMalloc(&B, sizeof(__half) * n * k);
    cudaMalloc(&C, sizeof(__half) * m * n);

    cudaStream_t stream = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    const int repeats = 10000;
    for (int loop = 0; loop < repeats; loop++) {
        // Non Tensorcore
        // status = cublasHgemm(handle,
        //     transa, transb,
        //     m, n, k,
        //     &alpha,
        //     A, lda,
        //     B, ldb,
        //     &beta,
        //     C, ldc);
        // if (status != CUBLAS_STATUS_SUCCESS) {
        //     std::cout << "cublasHgemm GG with status " << status << std::endl;
        //     return;
        // }

        // Use Tensorcore
        // cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;//CUBLAS_GEMM_DFALT_TENSOR_OP;
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;

        status = cublasGemmEx(handle,
            transa,
            transb,
            m,
            // n,
            n,
            // m,
            k,
            (void*)&alpha,
            (void*)A,
            CUDA_R_16F,
            m,
            (void*)B,
            CUDA_R_16F,
            k,
            (void*)&beta,
            (void*)C,
            CUDA_R_16F,
            m,
            CUDA_R_32F,
            algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cublasHgecublasGemmExmm GG with status " << status << std::endl;
            return;
        }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float time = diffTime(start, end);
    std::cout << "cublasHgemm spend " << time/repeats << " ms in " << m  << ", " << k << ", " << n << std::endl;

}