#pragma once

#include <iostream>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <vector>
#include "utils.h"
#include "find_algo.cu.h"

namespace ct {
class DevContext {
};

class CPUContext : public DevContext {

};

class CUBLASLTContext : public DevContext {
public:
    CUBLASLTContext() {
        cublasLtCreate(&handle_);
    }

    cublasLtHandle_t handle_;
private:
};


template <typename InT, typename OutT, typename DevContext>
void GEMM(DevContext dev_ctx, 
          const std::vector<InT>& A, 
          const std::vector<InT>& B, 
          std::vector<OutT>& C, 
          int m, 
          int k, 
          int n,
          bool is_test) {
    std::vector<InT> Bt(k * n);
    Transpose(B.data(), Bt.data(), k, n);;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i*n + j] = Elementwise<InT, OutT>(&A[i * k], &Bt[j * k], k);
        }
    }
}

template<>
void GEMM<int8_t, int32_t, CPUContext>(CPUContext dev_ctx,
                              const std::vector<int8_t>& A, 
                              const std::vector<int8_t>& B, 
                              std::vector<int32_t>& C, 
                            int m, 
                            int k, 
                            int n,
                              bool is_test) {

}

template<>
void GEMM<int8_t, int32_t, CUBLASLTContext>(CUBLASLTContext dev_ctx,
                              const std::vector<int8_t>& A, 
                              const std::vector<int8_t>& B, 
                              std::vector<int32_t>& C, 
                            int m, 
                            int k, 
                            int n,
                              bool is_test) {
    int8_t* A_dev;
    int8_t* B_dev;
    int32_t* C_dev;
    char* workspace;

    cudaMalloc((void**)&A_dev, A.size() * sizeof(int8_t));
    cudaMalloc((void**)&B_dev, B.size() * sizeof(int8_t));
    cudaMalloc((void**)&C_dev, m * n * sizeof(int32_t));


    cudaMemcpy(A_dev, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size(), cudaMemcpyHostToDevice);

    //init data structure

    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_desc_;
    cublasLtMatrixLayout_t B_desc_;
    cublasLtMatrixLayout_t C_desc_;
    int32_t alpha_ = 1;
    int32_t beta_ = 0;


    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
    cublasLtMatmulDescCreate(
        &matmul_desc_, cudaComputeType, CUDA_R_32I);
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &op_transpose,
                                                 sizeof(op_transpose));
    cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
    cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m, k);
    cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m, n);

    cublasLtMatmulAlgo_t algo;
    int algoId;
    int swizzle;
    int customOption;
    int tile ;
    int splitK_val;
    int reductionScheme;
    int stages;
    size_t work_space_size;
    float time_ref;

    if (is_test) {
        std::vector<algoSelect_t> algos;
        ////////////
        // Select //
        ////////////
        auto results = FindAlgo(dev_ctx.handle_, m, n, k, 
            B_dev, A_dev, C_dev, matmul_desc_, B_desc_, A_desc_, C_desc_, 
            CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, algos);
        int i = 0;
        while (algos[i].time == 0) i++;
        algoId = algos[i].algoId;
        swizzle = algos[i].swizzle;
        customOption = algos[i].customOption;
        tile = algos[i].tile;
        splitK_val = algos[i].splitK_val;
        reductionScheme = algos[i].reductionScheme;
        stages = algos[i].stages;
        work_space_size = algos[i].workspaceSize;
    } else {
            algoId = 21;
            swizzle = 0;
            customOption = 0;
            tile = 15;
            splitK_val = 0;
            reductionScheme = 0;
            stages = 23;
            if (m >= 128) {
                tile = 20;
                stages = 17;
            }
            work_space_size = 0;
    }
    std::cout << "=======Res========" << std::endl;
    std::cout << "algoId: " << algoId <<  std::endl;
    std::cout << "swizzle: " << swizzle <<  std::endl;
    std::cout << "customOption: " << customOption <<  std::endl;
    std::cout << "tile: " << tile <<  std::endl;
    std::cout << "splitK_val: " << splitK_val <<  std::endl;
    std::cout << "reductionScheme: " << reductionScheme <<  std::endl;
    std::cout << "stages: " << stages <<  std::endl;
    std::cout << "work_space_size: " << work_space_size <<  std::endl;


    cudaMalloc((void**)&workspace, work_space_size);


    cublasLtMatmulAlgoInit(dev_ctx.handle_,
                                cudaComputeType,
                                CUDA_R_32I,
                                CUDA_R_8I,
                                CUDA_R_8I,
                                CUDA_R_32I,
                                CUDA_R_32I,
                                algoId,
                                &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(customOption),
        sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                              CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                              &(splitK_val),
                                              sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(reductionScheme),
        sizeof(int));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));

    cublasStatus_t status;
    // PrintMatrix(A_dev, m, k);
    // PrintMatrix(B_dev, k, n);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    const int repeats = 2;
    for (int loop = 0; loop < repeats; loop++) {
        status = cublasLtMatmul(dev_ctx.handle_,
                                    matmul_desc_,
                                    &alpha_,
                                    B_dev,
                                    B_desc_,
                                    A_dev,
                                    A_desc_,
                                    &beta_,
                                    C_dev,
                                    C_desc_,
                                    C_dev,
                                    C_desc_,
                                    &algo,
                                    //  nullptr,
                                    (void*)workspace,
                                    // 0,
                                    work_space_size,
                                    0);
    }                            
    std::cout << "status " << status << std::endl;
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    float time = diffTime(start, end);
    std::cout << "GEMM with cublaslt imma1 int8 spend " << time/repeats << " ms in " << m  << ", " << k << ", " << n << std::endl;

    // PrintMatrix(C_dev, m, n);
    cudaMemcpy(C.data(), C_dev, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);                        
}


template<>
void GEMM<__nv_bfloat16, __nv_bfloat16, CUBLASLTContext>(CUBLASLTContext dev_ctx,
                              const std::vector<__nv_bfloat16>& A, 
                              const std::vector<__nv_bfloat16>& B, 
                              std::vector<__nv_bfloat16>& C, 
                            int m, 
                            int k, 
                            int n,
                              bool is_test) {
    __nv_bfloat16* A_dev;
    __nv_bfloat16* B_dev;
    __nv_bfloat16* C_dev;
    char* workspace;

    cudaMalloc((void**)&A_dev, A.size() * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&B_dev, B.size() * sizeof(__nv_bfloat16));
    cudaMalloc((void**)&C_dev, m * n * sizeof(__nv_bfloat16));


    cudaMemcpy(A_dev, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size(), cudaMemcpyHostToDevice);

    //init data structure

    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_desc_;
    cublasLtMatrixLayout_t B_desc_;
    cublasLtMatrixLayout_t C_desc_;
    float alpha_ = 1.0;
    float beta_ = 0.0;


    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32F;
    cublasLtMatmulDescCreate(
        &matmul_desc_, cudaComputeType, CUDA_R_32F);
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    // cublasLtMatmulDescSetAttribute(matmul_desc_,
    //                                              CUBLASLT_MATMUL_DESC_TRANSA,
    //                                              &op_transpose,
    //                                              sizeof(op_transpose));
    cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_16BF, n, k, n);
    cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_16BF, n, m, n);

    cublasLtMatmulAlgo_t algo;
    int algoId;
    int swizzle;
    int customOption;
    int tile ;
    int splitK_val;
    int reductionScheme;
    int stages;
    size_t work_space_size;
    float time_ref;

    if (is_test) {
        std::vector<algoSelect_t> algos;
        ////////////
        // Select //
        ////////////
        auto results = FindAlgo<__nv_bfloat16, __nv_bfloat16, float>(dev_ctx.handle_, n, m, k, 
            B_dev, A_dev, C_dev, matmul_desc_, B_desc_, A_desc_, C_desc_, 
            CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, algos);
        int i = 0;
        while (algos[i].time == 0) i++;
        algoId = algos[i].algoId;
        swizzle = algos[i].swizzle;
        customOption = algos[i].customOption;
        tile = algos[i].tile;
        splitK_val = algos[i].splitK_val;
        reductionScheme = algos[i].reductionScheme;
        stages = algos[i].stages;
        work_space_size = algos[i].workspaceSize;
    } else {
        // int m_tmp, k_tmp, n_tmp;
        // FILE *fp;
        // fp=fopen("select.csv", "r");
        // if (!fp) {
        //     algoId = 21;
        //     swizzle = 0;
        //     customOption = 0;
        //     tile = 18;
        //     splitK_val = 0;
        //     reductionScheme = 0;
        //     stages = 21;
        //     work_space_size = 0;
        // } else {
        //     while(1) {
        //         fscanf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f",
        //             &m_tmp,&k_tmp, &n_tmp, &algoId, &swizzle, &customOption,  &tile, &splitK_val, 
        //             &reductionScheme,&stages, &work_space_size, &time_ref);
        //         if (feof(fp))break;
        //         if (m_tmp == m && k_tmp == k && n_tmp == n) break;
        //     }
        //     if (m_tmp != m || k_tmp != k || n_tmp != n) {
        //         std::cout << "Please use test mode to select\n, Now we use default params" << std::endl;
        //         algoId = 21;
        //         swizzle = 0;
        //         customOption = 0;
        //         tile = 15;
        //         splitK_val = 0;
        //         reductionScheme = 0;
        //         stages = 23;
        //         work_space_size = 0;
        //     }
        // }
        algoId = 6;
        swizzle = 1;
        customOption = 0;
        if (m <= 128) {
            tile = 15;
            stages = 18;
        } else {
            tile = 20;
            stages = 11;
        }
        splitK_val = 0;
        reductionScheme = 0;
        work_space_size = 0;
        
        
    }
    std::cout << "=======Res========" << std::endl;
    std::cout << "algoId: " << algoId <<  std::endl;
    std::cout << "swizzle: " << swizzle <<  std::endl;
    std::cout << "customOption: " << customOption <<  std::endl;
    std::cout << "tile: " << tile <<  std::endl;
    std::cout << "splitK_val: " << splitK_val <<  std::endl;
    std::cout << "reductionScheme: " << reductionScheme <<  std::endl;
    std::cout << "stages: " << stages <<  std::endl;
    std::cout << "work_space_size: " << work_space_size <<  std::endl;


    cudaMalloc((void**)&workspace, work_space_size);


    cublasLtMatmulAlgoInit(dev_ctx.handle_,
                                cudaComputeType,
                                CUDA_R_32F,
                                CUDA_R_16BF,
                                CUDA_R_16BF,
                                CUDA_R_16BF,
                                CUDA_R_16BF,
                                algoId,
                                &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(customOption),
        sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                              CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                              &(splitK_val),
                                              sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(reductionScheme),
        sizeof(int));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));

    cublasStatus_t status;
    // PrintMatrix(A_dev, m, k);
    // PrintMatrix(B_dev, k, n);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    const int repeats = 100;
    for (int loop = 0; loop < repeats; loop++) {
        status = cublasLtMatmul(dev_ctx.handle_,
                                    matmul_desc_,
                                    &alpha_,
                                    B_dev,
                                    B_desc_,
                                    A_dev,
                                    A_desc_,
                                    &beta_,
                                    C_dev,
                                    C_desc_,
                                    C_dev,
                                    C_desc_,
                                    &algo,
                                    //  nullptr,
                                    (void*)workspace,
                                    // 0,
                                    work_space_size,
                                    0);
    }                            
    std::cout << "status " << status << std::endl;
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    float time = diffTime(start, end);
    std::cout << "GEMM with cublaslt imma1 bf16 spend " << time/repeats << " ms in " << m  << ", " << k << ", " << n << std::endl;

    // PrintMatrix(C_dev, m, n);
    cudaMemcpy(C.data(), C_dev, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);                        
}

void MultiGEMM(CUBLASLTContext dev_ctx,
                const std::vector<int8_t>& A, 
                const std::vector<int8_t>& B, 
                std::vector<int32_t>& C, 
                int m, 
                int k, 
                int n,
                bool is_test,
                std::vector<cudaStream_t*>& streams,
                int repeate_times = 100,
                int warmup_times=10) {
    int8_t* A_dev;
    int8_t* B_dev;
    int32_t* C_dev;
    char* workspace;

    cudaMalloc((void**)&A_dev, A.size() * sizeof(int8_t));
    cudaMalloc((void**)&B_dev, B.size() * sizeof(int8_t));
    cudaMalloc((void**)&C_dev, m * n * sizeof(int32_t));


    cudaMemcpy(A_dev, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size(), cudaMemcpyHostToDevice);

    //init data structure

    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_desc_;
    cublasLtMatrixLayout_t B_desc_;
    cublasLtMatrixLayout_t C_desc_;
    int32_t alpha_ = 1;
    int32_t beta_ = 0;


    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
    cublasLtMatmulDescCreate(
        &matmul_desc_, cudaComputeType, CUDA_R_32I);
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &op_transpose,
                                                 sizeof(op_transpose));
    cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);

    cublasLtMatmulAlgo_t algo;
    int algoId = 21;
    int swizzle = 0;
    int customOption = 0;
    int tile = 15;
    int splitK_val = 0;
    int reductionScheme = 0;
    int stages = 23;
    if (m >= 128) {
      tile = 20;
      stages = 17;
    }
    int workspace_size = 30000;

    cudaMalloc((void**)&workspace, workspace_size);


    cublasLtMatmulAlgoInit(dev_ctx.handle_,
                                cudaComputeType,
                                CUDA_R_32I,
                                CUDA_R_8I,
                                CUDA_R_8I,
                                CUDA_R_32I,
                                CUDA_R_32I,
                                algoId,
                                &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(customOption),
        sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                              CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                              &(splitK_val),
                                              sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(reductionScheme),
        sizeof(int));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));

    int m_per_stream = m / streams.size();
    int stream_num = streams.size();


    cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m_per_stream, k);
    cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m_per_stream, n);

    cublasStatus_t status;




    struct timeval start, end;
    for (int i = 0; i < repeate_times; ++i) {
        if (i == warmup_times) {
            gettimeofday(&start, NULL);
        }

        for (int stream_id = 0; stream_id < streams.size(); ++stream_id) {

            auto A_dev_tmp = A_dev + m_per_stream * stream_id * k;
            auto C_dev_tmp = C_dev + m_per_stream * stream_id * n;

            status = cublasLtMatmul(dev_ctx.handle_,
                                        matmul_desc_,
                                        &alpha_,
                                        B_dev,
                                        B_desc_,
                                        A_dev_tmp,
                                        A_desc_,
                                        &beta_,
                                        C_dev_tmp,
                                        C_desc_,
                                        C_dev_tmp,
                                        C_desc_,
                                        &algo,
                                        //  nullptr,
                                        (void*)workspace,
                                        // 0,
                                        workspace_size,
                                        0);
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "status = " << status << std::endl;
                exit(0);
            }                
        } 
        cudaDeviceSynchronize();
    }

    gettimeofday(&end, NULL);
    float time = diffTime(start, end);
    std::cout << "GEMM with cublaslt imma1 int8 spend " << time/(repeate_times-warmup_times) << " ms in " << m  << ", " << k << ", " << n << " with " << stream_num << "streams." << std::endl;
    
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
}



} // namespace ct



