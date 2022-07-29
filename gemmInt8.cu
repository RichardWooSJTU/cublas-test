#include "gemm.cu.h"
#include <fstream>
#include <iostream>


void GemmInt8(int m, int k, int n, 
    cublasLtHandle_t ltHandle
    ) {
     // Init value
     std::vector<int8_t> A_vec(m * k);
     std::vector<int8_t> B_vec(k * n);
     int8_t * A_dev, * B_dev, * A_dev_tmp, * B_dev_tmp;
     cudaMalloc(reinterpret_cast<void**>(&A_dev), m * k * sizeof(int8_t));
     cudaMalloc(reinterpret_cast<void**>(&B_dev), k * n * sizeof(int8_t));
     cudaMalloc(reinterpret_cast<void**>(&A_dev_tmp), m * k * sizeof(int8_t));
     cudaMalloc(reinterpret_cast<void**>(&B_dev_tmp), k * n * sizeof(int8_t));
 
     std::vector<int32_t> C_vec(m * n);
     int32_t * C_dev;
     cudaMalloc(reinterpret_cast<void**>(&C_dev), m * n * sizeof(int32_t));
 
    //  std::cout << "origin A: "<< std::endl;
     // for (int i = 0; i < m; ++i) {
     //     for (int j = 0; j < k; ++j) {
     //         A_vec[i * k + j] = static_cast<int8_t>(j);
     //         std::cout << static_cast<int>(A_vec[i * k + j]) << " ";
     //     }
     //     std::cout << std::endl;
     // }
 
 
    //  std::cout << "origin B: "<< std::endl;
     // for (int i = 0; i < k; ++i) {
     //     for (int j = 0; j < n; ++j) {
     //         B_vec[i * n + j] = static_cast<int8_t>(j);
     //         std::cout << static_cast<int>(B_vec[i * n + j]) << " ";
     //     }
     //     std::cout << std::endl;
     // }
 
     cudaMemcpy(A_dev_tmp, A_vec.data(), A_vec.size() * sizeof(A_vec[0]), cudaMemcpyHostToDevice);
     cudaMemcpy(B_dev_tmp, B_vec.data(), B_vec.size() * sizeof(B_vec[0]), cudaMemcpyHostToDevice);
     cublasStatus_t status;
 
     // Transpose A B
     transpose_kernelLauncher(A_dev_tmp, A_dev, m, k, 0);
     transpose_kernelLauncher(B_dev_tmp, B_dev, k, n, 0);
 
     
 
 
     // Init origin matrix desc
     cublasLtMatrixLayout_t Adesc = NULL;
     cublasLtMatrixLayout_t Bdesc = NULL;
     cublasLtMatrixLayout_t Cdesc = NULL;
     cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, m);
     cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, k);
     cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, m);
    // Init matmul
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);


    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)); // opTranspose = CUBLAS_OP_T;

    ////////////
    // Select for non IMMA//
    ////////////
    // FindAlgo(ltHandle, m, n, k, A_dev, B_dev, C_dev, matmulDesc, Adesc, Bdesc, Cdesc);
    // return 0;



    // Init transform matrix desc
    cublasLtMatrixLayout_t ATransdesc = NULL;
    cublasLtMatrixLayout_t BTransdesc = NULL;
    cublasLtMatrixLayout_t CTransdesc = NULL;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
    cublasLtOrder_t order_matrixB;
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
    bool use_4r4 = true;
    if (use_4r4) {
        order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
    } else {
        order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
    }


    int ldatransform = 32 * m;
    // int ldbtransform = 32 * (k + 8 - 1) / 8 * 8; // B should be transposed
    int ldbtransform;
    if (use_4r4) {
        ldbtransform = 32 * ((n + 32 - 1) / 32) * 32;
    } else {
        ldbtransform = 32 * ((n + 8 - 1) / 8) * 8;
    }
    int ldctransform = 32 * m;
    cublasLtMatrixLayoutCreate(&ATransdesc, CUDA_R_8I, m, k, ldatransform);
    cublasLtMatrixLayoutSetAttribute(ATransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    cublasLtMatrixLayoutCreate(&BTransdesc, CUDA_R_8I, n, k, ldbtransform);
    cublasLtMatrixLayoutSetAttribute(BTransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
    cublasLtMatrixLayoutCreate(&CTransdesc, CUDA_R_32I, m, n, ldctransform);
    cublasLtMatrixLayoutSetAttribute(CTransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

    // Transform A and B
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    int8_t *Atransform = NULL;
    int8_t *Btransform = NULL;
    int32_t *Ctransform = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(int8_t) * (k + 32 - 1) / 32 * ldatransform);
    cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(int8_t) * (k + 32 - 1) / 32 * ldbtransform);
    cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(int32_t) * (n + 32 - 1) / 32 * ldctransform);
    cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);
    status = cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A_dev, Adesc, &transformBeta, NULL, NULL, Atransform, ATransdesc, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " cublasLtMatrixTransform A GG with status " << status << std::endl;
    }
    // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
    // CUBLAS_OP_T 加在了哪个位置哪个位置的矩阵就是真的会转置，所以matmul的时候B设为转置位就需要先转置，再在matmul中转置回来...
    opTranspose = CUBLAS_OP_T;
    cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)); 
    status = cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B_dev, Bdesc, &transformBeta, NULL, NULL, Btransform, BTransdesc, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasLtMatrixTransform B GG with status " << status << std::endl;
    }


    int32_t alpha = 1, beta = 0;
    // int alpha = 1, beta =  0;

    // Init algo
    cublasLtMatmulAlgo_t algo;
    int algoId;
    algoId = 7;
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    status = cublasLtMatmulAlgoInit(
        ltHandle, CUBLAS_COMPUTE_32I, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasLtMatmulAlgoInit GG with status " << status << std::endl;
    }
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
    int stages;
    stages = 15;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));

    ////////////
    // Select //
    ////////////
    auto results = FindAlgo(ltHandle, m, n, k, Atransform, Btransform, Ctransform, matmulDesc, ATransdesc, BTransdesc, CTransdesc, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I);
    // std::ofstream outfile;
    // outfile.open("res.csv", std::ios::app);
    // int i = 0;
    // while (results[i].time == 0) i++;
    // outfile << "int8, " << m << ", "  << k << ", " << n << ", " << results[i].time << ", " << results[i].wavesCount << ", " << results[i].workspaceSize << std::endl;

    // outfile.close();
    std::cout << "finsh select" << std::endl;
    

    status = cublasLtMatmul(ltHandle,
        matmulDesc,
        &alpha,
        Atransform,
        ATransdesc,
        Btransform,
        BTransdesc,
        &beta,
        Ctransform,
        CTransdesc,
        Ctransform,
        CTransdesc,
        &algo,
        NULL,
        0,
        0);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasLtMatmul GG with status " << status << std::endl;
    }

    // Transform C
    opTranspose = CUBLAS_OP_N;
    cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose));



    status = cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CTransdesc, &transformBeta, NULL, NULL, C_dev, Cdesc, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasLtMatrixTransform C GG with status " << status << std::endl;
    }
    checkCudaStatus(cudaDeviceSynchronize());
    if (Ctransform) checkCudaStatus(cudaFree(Ctransform));
    if (Btransform) checkCudaStatus(cudaFree(Btransform));
    if (Atransform) checkCudaStatus(cudaFree(Atransform));
    //  Watch result
    cudaMemcpy(C_vec.data(), C_dev, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost);
    // std::cout << "result: "<< std::endl;
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << static_cast<int>(C_vec[i * n + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // cublasHandle_t cublasH = NULL;
    // CUBLAS_CHECK(cublasCreate(&cublasH));
    // std::cout << cublasH <<std::endl;


    // for (int i = 0; i < m * n; ++i) {
    //     std::cout << static_cast<int>(C_vec[i]) << " ";
    // }


    // A_vec.clear();
    // A_vec.resize((k + 32 - 1) / 32 * ldatransform);
    // cudaMemcpy(A_vec.data(), Atransform, sizeof(int8_t) * (k + 32 - 1) / 32 * ldatransform, cudaMemcpyDeviceToHost);

    // B_vec.clear();
    // B_vec.resize((k + 32 - 1) / 32 * ldbtransform);
    // cudaMemcpy(B_vec.data(), Btransform, sizeof(int8_t) * (k + 32 - 1) / 32 * ldbtransform, cudaMemcpyDeviceToHost);


    // std::cout << "after transform A: "<< std::endl;
    // // for (int i = 0; i < m; ++i) {
    // //     for (int j = 0; j < k; ++j) {
    // //         std::cout << static_cast<int>(A_host[i * k + j]) << " ";
    // //     }
    // //     std::cout << std::endl;
    // // }
    // for (int i = 0; i < (k + 32 - 1) / 32 * ldatransform; ++i) std::cout << static_cast<int>(A_vec[i]) << " ";

    // std::cout << std::endl;

    // std::cout << "after transform B: "<< std::endl;
    // for (int i = 0; i < (k + 32 - 1) / 32 * ldbtransform; ++i) std::cout << static_cast<int>(B_vec[i]) << " ";

}


void CublasGemmInt8(int m, int k, int n, cublasHandle_t handle) {
    cublasStatus_t status;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    const int32_t alpha = (int32_t)1;
    const int32_t beta = (int32_t)0;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    // int32_t alpha = 1;
    // int32_t beta = 0;
    int lda = 32 * m;
    int ldb =  32 * ((n + 32 - 1) / 32) * 32;
    int ldc = 32 * m;



    int32_t *A, *B, *C;
    cudaMalloc(&A, sizeof(int32_t) * m * k);
    cudaMalloc(&B, sizeof(int32_t) * n * k);
    cudaMalloc(&C, sizeof(int32_t) * m * n);

    cudaStream_t stream = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    const int repeats = 10000;
    for (int loop = 0; loop < repeats; loop++) {

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
            CUDA_R_8I,
            lda,
            (void*)B,
            CUDA_R_8I,
            ldb,
            (void*)&beta,
            (void*)C,
            CUDA_R_32I,
            ldc,
            CUBLAS_COMPUTE_32I,
            algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cublasGemmEx GG with status " << status << std::endl;
            return;
        }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float time = diffTime(start, end);
    std::cout << "cublasHgemm spend " << time/repeats << " ms in " << m  << ", " << k << ", " << n << std::endl;

}