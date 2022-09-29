#include "gemm.cu.h"
#include <fstream>
#include <iostream>

template <typename T>
void PrintMatrix(T* mat_d, int m, int n) {
    std::vector<T> tmp(m*n);
    std::cout << "=========PrintMatrix========" << std::endl;
    cudaMemcpy(tmp.data(), mat_d, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << static_cast<int>(tmp[i*n+j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=========PrintMatrixEnd========" << std::endl;
}

template <typename T>
T* PrepareData(int m, int n, std::string name) {
    T* data_d;
    std::vector<T> data(m * n);
    cudaMalloc(reinterpret_cast<void**>(&data_d), m * n * sizeof(T));
    std::cout << "origin " << name <<  " data: "<< std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            data[i * n + j] = static_cast<T>(j % 127);
            // std::cout << static_cast<int>(data[i * n + j]) << " ";
        }
        // std::cout << std::endl;
    }
    cudaMemcpy(data_d, data.data(), m * n * sizeof(T), cudaMemcpyHostToDevice);
    return data_d;
}

std::vector<int32_t> GemmInt8Imma1(const std::vector<int8_t>& A, const std::vector<int8_t>& B, int m, int n, int k) {
    int8_t* A_dev;
    int8_t* B_dev;
    int32_t* C_dev;

    std::vector<int32_t> C(m * n);

    cudaMalloc(&A_dev, A.size() * sizeof(int8_t));
    cudaMalloc(&B_dev, B.size() * sizeof(int8_t));
    cudaMalloc(&C_dev, C.size() * sizeof(int32_t));

    cudaMemcpy(A_dev, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size(), cudaMemcpyHostToDevice);

    //init data structure

    cublasLtHandle_t handle_;
    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_desc_;
    cublasLtMatrixLayout_t B_desc_;
    cublasLtMatrixLayout_t C_desc_;
    int32_t alpha_ = 1;
    int32_t beta_ = 0;


    cublasLtCreate(&handle_);
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
    int algoId = 21;
    int swizzle = 0;
    int customOption = 0;
    int tile = 15;
    int splitK_val = 0;
    int reductionScheme = 0;
    int stages = 23;


    cublasLtMatmulAlgoInit(handle_,
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
    status = cublasLtMatmul(handle_,
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
                                 nullptr,
                                 0,
                                 0);
    std::cout << "status " << status << std::endl;
    cudaDeviceSynchronize();

    PrintMatrix(C_dev, m, n);
    cudaMemcpy(C.data(), C_dev, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return C;
}

std::vector<int32_t> GemmInt8(int m, int k, int n, 
    cublasLtHandle_t ltHandle
    ) {
    std::cout << n;
    std::vector<int32_t> res(m*n);
     // Init value
     int8_t * A_dev, * B_dev, * A_dev_tmp, * B_dev_tmp;
     A_dev = PrepareData<int8_t>(m, k, "A");
     B_dev = PrepareData<int8_t>(k, n, "B");

     cudaMalloc(reinterpret_cast<void**>(&A_dev_tmp), m * k * sizeof(int8_t));
     cudaMalloc(reinterpret_cast<void**>(&B_dev_tmp), k * n * sizeof(int8_t));
 
     std::vector<int32_t> C_vec(m * n);
     int32_t * C_dev;
     cudaMalloc(reinterpret_cast<void**>(&C_dev), m * n * sizeof(int32_t));

     cublasStatus_t status;
 
     // Transpose A B
     transpose_kernelLauncher(A_dev, A_dev_tmp, m, k, 0);
     A_dev = A_dev_tmp;

    //  transpose_kernelLauncher(B_dev, B_dev_tmp, k, n, 0);
    //  B_dev = B_dev_tmp;
         
 
     // Init origin matrix desc
     cublasLtMatrixLayout_t Adesc = NULL;
     cublasLtMatrixLayout_t Bdesc = NULL;
     cublasLtMatrixLayout_t Cdesc = NULL;
     // Describe A and B in col-major
     cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, m);
     cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, n);
     cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, n, m, n);
    // Init matmul
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);

    // For IMMA
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)); // opTranspose = CUBLAS_OP_T;

    // Init transform matrix desc
    cublasLtMatrixLayout_t ATransdesc = NULL;
    cublasLtMatrixLayout_t BTransdesc = NULL;
    cublasLtMatrixLayout_t CTransdesc = NULL;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
    cublasLtOrder_t order_matrixB;
    bool use_4r4 = true;
    if (use_4r4) {
        order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
    } else {
        order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
    }

    // int ldatransform = 32 * m;
    int ldbtransform = 32 * n;
    // int ldbtransform = 32 * (k + 8 - 1) / 8 * 8; // B should be transposed
    // int ldbtransform;
    int ldatransform;
    if (use_4r4) {
        // ldbtransform = 32 * ((n + 32 - 1) / 32) * 32;
        ldatransform = 32 * ((m + 32 - 1) / 32) * 32;
    } else {
        // ldbtransform = 32 * ((n + 8 - 1) / 8) * 8;
        ldatransform = 32 * ((m + 8 - 1) / 8) * 8;
    }
    // int ldctransform = 32 * m;
    int ldctransform = 32 * n;
    // cublasLtMatrixLayoutCreate(&ATransdesc, CUDA_R_8I, m, k, ldatransform);
    // cublasLtMatrixLayoutSetAttribute(ATransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
    // cublasLtMatrixLayoutCreate(&BTransdesc, CUDA_R_8I, n, k, ldbtransform);
    // cublasLtMatrixLayoutSetAttribute(BTransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
    // cublasLtMatrixLayoutCreate(&CTransdesc, CUDA_R_32I, m, n, ldctransform);
    // cublasLtMatrixLayoutSetAttribute(CTransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

    cublasLtMatrixLayoutCreate(&ATransdesc, CUDA_R_8I, m, k, ldatransform);
    cublasLtMatrixLayoutSetAttribute(ATransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_COL32));
    cublasLtMatrixLayoutCreate(&BTransdesc, CUDA_R_8I, n, k, ldbtransform);
    cublasLtMatrixLayoutSetAttribute(BTransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_matrixB));
    cublasLtMatrixLayoutCreate(&CTransdesc, CUDA_R_32I, n, m, ldctransform);
    cublasLtMatrixLayoutSetAttribute(CTransdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

    // Transform A and B
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    int8_t *Atransform = NULL;
    int8_t *Btransform = NULL;
    int32_t *Ctransform = NULL;
    // std::cout << "Before transform, A:" << std::endl;
    // PrintMatrix(A_dev, k, m);
    // std::cout << "Before transform, B:" << std::endl;
    // PrintMatrix(B_dev, k, n);
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
    // opTranspose = CUBLAS_OP_T;
    // cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)); 
    // std::cout << "After transform, A:" << std::endl;
    // PrintMatrix(Atransform, (k + 32 - 1) / 32 * 32, m);


    status = cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B_dev, Bdesc, &transformBeta, NULL, NULL, Btransform, BTransdesc, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasLtMatrixTransform B GG with status " << status << std::endl;
    }
    // std::cout << "After transform, B:" << std::endl;
    // PrintMatrix(Btransform, (k + 32 - 1) / 32 * 32, 32 * ((n + 32 - 1) / 32));


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
        ltHandle, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
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
    // auto results = FindAlgo(ltHandle, m, n, k, Atransform, Btransform, Ctransform, matmulDesc, ATransdesc, BTransdesc, CTransdesc, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I);
    // std::ofstream outfile;
    // outfile.open("res.csv", std::ios::app);
    // int i = 0;
    // while (results[i].time == 0) i++;
    // outfile << "int8, " << m << ", "  << k << ", " << n << ", " << results[i].time << ", " << results[i].wavesCount << ", " << results[i].workspaceSize << std::endl;

    // outfile.close();
    // std::cout << "finsh select" << std::endl;
    

    status = cublasLtMatmul(ltHandle,
        matmulDesc,
        &alpha,
        Btransform,
        BTransdesc,
        Atransform,
        ATransdesc,
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
    // std::cout << "Before transform, C:" << std::endl;
    // PrintMatrix(Ctransform, (n + 32 - 1) / 32 * 32, m);

    // Transform C
    opTranspose = CUBLAS_OP_N;
    cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose));

    status = cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CTransdesc, &transformBeta, NULL, NULL, C_dev, Cdesc, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasLtMatrixTransform C GG with status " << status << std::endl;
    }
    // std::cout << "After transform, C:" << std::endl;
    // PrintMatrix(C_dev, m, n);
    cudaMemcpy(res.data(), C_dev, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost);


    checkCudaStatus(cudaDeviceSynchronize());
    if (Ctransform) checkCudaStatus(cudaFree(Ctransform));
    if (Btransform) checkCudaStatus(cudaFree(Btransform));
    if (Atransform) checkCudaStatus(cudaFree(Atransform));

    return res;

}


std::vector<int32_t> CublasGemmInt8(int m, int k, int n, cublasHandle_t handle) {
    std::vector<int32_t> res(m * n);
    cublasStatus_t status;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    const int32_t alpha = (int32_t)1;
    const int32_t beta = (int32_t)0;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    // int32_t alpha = 1;
    // int32_t beta = 0;
    //std::vector<int8_t> A_v{1,2,3,4,5,6,7,8}, B_v{1,2,3,4,5,6,7,8};
   

    int8_t *A, *B, *B_tmp;
    int32_t *C;
    A = PrepareData<int8_t>(m, k, "A");
    B = PrepareData<int8_t>(k, n, "B");

    cudaMalloc(&B_tmp, sizeof(int8_t) * n * k);

    transpose_kernelLauncher(B, B_tmp, k, n, 0);
    B = B_tmp;
    
    cudaMalloc(&C, sizeof(int32_t) * m * n);

    cudaDeviceSynchronize();
    cudaStream_t stream = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    const int repeats = 100;
    for (int loop = 0; loop < repeats; loop++) {

        // Use Tensorcore
        // cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        // cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO23;

        status = cublasGemmEx(handle,
            transa,
            transb,
            n,
            m,
            k,
            (void*)&alpha,
            (void*)B,
            CUDA_R_8I,
            k,
            (void*)A,
            CUDA_R_8I,
            k,
            (void*)&beta,
            (void*)C,
            CUDA_R_32I,
            n,
            CUBLAS_COMPUTE_32I,
            algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cublasGemmEx GG with status " << status << std::endl;
            return res;
        }
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float time = diffTime(start, end);
    std::cout << "cublasGemmEx int8 spend " << time/repeats << " ms in " << m  << ", " << k << ", " << n << std::endl;
    //std::vector<int32_t> C_v(m*n, 0);

    cudaMemcpy(res.data(), C, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost);
    // PrintMatrix<int32_t>(C, m , n);
    // for (int i = 0; i < m * n; ++i) std::cout << C_v[i] << std::endl;
    
    
   
    return res;
    //d::ofstream outfile;
    //tfile.open("gemmex.csv", std::ios::app);
    //tfile << m << ", "  << k << ", " << n << ", " <<  time/repeats << std::endl;
    //tfile.close();

}