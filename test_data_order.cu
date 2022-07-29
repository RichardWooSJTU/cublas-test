#include "cublas.cu.h"
#include "gemm.cu.h"


int main() {
    std::vector<std::vector<int>> params{{4096, 12288}, {4096, 16384}, {4096, 4096}, {16384, 4096}, {256, 768}, {256, 1024}, {1024, 256}, {256, 256}};

    // std::vector<std::vector<int>> params{{256, 256}};
    // std::vector<std::vector<int>> params{{4096, 16384}};
    for (int m : {1, 4, 8, 16, 60, 128}) {
        for (auto p : params) {
            // Init cublasLt
            // int m = 1, k = 4096, n = 12288;
            int k = p[0], n = p[1];
            cublasLtHandle_t ltHandle;
            cublasLtCreate(&ltHandle);
            cublasHandle_t handle;
            cublasCreate(&handle);

            // GemmInt8(m, k, n, ltHandle);
            // GemmInt8(n, k, m, ltHandle);
            // GemmFp16(m, k, n, ltHandle);
            // CublasGemmFp16(m, k, n, handle);
            CublasGemmInt8(m, k, n, handle);
            
        }
    }
    

    


}