#include <string>
#include <vector>
#include <fstream>

#include "cublas.cu.h"
#include "gemm.cu.h"
#include "gemm.h"

__global__ void transpose(int8_t * src, int8_t* dst, int m, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / n;
    int j = tid % n;
    
    dst[j*m+i] = src[i * n + j];
    
}

template<typename T>
std::vector<T> OpenInput(int m, int n, std::string name) {
    std::vector<T> res(m * n);
    std::ifstream infile;
    infile.open(name+".txt", std::ios::in);
    for (int i = 0; i < m * n; ++i) {
        int tmp;
        infile >> tmp;
        // std::cout << static_cast<int>(tmp) << std::endl;
        res[i] = tmp;
    }
    infile.close();
    return res;
}


int main() {
    //  std::vector<std::vector<int>> params{{4096, 12288}, {4096, 16384}, {4096, 4096}, {16384, 4096}, {256, 768}, {256, 1024}, {1024, 256}, {256, 256}};
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    cublasHandle_t handle;
     cublasCreate(&handle);
    //  std::vector<std::vector<int>> params{{32, 32}};
      std::vector<std::vector<int>> params{{768, 768}};
    //  int m = 4;
    // for (int m : {1}) {
    //      for (auto p : params) {
            // Init cublasLt
            int m = 1, k = 768, n = 768;
            // int k = p[0], n = p[1];
            auto A = OpenInput<int8_t>(m, k, "input");
            auto B = OpenInput<int8_t>(k, n, "weight");
            auto C_1 = GemmInt8Imma1(A, B, m, n, k);

            std::vector<int32_t> C_2(m*n);
            Gemm(A.data(), B.data(), C_2.data(), m, n, k);



            
            // auto C_v1 =  GemmInt8(m, k, n, ltHandle);
            //auto C_v1 =  GemmInt8(n, k, m, ltHandle);
            // GemmFp16(m, k, n, ltHandle);
            // CublasGemmFp16(m, k, n, handle);
            // std::vector<int8_t> A_v(m*k), B_v(k*n);
            // std::vector<int32_t> C_v1(m*n), C_v2(m*n);
            // auto C_v2 = CublasGemmInt8(m, k, n, handle);
            //CublasGemmInt8(n, k, m, handle, B_v.data(), A_v.data(), C_v1.data());
            // Gemm(handle, A_v.data(), B_v.data(), C_v2.data(), m, n, k);
            bool f = true;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (C_1[i*n+j] != C_2[i*n+j]) {
                        std::cout << i << " " << j  << " "<< C_1[i*n+j] << " != " << C_2[i*n+j] << std::endl;
                        f = false;
                    }
                }
            }
            if (f) {
                std::cout << "Congratulations!!!" << std::endl;
            }
            
    //     }
    // }
    
//CublasGemmInt8(2, 4, 2, handle);
    


}