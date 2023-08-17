#include "gemm.h"
#include "multistream.h"

void Int8Test() {
    int m = 40, k = 1024, n = 8192;
    auto A = std::vector<int8_t>(m * k);
    auto B = std::vector<int8_t>(k * n);
    auto C = std::vector<int32_t>(m * n);

    ct::CUBLASLTContext dev_ctx;

    ct::GEMM(dev_ctx, A,B,C, m,k,n,false);
}

void Int8TestBench() {
   std::vector<int> mm{1, 2, 4, 8, 16, 32};
    auto mm_tmp = std::vector<int>(mm);
    int seq_len = 800;
    for (auto& m : mm) {
        mm_tmp.push_back(m * seq_len);
        m *= seq_len;
    }
    std::vector<int> kk{8192, 1024, 8192, 2752};
    std::vector<int> nn{3072, 8192, 5504, 8192};
    for (auto m : mm) {
        for (int i = 0; i < kk.size(); ++i) {
                int n = nn[i];
                int k = kk[i];
                auto A = std::vector<int8_t>(m * k);
                auto B = std::vector<int8_t>(k * n);
                auto C = std::vector<int32_t>(m * n);

                ct::CUBLASLTContext dev_ctx;

                ct::GEMM(dev_ctx, A,B,C, m,k,n,true);
        }
    }
}

void BF16TestBench() {
   std::vector<int> mm{1, 2, 4, 8, 16, 32};
    auto mm_tmp = std::vector<int>(mm);
    int seq_len = 800;
    for (auto& m : mm) {
        mm_tmp.push_back(m * seq_len);
    }
    std::vector<int> kk{8192, 1024, 8192, 2752};
    std::vector<int> nn{3072, 8192, 5504, 8192};
    for (auto m : mm_tmp) {
        for (int i = 0; i < kk.size(); ++i) {
                int n = nn[i];
                int k = kk[i];
                auto A = std::vector<__nv_bfloat16>(m * k);
                auto B = std::vector<__nv_bfloat16>(k * n);
                auto C = std::vector<__nv_bfloat16>(m * n);

                ct::CUBLASLTContext dev_ctx;

                ct::GEMM(dev_ctx, A,B,C, m,k,n,false);
        }
    }
}

// void Fp8TestBench() {
//     std::vector<int> mm{1, 2, 4, 8, 16, 32};
//     auto mm_tmp = std::vector<int>(mm);
//     int seq_len = 800;
//     for (auto& m : mm) {
//         mm_tmp.push_back(m * seq_len);
//         m *= seq_len;
//     }
//     std::vector<int> kk{8192, 1024, 8192, 2752};
//     std::vector<int> nn{3072, 8192, 5504, 8192};
//     for (auto m : mm_tmp) {
//         for (int i = 0; i < kk.size(); ++i) {
//                 int n = nn[i];
//                 int k = kk[i];
//                 auto A = std::vector<__nv_fp8_e4m3>(m * k);
//                 auto B = std::vector<__nv_fp8_e4m3>(k * n);
//                 auto C = std::vector<half>(m * n);

//                 ct::CUBLASLTContext dev_ctx;

//                 ct::GEMM(dev_ctx, A,B,C, m,k,n,true);
//         }
//     }
// }

int main() {
    // Fp8TestBench();
    Int8Test();
    // for (int i = 1; i <= 1; ++i) {
    //     MultiStreamGemmInt8(i);
    // }
    // BF16TestBench();
}