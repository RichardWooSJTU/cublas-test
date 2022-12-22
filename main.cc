#include "gemm.h"

int main() {
    int k = 12288, n = 4608;
    std::vector<int> mm{1, 2, 4, 8, 16, 32};
    auto mm_tmp = std::vector<int>(mm);
    int seq_len = 128;
    for (auto& m : mm) {
        mm.push_back(m * seq_len);
        m *= seq_len;
    }
    std::vector<int> kk{384, 384, 1536, 384}; //, 12288, 1536, 12288, 6144};
    std::vector<int> nn{1152, 384, 384, 1536} ;//, 4096, 12288, 6144, 12288};
    for (auto m : mm) {
        for (int i = 0; i < kk.size(); ++i) {
                n = nn[i];
                k = kk[i];
                auto A = std::vector<int8_t>(m * k);
                auto B = std::vector<int8_t>(k * n);
                auto C = std::vector<int32_t>(m * n);

                ct::CUBLASLTContext dev_ctx;

                ct::GEMM(dev_ctx, A,B,C, m,k,n,true);
        }
    }
}