#include "gemm.h"

int main() {
    int k = 12288, n = 4608;
    std::vector<int> mm{16, 16*128};
    std::vector<int> kk{4096};
    std::vector<int> nn{1024};
    for (auto m : mm) {
        for (auto k : kk) {
            for (auto n : nn) {
                auto A = std::vector<int8_t>(m * k);
                auto B = std::vector<int8_t>(k * n);
                auto C = std::vector<int32_t>(m * n);

                ct::CUBLASLTContext dev_ctx;

                ct::GEMM(dev_ctx, A,B,C, m,k,n,true);
            }
        }
    }
}