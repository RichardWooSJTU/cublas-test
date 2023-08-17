# include "gemm.h"


void MultiStreamGemmInt8(int stream_num, int repeate_times=100, int warmup_times=10) {

  ct::CUBLASLTContext dev_ctx;

  int m = 30000, k = 8192, n = 3072;


  auto A = std::vector<int8_t>(m * k);
  auto B = std::vector<int8_t>(k * n);
  auto C = std::vector<int32_t>(m * n);

  // multi stream 
  std::vector<cudaStream_t*> streams;

  for (int i = 0; i < stream_num; ++i) {
    cudaStream_t* stream = new cudaStream_t;
    cudaStreamCreate(stream);
    streams.push_back(stream);
  }
  
  ct::MultiGEMM(dev_ctx, A,B,C, m,k,n,false, streams); 



  
  for (int i = 0; i < stream_num; ++i) {
    delete streams[i];
  }


}