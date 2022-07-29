#include "cublas.cu.h"

__global__ void transpose_kernel(const int8_t* src,
    int8_t* dst,
    int row,
    int col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    if (i >= row) return;
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col
    if (j >= col) return;

    dst[j * row + i] = src[i * col + j];
}

void transpose_kernelLauncher(const int8_t* input,
    int8_t* output,
       int row,
       int col,
       cudaStream_t stream) {
       dim3 grid((row + 31) / 32, (col + 31) / 32);
       dim3 block(32, 32);
       transpose_kernel<<<grid, block, 0, stream>>>(input, output, row, col);
   }
   
void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}
   