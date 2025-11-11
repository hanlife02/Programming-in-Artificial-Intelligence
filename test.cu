#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_block_id() {
    printf("Block ID: %d started.\n", blockIdx.x);
}

int main() {
    // 启动16个block，每个block有1个thread
    print_block_id<<<16, 1>>>();
    
    // 等待GPU上所有任务完成
    cudaDeviceSynchronize();

    printf("Done.\n");
    return 0;
}