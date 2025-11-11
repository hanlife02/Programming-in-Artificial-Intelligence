#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cmath>

void forward_fc(float* input, float* output, float* weights, float* bias,
                int batch_size, int in_features, int out_features) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    int m = batch_size;
    int n = out_features;
    int k = in_features;
    // 矩阵乘法：output = input × weights^T
    // input: [batch_size, in_features]
    // weights: [out_features, in_features] 
    // output: [batch_size, out_features]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, k,
                &alpha, input, m,
                weights, n,
                &beta, output, m);
    
    // add bias
    if (bias != nullptr) {
        cublasSaxpy(handle, batch_size * out_features, 
                    &alpha, bias, 1, output, 1);
    }
    
    cublasDestroy(handle);
}

void test_forward_fc(){
    int m = 2, k = 3, n = 4;
    float h_input[] = {1, 2, 3,
                       4, 5, 6};
    float h_weights[] = {1, 0, 1,
                         0, 1, 1,
                         1, 1, 0,
                         0, 0, 1};
    float h_bias[] = {0.5, 0.5, 0.5, 0.5};
    float h_output[8] = {0};

    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, m * k * sizeof(float));
    cudaMalloc(&d_weights, n * k * sizeof(float));
    cudaMalloc(&d_bias, n * sizeof(float));
    cudaMalloc(&d_output, m * n * sizeof(float));
    cudaMemcpy(d_input, h_input, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, n * sizeof(float), cudaMemcpyHostToDevice);

    forward_fc(d_input, d_output, d_weights, d_bias, m, k, n);
    cudaMemcpy(h_output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Output:" << std::endl;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            std::cout << h_output[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Expected output:
    // 2.5  3.2  3.9  4.6
    // 5.8  7.0  8.2  9.4  

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}

int main(){
    test_forward_fc();
    return 0;
}
