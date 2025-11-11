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
    
    // 矩阵乘法：output = input × weights^T
    // input: [batch_size, in_features]
    // weights: [out_features, in_features] 
    // output: [batch_size, out_features]
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size, out_features, in_features,
                &alpha, input, batch_size,
                weights, out_features,
                &beta, output, batch_size);
    
    // add bias
    if (bias != nullptr) {
        cublasSaxpy(handle, batch_size * out_features, 
                    &alpha, bias, 1, output, 1);
    }
    
    cublasDestroy(handle);
}

