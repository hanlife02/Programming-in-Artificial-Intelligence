#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


// GEMM , C(m,n) = A(m,k) * B(k,n)^T    (C = alpha × (A × B^T) + beta × C)
// A:INPUT ,B:WEIGHT ,C:OUTPUT
void gemm_gpu(const int m, const int n, const int k,  float alf, const float *A, const float *B, float bet, float *C) {
    int lda = m, ldb = k, ldc = m;
    float *alpha = &alf;
    float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle; cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, k, 
                alpha, A, lda, 
                B, ldb, 
                beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}

// Fully connected layer forward pass
void forward_fc(float* input, float* output, float* weights, float* bias,
                int batch_size, int in_features, int out_features) {
    // matrix product with gemm
    gemm_gpu(batch_size, out_features, in_features, 1.0, input, weights, 0.0, output);
    // add bias
    gemm_gpu(batch_size, out_features, out_features, 1.0, bias, ones_, 1.0, output);
}