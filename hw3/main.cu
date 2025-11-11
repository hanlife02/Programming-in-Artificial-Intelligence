#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// GEMM , C(m,n) = A(m,k) * B(k,n)    (C = alpha × (A × B) + beta × C)
void gemm_gpu(const float *A, const float *B, float *C, const int m,const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    const float alf = 1, bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
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
void fully_connected_forward_gpu(float* input, float* weights, float* output, int batch_size, int in_feature, int out_feature) {
    gemm_gpu(weights, input, output, batch_size, in_feature, out_feature);
}