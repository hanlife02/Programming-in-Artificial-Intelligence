#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cmath>

// C(m,n) = A(m,k) * B(k,n)
void gemm_gpu(const float *A, const float *B, float *C, const int m,
            const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    const float alf = 1, bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle; cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
                A, lda, B, ldb, beta, C, ldc);  
    // Destroy the handle
    cublasDestroy(handle);
}

// Fully connected layer forward pass (bias)
void forward_fc(float* input, float* output, float* weights, float* bias,
                int batch_size, int in_features, int out_features) {
    // matrix product with gemm
    gemm_gpu(CublasNoTrans, CublasTrans, batch_size, out_features, in_features,
            1.0, input, weight, 0.0, output);
    // add bias
    gemm_gpu(CublasNoTrans, CublasNoTrans, batch_size, out_features, 1,
            1.0, ones_, bias, 1.0, output);
}

int main() {
    // Example usage of gemm_gpu and forward_fc
    const int m = 2, k = 3, n = 4;
    thrust::host_vector<float> h_A(m * k, 1.0f); // Example matrix A
    thrust::host_vector<float> h_B(k * n, 2.0f); // Example matrix B
    thrust::host_vector<float> h_C(m * n);       // Result matrix C

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, h_A.size() * sizeof(float));
    cudaMalloc((void**)&d_B, h_B.size() * sizeof(float));
    cudaMalloc((void**)&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    gemm_gpu(d_A, d_B, d_C, m, k, n);

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << h_C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}