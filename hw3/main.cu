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

void test_forward_fc(){
    const int batch_size = 1;
    const int in_features = 2;
    const int out_features = 2;

    // 输入：[[1, 2]]
    thrust::host_vector<float> h_input = {1.f, 2.f};

    // 权重：
    // [[1, 2],
    //  [3, 4]]
    thrust::host_vector<float> h_weights = {1.f, 2.f,
                                            3.f, 4.f};

    // 偏置：[[0.1, 0.2]]
    thrust::host_vector<float> h_bias = {0.1f, 0.2f};

    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_weights = h_weights;
    thrust::device_vector<float> d_bias = h_bias;
    thrust::device_vector<float> d_output(batch_size * out_features);

    forward_fc(thrust::raw_pointer_cast(d_input.data()),
               thrust::raw_pointer_cast(d_output.data()),
               thrust::raw_pointer_cast(d_weights.data()),
               thrust::raw_pointer_cast(d_bias.data()),
               batch_size, in_features, out_features);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_output = d_output;

    // 手算期望结果：
    // y0 = 1*1 + 2*2 + 0.1 = 5.1
    // y1 = 1*3 + 2*4 + 0.2 = 11.2
    const float expected0 = 5.1f;
    const float expected1 = 11.2f;

    bool pass = (std::fabs(h_output[0] - expected0) < 1e-4f) &&
                (std::fabs(h_output[1] - expected1) < 1e-4f);

    std::cout << "forward_fc test " << (pass ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Expected: " << expected0 << " " << expected1 << std::endl;
    std::cout << "Actual:   " << h_output[0] << " " << h_output[1] << std::endl;
}

int main(){
    test_forward_fc();
    return 0;
}
