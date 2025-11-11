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
    
    cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,           
                out_features, batch_size, in_features, 
                &alpha, 
                weights, in_features,                
                input, in_features,                  
                &beta, 
                output, out_features);               
    
    if (bias != nullptr) { 
        float* ones;
        cudaMalloc(&ones, batch_size * sizeof(float));
        thrust::device_ptr<float> ones_ptr(ones);
        thrust::fill(ones_ptr, ones_ptr + batch_size, 1.0f);
        const float alpha_bias = 1.0f, beta_output = 1.0f;
        cublasSgemm(handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    out_features, batch_size, 1,
                    &alpha_bias,
                    bias, out_features,
                    ones, 1,
                    &beta_output,
                    output, out_features);
        
        cudaFree(ones);
    } 
    cublasDestroy(handle); 
}

void backward_fc(float* input, float* grad_output, float* weights,
                 float* grad_input, float* grad_weights, float* grad_bias,
                 int batch_size, int in_features, int out_features) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta_zero = 0.0f;

    if (grad_input != nullptr) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    in_features, batch_size, out_features,
                    &alpha,
                    weights, in_features,
                    grad_output, out_features,
                    &beta_zero,
                    grad_input, in_features);
    }

    if (grad_weights != nullptr) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    in_features, out_features, batch_size,
                    &alpha,
                    input, in_features,
                    grad_output, out_features,
                    &beta_zero,
                    grad_weights, in_features);
    }

    if (grad_bias != nullptr) {
        float* ones;
        cudaMalloc(&ones, batch_size * sizeof(float));
        thrust::device_ptr<float> ones_ptr(ones);
        thrust::fill(ones_ptr, ones_ptr + batch_size, 1.0f);

        cublasSgemv(handle,
                    CUBLAS_OP_N,
                    out_features, batch_size,
                    &alpha,
                    grad_output, out_features,
                    ones, 1,
                    &beta_zero,
                    grad_bias, 1);

        cudaFree(ones);
    }

    cublasDestroy(handle);
}

void test_forward_fc(){ 
    int batch_size = 2, in_features = 3, out_features = 4; 
    float h_input[] = {1, 2, 3,  
                       4, 5, 6};   
    float h_weights[] = {1, 0, 1,
                         0, 1, 1,
                         1, 1, 0,  
                         0, 0, 1}; 
    float h_bias[] = {0.5, 0.5, 0.5, 0.5}; 
    float h_output[8] = {0}; 

    float *d_input, *d_weights, *d_bias, *d_output; 
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float)); 
    cudaMalloc(&d_weights, out_features * in_features * sizeof(float)); 
    cudaMalloc(&d_bias, out_features * sizeof(float)); 
    cudaMalloc(&d_output, batch_size * out_features * sizeof(float)); 
    
    cudaMemcpy(d_input, h_input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_weights, h_weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_bias, h_bias, out_features * sizeof(float), cudaMemcpyHostToDevice); 

    forward_fc(d_input, d_output, d_weights, d_bias, 
                        batch_size, in_features, out_features); 
    
    cudaMemcpy(h_output, d_output, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToHost); 
    
    float expected_output[] = {4.5, 5.5, 3.5, 3.5, 
                               10.5, 11.5, 9.5, 6.5};
    
    int passed = 1;
    for(int i = 0; i < batch_size * out_features; i++){ 
        if(std::fabs(h_output[i] - expected_output[i]) > 1e-5){ 
            std::cout << "Test failed at index " << i << ": expected " << expected_output[i] << ", got " << h_output[i] << std::endl; 
            passed = 0;
        }
    }

    if (passed) {
        printf("Test1 passed!\n");
    }

    cudaFree(d_input); 
    cudaFree(d_weights); 
    cudaFree(d_bias); 
    cudaFree(d_output); 
} 

void test_backward_fc(){
    int batch_size = 2, in_features = 3, out_features = 4;
    float h_input[] = {1, 2, 3,
                       4, 5, 6};
    float h_weights[] = {1, 0, 1,
                         0, 1, 1,
                         1, 1, 0,
                         0, 0, 1};
    float h_grad_output[] = {0.1f, 0.2f, 0.3f, 0.4f,
                             0.5f, 0.6f, 0.7f, 0.8f};
    float h_grad_input[6] = {0};
    float h_grad_weights[12] = {0};
    float h_grad_bias[4] = {0};

    float *d_input, *d_weights, *d_grad_output;
    float *d_grad_input, *d_grad_weights, *d_grad_bias;
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_weights, out_features * in_features * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * out_features * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_grad_weights, out_features * in_features * sizeof(float));
    cudaMalloc(&d_grad_bias, out_features * sizeof(float));

    cudaMemcpy(d_input, h_input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, batch_size * out_features * sizeof(float), cudaMemcpyHostToDevice);

    backward_fc(d_input, d_grad_output, d_weights,
                d_grad_input, d_grad_weights, d_grad_bias,
                batch_size, in_features, out_features);

    cudaMemcpy(h_grad_input, d_grad_input, batch_size * in_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_weights, d_grad_weights, out_features * in_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_bias, d_grad_bias, out_features * sizeof(float), cudaMemcpyDeviceToHost);

    float expected_grad_input[] = {0.4f, 0.5f, 0.7f,
                                   1.2f, 1.3f, 1.9f};
    float expected_grad_weights[] = {2.1f, 2.7f, 3.3f,
                                     2.6f, 3.4f, 4.2f,
                                     3.1f, 4.1f, 5.1f,
                                     3.6f, 4.8f, 6.0f};
    float expected_grad_bias[] = {0.6f, 0.8f, 1.0f, 1.2f};

    int passed = 1;
    for (int i = 0; i < batch_size * in_features; ++i) {
        if (std::fabs(h_grad_input[i] - expected_grad_input[i]) > 1e-5f) {
            std::cout << "Grad input mismatch at index " << i
                      << ": expected " << expected_grad_input[i]
                      << ", got " << h_grad_input[i] << std::endl;
            passed = 0;
        }
    }

    for (int i = 0; i < out_features * in_features; ++i) {
        if (std::fabs(h_grad_weights[i] - expected_grad_weights[i]) > 1e-5f) {
            std::cout << "Grad weights mismatch at index " << i
                      << ": expected " << expected_grad_weights[i]
                      << ", got " << h_grad_weights[i] << std::endl;
            passed = 0;
        }
    }

    for (int i = 0; i < out_features; ++i) {
        if (std::fabs(h_grad_bias[i] - expected_grad_bias[i]) > 1e-5f) {
            std::cout << "Grad bias mismatch at index " << i
                      << ": expected " << expected_grad_bias[i]
                      << ", got " << h_grad_bias[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("Test2 passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_weights);
    cudaFree(d_grad_bias);
}

int main(){ 
    test_forward_fc(); 
    test_backward_fc();
    return 0; 
} 
