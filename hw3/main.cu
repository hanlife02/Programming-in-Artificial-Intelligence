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

int main(){ 
    test_forward_fc(); 
    return 0; 
}