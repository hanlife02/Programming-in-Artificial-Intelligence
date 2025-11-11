#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cmath>
#include <cfloat>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// void forward_fc(float* input, float* output, float* weights, float* bias, 
//                          int batch_size, int in_features, int out_features) { 
//     cublasHandle_t handle; 
//     cublasCreate(&handle); 
    
//     const float alpha = 1.0f, beta = 0.0f; 
    
//     cublasSgemm(handle, 
//                 CUBLAS_OP_T, CUBLAS_OP_N,           
//                 out_features, batch_size, in_features, 
//                 &alpha, 
//                 weights, in_features,                
//                 input, in_features,                  
//                 &beta, 
//                 output, out_features);               
    
//     if (bias != nullptr) { 
//         float* ones;
//         cudaMalloc(&ones, batch_size * sizeof(float));
//         thrust::device_ptr<float> ones_ptr(ones);
//         thrust::fill(ones_ptr, ones_ptr + batch_size, 1.0f);
//         const float alpha_bias = 1.0f, beta_output = 1.0f;
//         cublasSgemm(handle, 
//                     CUBLAS_OP_N, CUBLAS_OP_N,
//                     out_features, batch_size, 1,
//                     &alpha_bias,
//                     bias, out_features,
//                     ones, 1,
//                     &beta_output,
//                     output, out_features);
        
//         cudaFree(ones);
//     } 
//     cublasDestroy(handle); 
// }

// void backward_fc(float* input, float* grad_output, float* weights,
//                  float* grad_input, float* grad_weights, float* grad_bias,
//                  int batch_size, int in_features, int out_features) {
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     const float alpha = 1.0f, beta_zero = 0.0f;

//     if (grad_input != nullptr) {
//         cublasSgemm(handle,
//                     CUBLAS_OP_N, CUBLAS_OP_N,
//                     in_features, batch_size, out_features,
//                     &alpha,
//                     weights, in_features,
//                     grad_output, out_features,
//                     &beta_zero,
//                     grad_input, in_features);
//     }

//     if (grad_weights != nullptr) {
//         cublasSgemm(handle,
//                     CUBLAS_OP_N, CUBLAS_OP_T,
//                     in_features, out_features, batch_size,
//                     &alpha,
//                     input, in_features,
//                     grad_output, out_features,
//                     &beta_zero,
//                     grad_weights, in_features);
//     }

//     if (grad_bias != nullptr) {
//         float* ones;
//         cudaMalloc(&ones, batch_size * sizeof(float));
//         thrust::device_ptr<float> ones_ptr(ones);
//         thrust::fill(ones_ptr, ones_ptr + batch_size, 1.0f);

//         cublasSgemv(handle,
//                     CUBLAS_OP_N,
//                     out_features, batch_size,
//                     &alpha,
//                     grad_output, out_features,
//                     ones, 1,
//                     &beta_zero,
//                     grad_bias, 1);

//         cudaFree(ones);
//     }

//     cublasDestroy(handle);
// }

// void test_forward_fc(){ 
//     int batch_size = 2, in_features = 3, out_features = 4; 
//     float h_input[] = {1, 2, 3,  
//                        4, 5, 6};   
//     float h_weights[] = {1, 0, 1,
//                          0, 1, 1,
//                          1, 1, 0,  
//                          0, 0, 1}; 
//     float h_bias[] = {0.5, 0.5, 0.5, 0.5}; 
//     float h_output[8] = {0}; 

//     float *d_input, *d_weights, *d_bias, *d_output; 
//     cudaMalloc(&d_input, batch_size * in_features * sizeof(float)); 
//     cudaMalloc(&d_weights, out_features * in_features * sizeof(float)); 
//     cudaMalloc(&d_bias, out_features * sizeof(float)); 
//     cudaMalloc(&d_output, batch_size * out_features * sizeof(float)); 
    
//     cudaMemcpy(d_input, h_input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice); 
//     cudaMemcpy(d_weights, h_weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice); 
//     cudaMemcpy(d_bias, h_bias, out_features * sizeof(float), cudaMemcpyHostToDevice); 

//     forward_fc(d_input, d_output, d_weights, d_bias, 
//                         batch_size, in_features, out_features); 
    
//     cudaMemcpy(h_output, d_output, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToHost); 
    
//     float expected_output[] = {4.5, 5.5, 3.5, 3.5, 
//                                10.5, 11.5, 9.5, 6.5};
    
//     int passed = 1;
//     for(int i = 0; i < batch_size * out_features; i++){ 
//         if(std::fabs(h_output[i] - expected_output[i]) > 1e-5){ 
//             std::cout << "Test failed at index " << i << ": expected " << expected_output[i] << ", got " << h_output[i] << std::endl; 
//             passed = 0;
//         }
//     }

//     if (passed) {
//         printf("Test1 passed!\n");
//     }

//     cudaFree(d_input); 
//     cudaFree(d_weights); 
//     cudaFree(d_bias); 
//     cudaFree(d_output); 
// } 

// void test_backward_fc(){
//     int batch_size = 2, in_features = 3, out_features = 4;
//     float h_input[] = {1, 2, 3,
//                        4, 5, 6};
//     float h_weights[] = {1, 0, 1,
//                          0, 1, 1,
//                          1, 1, 0,
//                          0, 0, 1};
//     float h_grad_output[] = {0.1f, 0.2f, 0.3f, 0.4f,
//                              0.5f, 0.6f, 0.7f, 0.8f};
//     float h_grad_input[6] = {0};
//     float h_grad_weights[12] = {0};
//     float h_grad_bias[4] = {0};

//     float *d_input, *d_weights, *d_grad_output;
//     float *d_grad_input, *d_grad_weights, *d_grad_bias;
//     cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
//     cudaMalloc(&d_weights, out_features * in_features * sizeof(float));
//     cudaMalloc(&d_grad_output, batch_size * out_features * sizeof(float));
//     cudaMalloc(&d_grad_input, batch_size * in_features * sizeof(float));
//     cudaMalloc(&d_grad_weights, out_features * in_features * sizeof(float));
//     cudaMalloc(&d_grad_bias, out_features * sizeof(float));

//     cudaMemcpy(d_input, h_input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_weights, h_weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_grad_output, h_grad_output, batch_size * out_features * sizeof(float), cudaMemcpyHostToDevice);

//     backward_fc(d_input, d_grad_output, d_weights,
//                 d_grad_input, d_grad_weights, d_grad_bias,
//                 batch_size, in_features, out_features);

//     cudaMemcpy(h_grad_input, d_grad_input, batch_size * in_features * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_grad_weights, d_grad_weights, out_features * in_features * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_grad_bias, d_grad_bias, out_features * sizeof(float), cudaMemcpyDeviceToHost);

//     float expected_grad_input[] = {0.4f, 0.5f, 0.7f,
//                                    1.2f, 1.3f, 1.9f};
//     float expected_grad_weights[] = {2.1f, 2.7f, 3.3f,
//                                      2.6f, 3.4f, 4.2f,
//                                      3.1f, 4.1f, 5.1f,
//                                      3.6f, 4.8f, 6.0f};
//     float expected_grad_bias[] = {0.6f, 0.8f, 1.0f, 1.2f};

//     int passed = 1;
//     for (int i = 0; i < batch_size * in_features; ++i) {
//         if (std::fabs(h_grad_input[i] - expected_grad_input[i]) > 1e-5f) {
//             std::cout << "Grad input mismatch at index " << i
//                       << ": expected " << expected_grad_input[i]
//                       << ", got " << h_grad_input[i] << std::endl;
//             passed = 0;
//         }
//     }

//     for (int i = 0; i < out_features * in_features; ++i) {
//         if (std::fabs(h_grad_weights[i] - expected_grad_weights[i]) > 1e-5f) {
//             std::cout << "Grad weights mismatch at index " << i
//                       << ": expected " << expected_grad_weights[i]
//                       << ", got " << h_grad_weights[i] << std::endl;
//             passed = 0;
//         }
//     }

//     for (int i = 0; i < out_features; ++i) {
//         if (std::fabs(h_grad_bias[i] - expected_grad_bias[i]) > 1e-5f) {
//             std::cout << "Grad bias mismatch at index " << i
//                       << ": expected " << expected_grad_bias[i]
//                       << ", got " << h_grad_bias[i] << std::endl;
//             passed = 0;
//         }
//     }

//     if (passed) {
//         printf("Test2 passed!\n");
//     }

//     cudaFree(d_input);
//     cudaFree(d_weights);
//     cudaFree(d_grad_output);
//     cudaFree(d_grad_input);
//     cudaFree(d_grad_weights);
//     cudaFree(d_grad_bias);
// }

__global__ void max_pool_forward_kernel(const float* in_data, float* out_data, float* out_mask,
                                        int nthreads, int num, int channels,
                                        int in_h, int in_w, int out_h, int out_w,
                                        int kernel_h, int kernel_w, int stride_h, int stride_w) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / (channels * out_h * out_w);
        int c = (index / (out_h * out_w)) % channels;
        int ph = (index / out_w) % out_h;
        int pw = index % out_w;

        int hstart = ph * stride_h;
        int wstart = pw * stride_w;

        float max_val = -FLT_MAX;
        int max_idx = -1;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = hstart + kh;
                int w = wstart + kw;
                int in_index = ((n * channels + c) * in_h + h) * in_w + w;
                float val = in_data[in_index];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_index;
                }
            }
        }

        out_data[index] = max_val;
        if (out_mask != nullptr) {
            out_mask[index] = static_cast<float>(max_idx);
        }
    }
}

__global__ void max_pool_backward_kernel(const float* grad_out, const float* mask,
                                         float* grad_in, int nthreads) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int in_index = static_cast<int>(mask[index]);
        if (in_index >= 0) {
            atomicAdd(&grad_in[in_index], grad_out[index]);
        }
    }
}

void max_pool_forward_layer(float* input, float* output, float* mask,
                            int num, int channels, int height, int width) {
    const int kernel_h = 2, kernel_w = 2;
    const int stride_h = 2, stride_w = 2;
    int out_h = height / 2;
    int out_w = width / 2;
    int nthreads = num * channels * out_h * out_w;
    int threads = 256;
    int blocks = (nthreads + threads - 1) / threads;
    max_pool_forward_kernel<<<blocks, threads>>>(input, output, mask,
                                                 nthreads, num, channels,
                                                 height, width, out_h, out_w,
                                                 kernel_h, kernel_w, stride_h, stride_w);
}

void max_pool_backward_layer(float* grad_out, float* mask, float* grad_in,
                             int num, int channels, int height, int width) {
    int out_h = height / 2;
    int out_w = width / 2;
    int nthreads = num * channels * out_h * out_w;
    cudaMemset(grad_in, 0, num * channels * height * width * sizeof(float));
    int threads = 256;
    int blocks = (nthreads + threads - 1) / threads;
    max_pool_backward_kernel<<<blocks, threads>>>(grad_out, mask, grad_in, nthreads);
}

void test_max_pool_forward() {
    int num = 1, channels = 1, height = 4, width = 4;
    float h_input[] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f
    };
    float expected_output[] = {6.f, 8.f, 14.f, 16.f};

    float *d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, num * channels * height * width * sizeof(float));
    cudaMalloc(&d_output, num * channels * (height / 2) * (width / 2) * sizeof(float));
    cudaMalloc(&d_mask, num * channels * (height / 2) * (width / 2) * sizeof(float));

    cudaMemcpy(d_input, h_input, num * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    max_pool_forward_layer(d_input, d_output, d_mask, num, channels, height, width);

    float h_output[4] = {0};
    cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(h_output[i] - expected_output[i]) > 1e-5f) {
            std::cout << "MaxPool forward mismatch at index " << i
                      << ": expected " << expected_output[i]
                      << ", got " << h_output[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("MaxPool forward test passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}

void test_max_pool_backward() {
    int num = 1, channels = 1, height = 4, width = 4;
    float h_input[] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f
    };
    float h_grad_out[] = {1.f, 2.f, 3.f, 4.f};
    float expected_grad_in[] = {
        0.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 2.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 3.f, 0.f, 4.f
    };

    size_t input_bytes = num * channels * height * width * sizeof(float);
    size_t output_bytes = num * channels * (height / 2) * (width / 2) * sizeof(float);

    float *d_input, *d_output, *d_mask;
    float *d_grad_out, *d_grad_in;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_mask, output_bytes);
    cudaMalloc(&d_grad_out, output_bytes);
    cudaMalloc(&d_grad_in, input_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    max_pool_forward_layer(d_input, d_output, d_mask, num, channels, height, width);

    cudaMemcpy(d_grad_out, h_grad_out, output_bytes, cudaMemcpyHostToDevice);
    max_pool_backward_layer(d_grad_out, d_mask, d_grad_in, num, channels, height, width);

    float h_grad_in[16] = {0};
    cudaMemcpy(h_grad_in, d_grad_in, input_bytes, cudaMemcpyDeviceToHost);

    int passed = 1;
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(h_grad_in[i] - expected_grad_in[i]) > 1e-5f) {
            std::cout << "MaxPool backward mismatch at index " << i
                      << ": expected " << expected_grad_in[i]
                      << ", got " << h_grad_in[i] << std::endl;
            passed = 0;
        }
    }

    if (passed) {
        printf("MaxPool backward test passed!\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
}

int main(){ 
    // test_forward_fc(); 
    // test_backward_fc();
    test_max_pool_forward();
    test_max_pool_backward();
    return 0; 
} 
