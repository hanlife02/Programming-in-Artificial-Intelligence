#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <cmath>

namespace {
thrust::device_vector<float> ones_cache;

const float* get_device_ones(int length) {
    if (length <= 0) {
        return nullptr;
    }
    if (ones_cache.size() != static_cast<size_t>(length)) {
        ones_cache.assign(length, 1.0f);
    }
    return thrust::raw_pointer_cast(ones_cache.data());
}
}

// GEMM , C(m,n) = A(m,k) * B(k,n)^T  
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

    // add bias via ones vector broadcast
    const float* ones = get_device_ones(batch_size);
    gemm_gpu(batch_size, out_features, 1, 1.0, ones, bias, 1.0, output);
}

int main() {
    const int batch_size = 2;
    const int in_features = 3;
    const int out_features = 2;

    // Host data
    thrust::host_vector<float> h_input = {
        1.f, 2.f, 3.f,   // sample 0
        4.f, 5.f, 6.f    // sample 1
    };
    thrust::host_vector<float> h_weights = {
        1.f, 0.f, -1.f,  // neuron 0
        0.5f, 0.5f, 0.5f // neuron 1
    };
    thrust::host_vector<float> h_bias = {0.5f, -1.f};

    // Device buffers
    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_weights = h_weights;
    thrust::device_vector<float> d_bias = h_bias;
    thrust::device_vector<float> d_output(batch_size * out_features, 0.f);

    forward_fc(thrust::raw_pointer_cast(d_input.data()),
               thrust::raw_pointer_cast(d_output.data()),
               thrust::raw_pointer_cast(d_weights.data()),
               thrust::raw_pointer_cast(d_bias.data()),
               batch_size, in_features, out_features);

    thrust::host_vector<float> h_output = d_output;

    // Compute expected result on host
    bool ok = true;
    auto idx = [out_features](int b, int o) { return b * out_features + o; };
    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < out_features; ++o) {
            float expected = h_bias[o];
            for (int k = 0; k < in_features; ++k) {
                expected += h_input[b * in_features + k] *
                            h_weights[o * in_features + k];
            }
            float got = h_output[idx(b, o)];
            if (std::fabs(expected - got) > 1e-4f) {
                std::cerr << "Mismatch at batch " << b << ", neuron " << o
                          << ": expected " << expected << " got " << got << "\n";
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "forward_fc test passed.\n";
    } else {
        std::cout << "forward_fc test FAILED.\n";
    }
    return ok ? 0 : 1;
}
