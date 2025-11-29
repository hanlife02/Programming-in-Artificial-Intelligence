#include "ActivationFunctions.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace ActivationFunctions {

__global__ void relu_forward_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<int>(size)) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<int>(size)) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

__global__ void sigmoid_forward_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<int>(size)) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float* grad_output, const float* output, float* grad_input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<int>(size)) {
        grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
    }
}

Tensor ReLU::forward(const Tensor& input) {
    Tensor output(input.shape(), input.device());

    if (input.device() == Device::CPU) {
        forward(input.data(), output.data(), input.size());
    } else {
        int block_size = 256;
        int grid_size = static_cast<int>((input.size() + block_size - 1) / block_size);
        relu_forward_kernel<<<grid_size, block_size>>>(input.data(), output.data(), input.size());
        cudaDeviceSynchronize();
    }

    return output;
}

void ReLU::forward(float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::max(input[i], 0.0f);
    }
}

Tensor ReLU::backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.shape() != input.shape()) {
        throw std::invalid_argument("ReLU backward: gradient and input shapes must match");
    }

    Tensor grad_input(input.shape(), input.device());

    if (input.device() == Device::CPU) {
        backward(grad_output.data(), input.data(), grad_input.data(), input.size());
    } else {
        int block_size = 256;
        int grid_size = static_cast<int>((input.size() + block_size - 1) / block_size);
        relu_backward_kernel<<<grid_size, block_size>>>(grad_output.data(), input.data(), grad_input.data(), input.size());
        cudaDeviceSynchronize();
    }

    return grad_input;
}

void ReLU::backward(const float* grad_output, const float* input, float* grad_input, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

Tensor Sigmoid::forward(const Tensor& input) {
    Tensor output(input.shape(), input.device());

    if (input.device() == Device::CPU) {
        forward(input.data(), output.data(), input.size());
    } else {
        int block_size = 256;
        int grid_size = static_cast<int>((input.size() + block_size - 1) / block_size);
        sigmoid_forward_kernel<<<grid_size, block_size>>>(input.data(), output.data(), input.size());
        cudaDeviceSynchronize();
    }

    return output;
}

void Sigmoid::forward(float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

Tensor Sigmoid::backward(const Tensor& grad_output, const Tensor& output) {
    if (grad_output.shape() != output.shape()) {
        throw std::invalid_argument("Sigmoid backward: gradient and output shapes must match");
    }

    Tensor grad_input(output.shape(), output.device());

    if (output.device() == Device::CPU) {
        backward(grad_output.data(), output.data(), grad_input.data(), output.size());
    } else {
        int block_size = 256;
        int grid_size = static_cast<int>((output.size() + block_size - 1) / block_size);
        sigmoid_backward_kernel<<<grid_size, block_size>>>(grad_output.data(), output.data(), grad_input.data(), output.size());
        cudaDeviceSynchronize();
    }

    return grad_input;
}

void Sigmoid::backward(const float* grad_output, const float* output, float* grad_input, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        grad_input[i] = grad_output[i] * output[i] * (1.0f - output[i]);
    }
}

}  // namespace ActivationFunctions
