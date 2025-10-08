#include "Tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <random>

// Check CUDA API call return values
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// CUDA GPU memory deleter
struct CudaDeleter
{
    void operator()(float *ptr) const
    {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
};

// Constructor
Tensor::Tensor(const std::vector<size_t> &shape, Device device)
    : shape_(shape), device_(device)
{
    // Calculate total number of elements
    num_elements_ = 1;
    for (size_t dim : shape_)
    {
        num_elements_ *= dim;
    }

    if (num_elements_ == 0)
        return;

    // Allocate memory based on device type
    if (device_ == Device::kCPU)
    {
        data_ptr_ = std::shared_ptr<float>(new float[num_elements_], std::default_delete<float[]>());
    }
    else
    { // Device::kGPU
        float *gpu_ptr;
        CUDA_CHECK(cudaMalloc(&gpu_ptr, num_elements_ * sizeof(float)));
        data_ptr_ = std::shared_ptr<float>(gpu_ptr, CudaDeleter());
    }
}

// Copy constructor
Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), device_(other.device_), num_elements_(other.num_elements_)
{
    if (num_elements_ == 0)
        return;

    // Allocate new memory and copy data based on device type
    if (device_ == Device::kCPU)
    {
        data_ptr_ = std::shared_ptr<float>(new float[num_elements_], std::default_delete<float[]>());
        std::memcpy(data(), other.data(), num_elements_ * sizeof(float));
    }
    else
    { // Device::kGPU
        float *gpu_ptr;
        CUDA_CHECK(cudaMalloc(&gpu_ptr, num_elements_ * sizeof(float)));
        data_ptr_ = std::shared_ptr<float>(gpu_ptr, CudaDeleter());
        CUDA_CHECK(cudaMemcpy(data(), other.data(), num_elements_ * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_(std::move(other.data_ptr_)),
      shape_(std::move(other.shape_)),
      num_elements_(other.num_elements_),
      device_(other.device_)
{
    other.num_elements_ = 0;
}

// Copy assignment operator
Tensor &Tensor::operator=(const Tensor &other)
{
    if (this == &other)
    {
        return *this;
    }

    // copy-and-swap idiom
    Tensor temp(other);
    *this = std::move(temp);
    return *this;
}

// Move assignment operator
Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this == &other)
        return *this;

    data_ptr_ = std::move(other.data_ptr_);
    shape_ = std::move(other.shape_);
    num_elements_ = other.num_elements_;
    device_ = other.device_;

    other.num_elements_ = 0;
    return *this;
}

// Move data to CPU
Tensor Tensor::cpu() const
{
    if (device_ == Device::kCPU)
    {
        return *this;  // Already on CPU, return copy
    }

    Tensor cpu_tensor(shape_, Device::kCPU);
    if (num_elements_ > 0) {
        CUDA_CHECK(cudaMemcpy(cpu_tensor.data(), this->data(), num_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return cpu_tensor;
}

// Move data to GPU
Tensor Tensor::gpu() const
{
    if (device_ == Device::kGPU)
    {
        return *this;  // Already on GPU, return copy
    }

    Tensor gpu_tensor(shape_, Device::kGPU);
    if (num_elements_ > 0) {
        CUDA_CHECK(cudaMemcpy(gpu_tensor.data(), this->data(), num_elements_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    return gpu_tensor;
}

// Initialize with zeros
void Tensor::zeros()
{
    if (num_elements_ == 0) return;

    if (device_ == Device::kCPU)
    {
        std::memset(data(), 0, num_elements_ * sizeof(float));
    }
    else
    {
        CUDA_CHECK(cudaMemset(data(), 0, num_elements_ * sizeof(float)));
    }
}

// Initialize with ones
void Tensor::ones()
{
    if (num_elements_ == 0) return;

    if (device_ == Device::kCPU)
    {
        std::fill_n(data(), num_elements_, 1.0f);
    }
    else
    {
        // For GPU, create data on CPU first, then copy to GPU
        std::vector<float> ones_data(num_elements_, 1.0f);
        CUDA_CHECK(cudaMemcpy(data(), ones_data.data(), num_elements_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

// Random initialization
void Tensor::random(float min, float max)
{
    if (num_elements_ == 0) return;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    if (device_ == Device::kCPU)
    {
        float* ptr = data();
        for (size_t i = 0; i < num_elements_; ++i)
        {
            ptr[i] = dis(gen);
        }
    }
    else
    {
        // For GPU, generate random numbers on CPU first, then copy to GPU
        std::vector<float> random_data(num_elements_);
        for (size_t i = 0; i < num_elements_; ++i)
        {
            random_data[i] = dis(gen);
        }
        CUDA_CHECK(cudaMemcpy(data(), random_data.data(), num_elements_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}