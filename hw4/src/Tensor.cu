#include "Tensor.h"
#include <cuda_runtime.h>
#include <cstring>

// Calculate total tensor elements
size_t Tensor::calculate_total_size(const std::vector<int>& shape) const {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }

    size_t total = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor dimension must be positive");
        }
        total *= dim;
    }
    return total;
}

// Allocate memory
void Tensor::allocate_memory() {
    if (total_size_ == 0) {
        data_ = nullptr;
        return;
    }

    if (device_ == Device::CPU) {
        // Allocate memory on CPU
        data_ = new float[total_size_];
        // Initialize to 0
        std::memset(data_, 0, total_size_ * sizeof(float));
    } else {  // Device::GPU
        // Allocate memory on GPU
        cudaError_t error = cudaMalloc(&data_, total_size_ * sizeof(float));
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU memory allocation failed: " + std::string(cudaGetErrorString(error)));
        }
        // Initialize GPU memory to 0
        cudaMemset(data_, 0, total_size_ * sizeof(float));
    }
}

// Deallocate memory
void Tensor::deallocate_memory() {
    if (data_ != nullptr) {
        if (device_ == Device::CPU) {
            delete[] data_;
        } else {  // Device::GPU
            cudaFree(data_);
        }
        data_ = nullptr;
    }
}

// Copy data from another tensor
void Tensor::copy_data_from(const Tensor& other) {
    if (total_size_ != other.total_size_) {
        throw std::invalid_argument("Tensor sizes do not match, cannot copy data");
    }

    if (total_size_ == 0) {
        return;  // Empty tensor, no need to copy
    }

    if (device_ == Device::CPU && other.device_ == Device::CPU) {
        // CPU to CPU: direct memory copy
        std::memcpy(data_, other.data_, total_size_ * sizeof(float));
    }
    else if (device_ == Device::GPU && other.device_ == Device::GPU) {
        // GPU to GPU: device to device memory copy
        cudaError_t error = cudaMemcpy(data_, other.data_,
                                      total_size_ * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU to GPU data copy failed: " + std::string(cudaGetErrorString(error)));
        }
    }
    else if (device_ == Device::CPU && other.device_ == Device::GPU) {
        // GPU to CPU: device to host copy
        cudaError_t error = cudaMemcpy(data_, other.data_,
                                      total_size_ * sizeof(float), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU to CPU data copy failed: " + std::string(cudaGetErrorString(error)));
        }
    }
    else {  // device_ == Device::GPU && other.device_ == Device::CPU
        // CPU to GPU: host to device copy
        cudaError_t error = cudaMemcpy(data_, other.data_,
                                      total_size_ * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error("CPU to GPU data copy failed: " + std::string(cudaGetErrorString(error)));
        }
    }
}

// Constructor
Tensor::Tensor(const std::vector<int>& shape, Device device)
    : shape_(shape), device_(device) {
    total_size_ = calculate_total_size(shape);
    allocate_memory();
}

// Copy constructor
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), device_(other.device_), total_size_(other.total_size_) {
    allocate_memory();
    copy_data_from(other);
}

// Destructor
Tensor::~Tensor() {
    deallocate_memory();
}

// Assignment operator: copy another tensor's content to current tensor
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;  // Self-assignment check
    }

    // Release original memory
    deallocate_memory();

    // Update tensor properties
    shape_ = other.shape_;
    device_ = other.device_;
    total_size_ = other.total_size_;

    // Reallocate memory and copy data
    allocate_memory();
    copy_data_from(other);

    return *this;
}

// Migrate to CPU
Tensor Tensor::cpu() const {
    if (device_ == Device::CPU) {
        // If already on CPU, return a copy directly
        return Tensor(*this);
    }

    // Create CPU tensor and copy data
    Tensor cpu_tensor(shape_, Device::CPU);
    cpu_tensor.copy_data_from(*this);
    return cpu_tensor;
}

// Migrate to GPU
Tensor Tensor::gpu() const {
    if (device_ == Device::GPU) {
        // If already on GPU, return a copy directly
        return Tensor(*this);
    }

    // Create GPU tensor and copy data
    Tensor gpu_tensor(shape_, Device::GPU);
    gpu_tensor.copy_data_from(*this);
    return gpu_tensor;
}