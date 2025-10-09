#include "Tensor.h"
#include <cuda_runtime.h>
#include <cstring>

// ============================================================================
// 私有辅助方法实现
// ============================================================================

// 计算张量总元素数量
size_t Tensor::calculate_total_size(const std::vector<int>& shape) const {
    size_t total = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("张量维度必须为正数");
        }
        total *= dim;
    }
    return total;
}

// 分配内存（根据设备类型分配CPU或GPU内存）
void Tensor::allocate_memory() {
    if (total_size_ == 0) {
        data_ = nullptr;
        return;
    }

    if (device_ == Device::CPU) {
        // 在CPU上分配内存，使用自定义删除器来管理内存
        float* cpu_ptr = new float[total_size_];
        // 初始化为0
        std::memset(cpu_ptr, 0, total_size_ * sizeof(float));
        // 使用自定义删除器创建智能指针
        data_ = std::shared_ptr<float>(cpu_ptr, [](float* ptr) {
            delete[] ptr;
        });
    } else {  // Device::GPU
        // 在GPU上分配内存
        float* gpu_ptr;
        cudaError_t error = cudaMalloc(&gpu_ptr, total_size_ * sizeof(float));
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU内存分配失败: " + std::string(cudaGetErrorString(error)));
        }
        // 初始化GPU内存为0
        cudaMemset(gpu_ptr, 0, total_size_ * sizeof(float));
        // 使用自定义删除器创建智能指针
        data_ = std::shared_ptr<float>(gpu_ptr, [](float* ptr) {
            cudaFree(ptr);
        });
    }
}

// 从另一个张量复制数据（处理CPU-GPU之间的数据传输）
void Tensor::copy_data_from(const Tensor& other) {
    if (total_size_ != other.total_size_) {
        throw std::invalid_argument("张量大小不匹配，无法复制数据");
    }

    if (total_size_ == 0) {
        return;  // 空张量无需复制
    }

    // 根据源和目标设备类型选择复制方式
    if (device_ == Device::CPU && other.device_ == Device::CPU) {
        // CPU到CPU：直接内存复制
        std::memcpy(data_.get(), other.data_.get(), total_size_ * sizeof(float));
    }
    else if (device_ == Device::GPU && other.device_ == Device::GPU) {
        // GPU到GPU：设备间内存复制
        cudaError_t error = cudaMemcpy(data_.get(), other.data_.get(),
                                      total_size_ * sizeof(float), cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU到GPU数据复制失败: " + std::string(cudaGetErrorString(error)));
        }
    }
    else if (device_ == Device::CPU && other.device_ == Device::GPU) {
        // GPU到CPU：从设备复制到主机
        cudaError_t error = cudaMemcpy(data_.get(), other.data_.get(),
                                      total_size_ * sizeof(float), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU到CPU数据复制失败: " + std::string(cudaGetErrorString(error)));
        }
    }
    else {  // device_ == Device::GPU && other.device_ == Device::CPU
        // CPU到GPU：从主机复制到设备
        cudaError_t error = cudaMemcpy(data_.get(), other.data_.get(),
                                      total_size_ * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error("CPU到GPU数据复制失败: " + std::string(cudaGetErrorString(error)));
        }
    }
}

// ============================================================================
// 构造函数和析构函数实现
// ============================================================================

// 主构造函数：创建指定形状和设备的张量
Tensor::Tensor(const std::vector<int>& shape, Device device)
    : shape_(shape), device_(device) {
    total_size_ = calculate_total_size(shape);
    allocate_memory();
}

// 拷贝构造函数：创建另一个张量的副本
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), device_(other.device_), total_size_(other.total_size_) {
    allocate_memory();
    copy_data_from(other);
}

// ============================================================================
// 赋值运算符实现
// ============================================================================

// 赋值运算符：将另一个张量的内容复制到当前张量
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;  // 自赋值检查
    }

    // 更新张量属性
    shape_ = other.shape_;
    device_ = other.device_;
    total_size_ = other.total_size_;

    // 重新分配内存并复制数据
    allocate_memory();
    copy_data_from(other);

    return *this;
}

// ============================================================================
// 设备迁移方法实现
// ============================================================================

// 将张量迁移到CPU
Tensor Tensor::cpu() const {
    if (device_ == Device::CPU) {
        // 如果已经在CPU上，直接返回副本
        return Tensor(*this);
    }

    // 创建CPU张量并复制数据
    Tensor cpu_tensor(shape_, Device::CPU);
    cpu_tensor.copy_data_from(*this);
    return cpu_tensor;
}

// 将张量迁移到GPU
Tensor Tensor::gpu() const {
    if (device_ == Device::GPU) {
        // 如果已经在GPU上，直接返回副本
        return Tensor(*this);
    }

    // 创建GPU张量并复制数据
    Tensor gpu_tensor(shape_, Device::GPU);
    gpu_tensor.copy_data_from(*this);
    return gpu_tensor;
}