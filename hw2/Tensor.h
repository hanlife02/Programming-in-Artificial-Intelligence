#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

// 设备类型枚举
enum class Device {
    CPU,  // CPU设备
    GPU   // GPU设备
};

// Tensor类声明
class Tensor {
private:
    std::vector<int> shape_;          // 张量形状，使用vector存储各维度大小
    Device device_;                   // 当前张量所在的设备
    size_t total_size_;              // 总元素数量
    std::shared_ptr<float> data_;    // 数据指针，使用智能指针管理内存

    // 私有辅助方法
    size_t calculate_total_size(const std::vector<int>& shape) const;  // 计算总元素数
    void allocate_memory();                                           // 分配内存
    void copy_data_from(const Tensor& other);                        // 从另一个张量复制数据

public:
    // 构造函数和析构函数
    Tensor(const std::vector<int>& shape, Device device);  // 主构造函数
    Tensor(const Tensor& other);                          // 拷贝构造函数
    ~Tensor() = default;                                   // 析构函数（智能指针自动管理）

    // 赋值运算符
    Tensor& operator=(const Tensor& other);

    // 设备迁移方法
    Tensor cpu() const;  // 迁移到CPU
    Tensor gpu() const;  // 迁移到GPU

    // 访问方法
    const std::vector<int>& shape() const { return shape_; }  // 获取形状
    Device device() const { return device_; }                // 获取设备类型
    size_t size() const { return total_size_; }             // 获取总元素数
    float* data() const { return data_.get(); }             // 获取数据指针
};

#endif // TENSOR_H