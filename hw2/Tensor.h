#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>

// 设备类型
enum class Device {
    CPU,
    GPU
};

class Tensor {
private:
    std::vector<int> shape_;          // 张量形状
    Device device_;                   // 设备类型
    size_t total_size_;              // 总元素数量
    float* data_;                    // 数据指针，手动管理内存

    // 实用方法
    size_t calculate_total_size(const std::vector<int>& shape) const;  // 计算总元素数
    void allocate_memory();                                           // 分配内存
    void deallocate_memory();                                         // 释放内存
    void copy_data_from(const Tensor& other);                        // 从另一个张量复制数据

public:
    // 构造函数和析构函数
    Tensor(const std::vector<int>& shape, Device device);
    Tensor(const Tensor& other);
    ~Tensor();

    // 赋值运算符
    Tensor& operator=(const Tensor& other);

    // 迁移方法
    Tensor cpu() const;
    Tensor gpu() const;

    // 访问方法
    const std::vector<int>& shape() const { return shape_; }
    Device device() const { return device_; }
    size_t size() const { return total_size_; }
    float* data() const { return data_; }
};