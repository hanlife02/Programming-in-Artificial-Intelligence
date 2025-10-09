#include "Tensor.h"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "=== 测试Tensor类基本功能 ===" << std::endl;

        // 测试1: 创建CPU张量
        std::cout << "\n--- 测试1: 创建CPU张量 ---" << std::endl;
        std::vector<int> shape = {2, 3, 4};  // 形状: 2x3x4
        Tensor cpu_tensor(shape, Device::CPU);
        std::cout << "CPU张量形状: [";
        for (size_t i = 0; i < cpu_tensor.shape().size(); ++i) {
            std::cout << cpu_tensor.shape()[i];
            if (i < cpu_tensor.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "], 总元素数: " << cpu_tensor.size() << std::endl;

        // 测试2: 创建GPU张量
        std::cout << "\n--- 测试2: 创建GPU张量 ---" << std::endl;
        Tensor gpu_tensor(shape, Device::GPU);
        std::cout << "GPU张量形状: [";
        for (size_t i = 0; i < gpu_tensor.shape().size(); ++i) {
            std::cout << gpu_tensor.shape()[i];
            if (i < gpu_tensor.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "], 总元素数: " << gpu_tensor.size() << std::endl;

        // 测试3: CPU到GPU迁移
        std::cout << "\n--- 测试3: CPU到GPU迁移 ---" << std::endl;
        Tensor gpu_from_cpu = cpu_tensor.gpu();
        std::cout << "迁移后设备类型: " << (gpu_from_cpu.device() == Device::GPU ? "GPU" : "CPU") << std::endl;

        // 测试4: GPU到CPU迁移
        std::cout << "\n--- 测试4: GPU到CPU迁移 ---" << std::endl;
        Tensor cpu_from_gpu = gpu_tensor.cpu();
        std::cout << "迁移后设备类型: " << (cpu_from_gpu.device() == Device::CPU ? "CPU" : "GPU") << std::endl;

        // 测试5: 连续迁移
        std::cout << "\n--- 测试5: 连续迁移测试 ---" << std::endl;
        Tensor tensor1(shape, Device::CPU);
        Tensor tensor2 = tensor1.gpu();  // CPU -> GPU
        Tensor tensor3 = tensor2.cpu();  // GPU -> CPU
        Tensor tensor4 = tensor3.gpu();  // CPU -> GPU
        std::cout << "连续迁移完成: CPU -> GPU -> CPU -> GPU" << std::endl;

        // 测试6: 拷贝构造函数
        std::cout << "\n--- 测试6: 拷贝构造函数 ---" << std::endl;
        Tensor copy_cpu(cpu_tensor);
        Tensor copy_gpu(gpu_tensor);
        std::cout << "拷贝构造测试完成" << std::endl;

        // 测试7: 赋值运算符
        std::cout << "\n--- 测试7: 赋值运算符 ---" << std::endl;
        Tensor assign_test(shape, Device::CPU);
        assign_test = gpu_tensor;
        std::cout << "赋值运算符测试完成" << std::endl;

        std::cout << "\n=== 所有测试完成! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}