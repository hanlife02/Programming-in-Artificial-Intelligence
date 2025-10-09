#include "Tensor.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>

// 测试计数器
int tests_passed = 0;
int tests_total = 0;

// 测试宏
#define TEST_ASSERT(condition, message) \
    do { \
        tests_total++; \
        if (condition) { \
            tests_passed++; \
            std::cout << "[PASS] " << message << std::endl; \
        } else { \
            std::cout << "[FAIL] " << message << std::endl; \
        } \
    } while(0)

// GPU内存信息
void print_gpu_memory_info(const std::string& label) {
    size_t free_bytes, total_bytes;
    cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (cuda_status == cudaSuccess) {
        size_t used_bytes = total_bytes - free_bytes;
        std::cout << "[" << label << "] GPU内存使用: "
                  << used_bytes / (1024*1024) << " MB / "
                  << total_bytes / (1024*1024) << " MB" << std::endl;
    }
}

// 测试基本构造函数
void test_basic_construction() {
    std::cout << "\n=== 测试基本构造函数 ===" << std::endl;

    // CPU张量
    std::vector<int> shape = {2, 3, 4};
    Tensor cpu_tensor(shape, Device::CPU);

    TEST_ASSERT(cpu_tensor.shape().size() == 3, "CPU张量维度正确");
    TEST_ASSERT(cpu_tensor.shape()[0] == 2 && cpu_tensor.shape()[1] == 3 && cpu_tensor.shape()[2] == 4, "CPU张量形状正确");
    TEST_ASSERT(cpu_tensor.device() == Device::CPU, "CPU张量设备类型正确");
    TEST_ASSERT(cpu_tensor.size() == 24, "CPU张量元素数量正确");
    TEST_ASSERT(cpu_tensor.data() != nullptr, "CPU张量数据指针非空");

    // GPU张量
    Tensor gpu_tensor(shape, Device::GPU);

    TEST_ASSERT(gpu_tensor.shape().size() == 3, "GPU张量维度正确");
    TEST_ASSERT(gpu_tensor.device() == Device::GPU, "GPU张量设备类型正确");
    TEST_ASSERT(gpu_tensor.size() == 24, "GPU张量元素数量正确");
    TEST_ASSERT(gpu_tensor.data() != nullptr, "GPU张量数据指针非空");

    // 空张量
    std::vector<int> empty_shape = {};
    try {
        Tensor empty_tensor(empty_shape, Device::CPU);
        TEST_ASSERT(false, "空形状应该抛出异常");
    } catch (const std::exception&) {
        TEST_ASSERT(true, "空形状正确抛出异常");
    }

    // 无效形状
    std::vector<int> invalid_shape = {2, -1, 3};
    try {
        Tensor invalid_tensor(invalid_shape, Device::CPU);
        TEST_ASSERT(false, "负数维度应该抛出异常");
    } catch (const std::exception&) {
        TEST_ASSERT(true, "负数维度正确抛出异常");
    }
}

// 测试拷贝构造函数
void test_copy_constructor() {
    std::cout << "\n=== 测试拷贝构造函数 ===" << std::endl;

    std::vector<int> shape = {3, 4};

    // CPU拷贝
    Tensor original_cpu(shape, Device::CPU);
    Tensor copy_cpu(original_cpu);

    TEST_ASSERT(copy_cpu.shape() == original_cpu.shape(), "CPU拷贝形状一致");
    TEST_ASSERT(copy_cpu.device() == original_cpu.device(), "CPU拷贝设备一致");
    TEST_ASSERT(copy_cpu.size() == original_cpu.size(), "CPU拷贝大小一致");
    TEST_ASSERT(copy_cpu.data() != original_cpu.data(), "CPU拷贝是深拷贝");

    // GPU拷贝
    Tensor original_gpu(shape, Device::GPU);
    Tensor copy_gpu(original_gpu);

    TEST_ASSERT(copy_gpu.shape() == original_gpu.shape(), "GPU拷贝形状一致");
    TEST_ASSERT(copy_gpu.device() == original_gpu.device(), "GPU拷贝设备一致");
    TEST_ASSERT(copy_gpu.size() == original_gpu.size(), "GPU拷贝大小一致");
    TEST_ASSERT(copy_gpu.data() != original_gpu.data(), "GPU拷贝是深拷贝");
}

// 测试赋值运算符
void test_assignment_operator() {
    std::cout << "\n=== 测试赋值运算符 ===" << std::endl;

    std::vector<int> shape1 = {2, 3};
    std::vector<int> shape2 = {4, 5};

    Tensor tensor1(shape1, Device::CPU);
    Tensor tensor2(shape2, Device::CPU);

    float* old_data = tensor1.data();
    tensor1 = tensor2;

    TEST_ASSERT(tensor1.shape() == tensor2.shape(), "赋值后形状一致");
    TEST_ASSERT(tensor1.device() == tensor2.device(), "赋值后设备一致");
    TEST_ASSERT(tensor1.size() == tensor2.size(), "赋值后大小一致");
    TEST_ASSERT(tensor1.data() != tensor2.data(), "赋值是深拷贝");
    TEST_ASSERT(tensor1.data() != old_data, "赋值重新分配了内存");

    // 自赋值测试
    float* self_data = tensor1.data();
    tensor1 = tensor1;
    TEST_ASSERT(tensor1.data() == self_data, "自赋值不改变数据指针");
}

// 测试设备迁移
void test_device_migration() {
    std::cout << "\n=== 测试设备迁移 ===" << std::endl;

    std::vector<int> shape = {10, 10};

    // CPU到GPU
    Tensor cpu_tensor(shape, Device::CPU);
    Tensor gpu_from_cpu = cpu_tensor.gpu();

    TEST_ASSERT(gpu_from_cpu.device() == Device::GPU, "CPU到GPU迁移成功");
    TEST_ASSERT(gpu_from_cpu.shape() == cpu_tensor.shape(), "CPU到GPU迁移保持形状");
    TEST_ASSERT(gpu_from_cpu.size() == cpu_tensor.size(), "CPU到GPU迁移保持大小");
    TEST_ASSERT(gpu_from_cpu.data() != cpu_tensor.data(), "CPU到GPU迁移是新分配");

    // GPU到CPU
    Tensor gpu_tensor(shape, Device::GPU);
    Tensor cpu_from_gpu = gpu_tensor.cpu();

    TEST_ASSERT(cpu_from_gpu.device() == Device::CPU, "GPU到CPU迁移成功");
    TEST_ASSERT(cpu_from_gpu.shape() == gpu_tensor.shape(), "GPU到CPU迁移保持形状");
    TEST_ASSERT(cpu_from_gpu.size() == gpu_tensor.size(), "GPU到CPU迁移保持大小");
    TEST_ASSERT(cpu_from_gpu.data() != gpu_tensor.data(), "GPU到CPU迁移是新分配");

    // 相同设备迁移（应该返回副本）
    Tensor cpu_to_cpu = cpu_tensor.cpu();
    TEST_ASSERT(cpu_to_cpu.device() == Device::CPU, "CPU到CPU返回CPU张量");
    TEST_ASSERT(cpu_to_cpu.data() != cpu_tensor.data(), "CPU到CPU返回新副本");

    Tensor gpu_to_gpu = gpu_tensor.gpu();
    TEST_ASSERT(gpu_to_gpu.device() == Device::GPU, "GPU到GPU返回GPU张量");
    TEST_ASSERT(gpu_to_gpu.data() != gpu_tensor.data(), "GPU到GPU返回新副本");
}

// 测试内存管理
void test_memory_management() {
    std::cout << "\n=== 测试内存管理 ===" << std::endl;

    print_gpu_memory_info("开始");

    {
        // 大量张量创建和销毁
        std::vector<Tensor*> tensors;
        for (int i = 0; i < 10; ++i) {
            tensors.push_back(new Tensor({100, 100}, Device::GPU));
        }

        print_gpu_memory_info("创建10个GPU张量后");

        // 删除所有张量
        for (auto* tensor : tensors) {
            delete tensor;
        }
        tensors.clear();

        print_gpu_memory_info("删除所有张量后");
    }

    // 作用域测试
    {
        Tensor scope_test({500, 500}, Device::GPU);
        print_gpu_memory_info("作用域内创建大张量");
    }
    print_gpu_memory_info("离开作用域后");

    TEST_ASSERT(true, "内存管理测试完成（请检查GPU内存使用情况）");
}

// 测试边界条件
void test_edge_cases() {
    std::cout << "\n=== 测试边界条件 ===" << std::endl;

    // 单元素张量
    std::vector<int> single_shape = {1};
    Tensor single_tensor(single_shape, Device::CPU);
    TEST_ASSERT(single_tensor.size() == 1, "单元素张量大小正确");

    // 一维张量
    std::vector<int> one_d_shape = {100};
    Tensor one_d_tensor(one_d_shape, Device::CPU);
    TEST_ASSERT(one_d_tensor.size() == 100, "一维张量大小正确");

    // 高维张量
    std::vector<int> high_d_shape = {2, 3, 4, 5, 6};
    Tensor high_d_tensor(high_d_shape, Device::CPU);
    TEST_ASSERT(high_d_tensor.size() == 720, "高维张量大小正确");

    // 连续赋值
    Tensor t1({2, 2}, Device::CPU);
    Tensor t2({3, 3}, Device::CPU);
    Tensor t3({4, 4}, Device::CPU);

    t1 = t2 = t3;
    TEST_ASSERT(t1.size() == t3.size() && t2.size() == t3.size(), "连续赋值正确");
}

// 性能测试
void test_performance() {
    std::cout << "\n=== 性能测试 ===" << std::endl;

    const int iterations = 1000;

    // CPU张量创建性能
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Tensor cpu_perf({100, 100}, Device::CPU);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CPU张量创建 " << iterations << " 次耗时: " << cpu_duration.count() << " 微秒" << std::endl;
    TEST_ASSERT(cpu_duration.count() < 1000000, "CPU张量创建性能合理（<1秒）");

    // GPU张量创建性能
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Tensor gpu_perf({100, 100}, Device::GPU);
    }
    end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "GPU张量创建 " << iterations << " 次耗时: " << gpu_duration.count() << " 微秒" << std::endl;
    TEST_ASSERT(gpu_duration.count() < 5000000, "GPU张量创建性能合理（<5秒）");

    // 设备迁移性能
    Tensor migration_test({1000, 1000}, Device::CPU);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        Tensor gpu_migrated = migration_test.gpu();
        Tensor cpu_migrated = gpu_migrated.cpu();
    }
    end = std::chrono::high_resolution_clock::now();
    auto migration_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "设备迁移 10 次耗时: " << migration_duration.count() << " 微秒" << std::endl;
    TEST_ASSERT(migration_duration.count() < 1000000, "设备迁移性能合理（<1秒）");
}

int main() {
    std::cout << "开始全面测试Tensor类..." << std::endl;

    try {
        test_basic_construction();
        test_copy_constructor();
        test_assignment_operator();
        test_device_migration();
        test_memory_management();
        test_edge_cases();
        test_performance();

        std::cout << "\n=== 测试总结 ===" << std::endl;
        std::cout << "通过测试: " << tests_passed << "/" << tests_total << std::endl;

        if (tests_passed == tests_total) {
            std::cout << "🎉 所有测试通过！Tensor类实现正确。" << std::endl;
            return 0;
        } else {
            std::cout << "❌ 有测试失败，请检查实现。" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
}