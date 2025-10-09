#include "Tensor.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

class TensorTester {
private:
    int passed_tests = 0;
    int total_tests = 0;

    void assert_test(bool condition, const std::string& test_name) {
        total_tests++;
        if (condition) {
            passed_tests++;
            std::cout << "✓ " << test_name << " PASSED" << std::endl;
        } else {
            std::cout << "✗ " << test_name << " FAILED" << std::endl;
        }
    }

    void print_tensor_info(const Tensor& tensor, const std::string& name) {
        std::cout << name << ": ";
        std::cout << "shape=[";
        const auto& shape = tensor.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ",";
        }
        std::cout << "], size=" << tensor.size()
                  << ", device=" << (tensor.device() == Device::CPU ? "CPU" : "GPU")
                  << std::endl;
    }

    void print_tensor_data(const Tensor& tensor, const std::string& name, int max_elements = 10) {
        // Convert to CPU for printing if needed
        Tensor cpu_tensor = (tensor.device() == Device::GPU) ? tensor.cpu() : tensor;

        std::cout << name << " data: [";
        float* data = cpu_tensor.data();
        size_t size = cpu_tensor.size();
        size_t print_size = std::min(static_cast<size_t>(max_elements), size);

        for (size_t i = 0; i < print_size; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data[i];
            if (i < print_size - 1) std::cout << ", ";
        }
        if (size > print_size) {
            std::cout << ", ... (" << (size - print_size) << " more)";
        }
        std::cout << "]" << std::endl;
    }

    bool tensors_equal(const Tensor& a, const Tensor& b, float tolerance = 1e-6) {
        if (a.shape() != b.shape()) return false;

        Tensor cpu_a = (a.device() == Device::GPU) ? a.cpu() : a;
        Tensor cpu_b = (b.device() == Device::GPU) ? b.cpu() : b;

        float* data_a = cpu_a.data();
        float* data_b = cpu_b.data();

        for (size_t i = 0; i < cpu_a.size(); ++i) {
            if (std::abs(data_a[i] - data_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    void fill_tensor(Tensor& tensor, float value) {
        Tensor cpu_tensor = (tensor.device() == Device::GPU) ? tensor.cpu() : tensor;
        float* data = cpu_tensor.data();
        for (size_t i = 0; i < cpu_tensor.size(); ++i) {
            data[i] = value;
        }

        if (tensor.device() == Device::GPU) {
            tensor = cpu_tensor.gpu();
        } else {
            tensor = cpu_tensor;
        }
    }

    void fill_tensor_sequence(Tensor& tensor, float start = 0.0f, float step = 1.0f) {
        Tensor cpu_tensor = (tensor.device() == Device::GPU) ? tensor.cpu() : tensor;
        float* data = cpu_tensor.data();
        for (size_t i = 0; i < cpu_tensor.size(); ++i) {
            data[i] = start + i * step;
        }

        if (tensor.device() == Device::GPU) {
            tensor = cpu_tensor.gpu();
        } else {
            tensor = cpu_tensor;
        }
    }

public:
    void test_basic_construction() {
        std::cout << "\n=== Testing Basic Construction ===\n";

        // Test 1D tensor
        {
            std::vector<int> shape1d = {5};
            Tensor tensor1d(shape1d, Device::CPU);
            print_tensor_info(tensor1d, "1D Tensor");
            assert_test(tensor1d.shape() == shape1d, "1D tensor shape");
            assert_test(tensor1d.size() == 5, "1D tensor size");
            assert_test(tensor1d.device() == Device::CPU, "1D tensor device");
            assert_test(tensor1d.data() != nullptr, "1D tensor data pointer");
        }

        // Test 2D tensor
        {
            std::vector<int> shape2d = {3, 4};
            Tensor tensor2d(shape2d, Device::CPU);
            print_tensor_info(tensor2d, "2D Tensor");
            assert_test(tensor2d.shape() == shape2d, "2D tensor shape");
            assert_test(tensor2d.size() == 12, "2D tensor size");
        }

        // Test 3D tensor
        {
            std::vector<int> shape3d = {2, 3, 4};
            Tensor tensor3d(shape3d, Device::CPU);
            print_tensor_info(tensor3d, "3D Tensor");
            assert_test(tensor3d.shape() == shape3d, "3D tensor shape");
            assert_test(tensor3d.size() == 24, "3D tensor size");
        }

        // Test empty tensor
        {
            std::vector<int> empty_shape = {0};
            Tensor empty_tensor(empty_shape, Device::CPU);
            assert_test(empty_tensor.size() == 0, "Empty tensor size");
            assert_test(empty_tensor.data() == nullptr, "Empty tensor data pointer");
        }
    }

    void test_copy_operations() {
        std::cout << "\n=== Testing Copy Operations ===\n";

        std::vector<int> shape = {2, 3};

        // Test copy constructor
        {
            Tensor original(shape, Device::CPU);
            fill_tensor_sequence(original, 1.0f, 1.0f);
            print_tensor_data(original, "Original");

            Tensor copied(original);
            print_tensor_data(copied, "Copied");

            assert_test(tensors_equal(original, copied), "Copy constructor data equality");
            assert_test(original.shape() == copied.shape(), "Copy constructor shape");
            assert_test(original.device() == copied.device(), "Copy constructor device");

            // Verify deep copy by modifying original
            fill_tensor(original, 99.0f);
            assert_test(!tensors_equal(original, copied), "Deep copy verification");
        }

        // Test assignment operator
        {
            Tensor tensor1(shape, Device::CPU);
            fill_tensor_sequence(tensor1, 10.0f, 2.0f);

            Tensor tensor2({1}, Device::CPU);
            fill_tensor(tensor2, 0.0f);

            tensor2 = tensor1;

            assert_test(tensors_equal(tensor1, tensor2), "Assignment operator data equality");
            assert_test(tensor1.shape() == tensor2.shape(), "Assignment operator shape");
            assert_test(tensor1.device() == tensor2.device(), "Assignment operator device");
        }

        // Test self-assignment
        {
            Tensor tensor(shape, Device::CPU);
            fill_tensor_sequence(tensor, 5.0f, 1.0f);
            Tensor original_copy(tensor);

            tensor = tensor;  // Self-assignment

            assert_test(tensors_equal(tensor, original_copy), "Self-assignment safety");
        }
    }

    void test_device_migration() {
        std::cout << "\n=== Testing Device Migration ===\n";

        std::vector<int> shape = {2, 3};

        // Test CPU to GPU migration
        {
            Tensor cpu_tensor(shape, Device::CPU);
            fill_tensor_sequence(cpu_tensor, 1.0f, 1.0f);
            print_tensor_data(cpu_tensor, "CPU Tensor");

            Tensor gpu_tensor = cpu_tensor.gpu();
            print_tensor_info(gpu_tensor, "GPU Tensor");

            assert_test(gpu_tensor.device() == Device::GPU, "CPU to GPU device migration");
            assert_test(gpu_tensor.shape() == cpu_tensor.shape(), "CPU to GPU shape preservation");

            // Convert back to CPU to compare data
            Tensor gpu_to_cpu = gpu_tensor.cpu();
            assert_test(tensors_equal(cpu_tensor, gpu_to_cpu), "CPU to GPU data preservation");
        }

        // Test GPU to CPU migration
        {
            Tensor cpu_original(shape, Device::CPU);
            fill_tensor_sequence(cpu_original, 10.0f, 2.0f);

            Tensor gpu_tensor = cpu_original.gpu();
            Tensor cpu_tensor = gpu_tensor.cpu();

            assert_test(cpu_tensor.device() == Device::CPU, "GPU to CPU device migration");
            assert_test(tensors_equal(cpu_original, cpu_tensor), "GPU to CPU data preservation");
        }

        // Test migration when already on target device
        {
            Tensor cpu_tensor(shape, Device::CPU);
            fill_tensor_sequence(cpu_tensor, 5.0f, 1.0f);

            Tensor cpu_to_cpu = cpu_tensor.cpu();
            assert_test(tensors_equal(cpu_tensor, cpu_to_cpu), "CPU to CPU migration");

            Tensor gpu_tensor = cpu_tensor.gpu();
            Tensor gpu_to_gpu = gpu_tensor.gpu();
            assert_test(gpu_to_gpu.device() == Device::GPU, "GPU to GPU device check");

            Tensor gpu_to_cpu = gpu_to_gpu.cpu();
            assert_test(tensors_equal(cpu_tensor, gpu_to_cpu), "GPU to GPU to CPU data consistency");
        }
    }

    void test_error_conditions() {
        std::cout << "\n=== Testing Error Conditions ===\n";

        // Test invalid dimensions
        {
            bool caught_exception = false;
            try {
                std::vector<int> invalid_shape = {2, -1, 3};
                Tensor tensor(invalid_shape, Device::CPU);
            } catch (const std::invalid_argument& e) {
                caught_exception = true;
                std::cout << "Caught expected exception: " << e.what() << std::endl;
            }
            assert_test(caught_exception, "Invalid dimension detection");
        }

        // Test zero dimension
        {
            bool caught_exception = false;
            try {
                std::vector<int> zero_shape = {2, 0, 3};
                Tensor tensor(zero_shape, Device::CPU);
            } catch (const std::invalid_argument& e) {
                caught_exception = true;
                std::cout << "Caught expected exception: " << e.what() << std::endl;
            }
            assert_test(caught_exception, "Zero dimension detection");
        }
    }

    void test_memory_management() {
        std::cout << "\n=== Testing Memory Management ===\n";

        // Test large tensor creation and destruction
        {
            const size_t large_size = 1000000;  // 1M elements
            std::vector<int> large_shape = {1000, 1000};

            {
                Tensor large_tensor(large_shape, Device::CPU);
                assert_test(large_tensor.size() == large_size, "Large tensor size");
                assert_test(large_tensor.data() != nullptr, "Large tensor data allocation");
                fill_tensor(large_tensor, 42.0f);

                // Test copy of large tensor
                Tensor large_copy(large_tensor);
                assert_test(tensors_equal(large_tensor, large_copy), "Large tensor copy");
            }
            // Tensor destructors should be called here
            std::cout << "Large tensors destructed successfully" << std::endl;
        }

        // Test multiple tensor operations
        {
            std::vector<Tensor> tensors;
            for (int i = 0; i < 10; ++i) {
                std::vector<int> shape = {10, 10};
                tensors.emplace_back(shape, Device::CPU);
                fill_tensor_sequence(tensors.back(), static_cast<float>(i), 1.0f);
            }

            assert_test(tensors.size() == 10, "Multiple tensor creation");

            // Verify each tensor has correct data
            bool all_correct = true;
            for (size_t i = 0; i < tensors.size(); ++i) {
                Tensor& t = tensors[i];
                float* data = t.data();
                if (data[0] != static_cast<float>(i)) {
                    all_correct = false;
                    break;
                }
            }
            assert_test(all_correct, "Multiple tensor data integrity");
        }
    }

    void test_edge_cases() {
        std::cout << "\n=== Testing Edge Cases ===\n";

        // Test 1x1 tensor
        {
            std::vector<int> single_shape = {1, 1};
            Tensor single_tensor(single_shape, Device::CPU);
            fill_tensor(single_tensor, 3.14f);

            assert_test(single_tensor.size() == 1, "Single element tensor size");
            assert_test(std::abs(single_tensor.data()[0] - 3.14f) < 1e-6, "Single element tensor value");
        }

        // Test very high dimensional tensor
        {
            std::vector<int> high_dim_shape = {2, 1, 1, 1, 1, 1, 3};
            Tensor high_dim_tensor(high_dim_shape, Device::CPU);

            assert_test(high_dim_tensor.size() == 6, "High dimensional tensor size");
            assert_test(high_dim_tensor.shape() == high_dim_shape, "High dimensional tensor shape");
        }

        // Test assignment between different sized tensors
        {
            Tensor small_tensor({2, 2}, Device::CPU);
            Tensor large_tensor({3, 4}, Device::CPU);

            fill_tensor_sequence(small_tensor, 1.0f, 1.0f);
            fill_tensor_sequence(large_tensor, 10.0f, 1.0f);

            small_tensor = large_tensor;

            assert_test(small_tensor.shape() == large_tensor.shape(), "Assignment shape change");
            assert_test(small_tensor.size() == large_tensor.size(), "Assignment size change");
            assert_test(tensors_equal(small_tensor, large_tensor), "Assignment data transfer");
        }
    }

    void run_all_tests() {
        std::cout << "========================================\n";
        std::cout << "         Tensor Class Test Suite       \n";
        std::cout << "========================================\n";

        test_basic_construction();
        test_copy_operations();
        test_device_migration();
        test_error_conditions();
        test_memory_management();
        test_edge_cases();

        std::cout << "\n========================================\n";
        std::cout << "Test Results: " << passed_tests << "/" << total_tests << " tests passed\n";
        if (passed_tests == total_tests) {
            std::cout << "🎉 All tests PASSED! 🎉\n";
        } else {
            std::cout << "❌ " << (total_tests - passed_tests) << " tests FAILED!\n";
        }
        std::cout << "========================================\n";
    }
};

int main() {
    try {
        TensorTester tester;
        tester.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test suite crashed with exception: " << e.what() << std::endl;
        return 1;
    }
}