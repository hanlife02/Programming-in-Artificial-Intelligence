#include "ActivationFunctions.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace ActivationFunctions;

void print_tensor(const Tensor& tensor, const std::string& name) {
    std::cout << name << " (device: " << (tensor.device() == Device::CPU ? "CPU" : "GPU") << "): ";

    // Move to CPU for printing if on GPU
    Tensor cpu_tensor = (tensor.device() == Device::GPU) ? tensor.cpu() : tensor;

    float* data = cpu_tensor.data();
    for (size_t i = 0; i < cpu_tensor.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;
}

void test_relu() {
    std::cout << "\n=== Testing ReLU Activation Function ===\n";

    // Test data: mix of positive, negative, and zero values
    std::vector<int> shape = {2, 3};
    float test_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};

    // Test on CPU
    std::cout << "\n--- CPU Test ---\n";
    Tensor cpu_input(shape, Device::CPU);
    std::memcpy(cpu_input.data(), test_data, sizeof(test_data));

    print_tensor(cpu_input, "Input");

    // Forward pass
    Tensor cpu_output = ReLU::forward(cpu_input);
    print_tensor(cpu_output, "ReLU Output");

    // Backward pass (simulate gradient from loss)
    Tensor cpu_grad_output(shape, Device::CPU);
    float grad_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::memcpy(cpu_grad_output.data(), grad_data, sizeof(grad_data));

    Tensor cpu_grad_input = ReLU::backward(cpu_grad_output, cpu_input);
    print_tensor(cpu_grad_input, "ReLU Gradient");

    // Test on GPU
    std::cout << "\n--- GPU Test ---\n";
    Tensor gpu_input = cpu_input.gpu();
    Tensor gpu_grad_output = cpu_grad_output.gpu();

    Tensor gpu_output = ReLU::forward(gpu_input);
    print_tensor(gpu_output, "ReLU Output");

    Tensor gpu_grad_input = ReLU::backward(gpu_grad_output, gpu_input);
    print_tensor(gpu_grad_input, "ReLU Gradient");

    // Verify CPU and GPU results match
    Tensor gpu_output_cpu = gpu_output.cpu();
    Tensor gpu_grad_cpu = gpu_grad_input.cpu();

    bool forward_match = true, backward_match = true;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        if (std::abs(cpu_output.data()[i] - gpu_output_cpu.data()[i]) > 1e-6) {
            forward_match = false;
        }
        if (std::abs(cpu_grad_input.data()[i] - gpu_grad_cpu.data()[i]) > 1e-6) {
            backward_match = false;
        }
    }

    std::cout << "CPU-GPU Forward Match: " << (forward_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "CPU-GPU Backward Match: " << (backward_match ? "PASS" : "FAIL") << std::endl;
}

void test_sigmoid() {
    std::cout << "\n=== Testing Sigmoid Activation Function ===\n";

    // Test data: range of values
    std::vector<int> shape = {2, 3};
    float test_data[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f};

    // Test on CPU
    std::cout << "\n--- CPU Test ---\n";
    Tensor cpu_input(shape, Device::CPU);
    std::memcpy(cpu_input.data(), test_data, sizeof(test_data));

    print_tensor(cpu_input, "Input");

    // Forward pass
    Tensor cpu_output = Sigmoid::forward(cpu_input);
    print_tensor(cpu_output, "Sigmoid Output");

    // Backward pass (simulate gradient from loss)
    Tensor cpu_grad_output(shape, Device::CPU);
    float grad_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::memcpy(cpu_grad_output.data(), grad_data, sizeof(grad_data));

    Tensor cpu_grad_input = Sigmoid::backward(cpu_grad_output, cpu_output);
    print_tensor(cpu_grad_input, "Sigmoid Gradient");

    // Test on GPU
    std::cout << "\n--- GPU Test ---\n";
    Tensor gpu_input = cpu_input.gpu();
    Tensor gpu_grad_output = cpu_grad_output.gpu();

    Tensor gpu_output = Sigmoid::forward(gpu_input);
    print_tensor(gpu_output, "Sigmoid Output");

    Tensor gpu_grad_input = Sigmoid::backward(gpu_grad_output, gpu_output);
    print_tensor(gpu_grad_input, "Sigmoid Gradient");

    // Verify CPU and GPU results match
    Tensor gpu_output_cpu = gpu_output.cpu();
    Tensor gpu_grad_cpu = gpu_grad_input.cpu();

    bool forward_match = true, backward_match = true;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        if (std::abs(cpu_output.data()[i] - gpu_output_cpu.data()[i]) > 1e-6) {
            forward_match = false;
        }
        if (std::abs(cpu_grad_input.data()[i] - gpu_grad_cpu.data()[i]) > 1e-6) {
            backward_match = false;
        }
    }

    std::cout << "CPU-GPU Forward Match: " << (forward_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "CPU-GPU Backward Match: " << (backward_match ? "PASS" : "FAIL") << std::endl;
}

void test_pointer_interface() {
    std::cout << "\n=== Testing Pointer Interface ===\n";

    const size_t size = 6;
    float input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float output[size];
    float grad_output[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float grad_input[size];

    // Test ReLU pointer interface
    std::cout << "\n--- ReLU Pointer Interface ---\n";
    ReLU::forward(input, output, size);
    std::cout << "ReLU Forward: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output[i] << " ";
    }
    std::cout << std::endl;

    ReLU::backward(grad_output, input, grad_input, size);
    std::cout << "ReLU Backward: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setprecision(4) << grad_input[i] << " ";
    }
    std::cout << std::endl;

    // Test Sigmoid pointer interface
    std::cout << "\n--- Sigmoid Pointer Interface ---\n";
    Sigmoid::forward(input, output, size);
    std::cout << "Sigmoid Forward: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output[i] << " ";
    }
    std::cout << std::endl;

    Sigmoid::backward(grad_output, output, grad_input, size);
    std::cout << "Sigmoid Backward: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setprecision(4) << grad_input[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        std::cout << "Testing Activation Functions Implementation\n";
        std::cout << "==========================================\n";

        test_relu();
        test_sigmoid();
        test_pointer_interface();

        std::cout << "\n=== All Tests Completed ===\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}