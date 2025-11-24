#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>

// Device type
enum class Device {
    CPU,
    GPU
};

class Tensor {
private:
    std::vector<int> shape_;          // Tensor shape
    Device device_;                   // Device type
    size_t total_size_;              // Total number of elements
    float* data_;                    // Data pointer, manually managed memory

    // Utility methods
    size_t calculate_total_size(const std::vector<int>& shape) const;  // Calculate total elements
    void allocate_memory();                                           // Allocate memory
    void deallocate_memory();                                         // Deallocate memory
    void copy_data_from(const Tensor& other);                        // Copy data from another tensor

public:
    // Constructors and destructor
    Tensor(const std::vector<int>& shape, Device device);
    Tensor(const Tensor& other);
    ~Tensor();

    // Assignment operator
    Tensor& operator=(const Tensor& other);

    // Migration methods
    Tensor cpu() const;
    Tensor gpu() const;

    // Access methods
    const std::vector<int>& shape() const { return shape_; }
    Device device() const { return device_; }
    size_t size() const { return total_size_; }
    float* data() const { return data_; }
};