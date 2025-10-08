#include <vector>
#include <memory>
#include <cstddef>
#include <random>

// Define device enumeration
enum class Device {
    kCPU,
    kGPU
};

class Tensor {
public:
    // Constructor: allocate memory based on shape and device
    Tensor(const std::vector<size_t>& shape, Device device = Device::kCPU);

    // Default destructor
    ~Tensor() = default;

    // Copy constructor: deep copy data
    Tensor(const Tensor& other);

    // Move constructor
    Tensor(Tensor&& other) noexcept;

    // Copy assignment operator
    Tensor& operator=(const Tensor& other);

    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept;

    // Move data to GPU
    Tensor gpu() const;

    // Move data to CPU
    Tensor cpu() const;

    // --- Initialization functions ---

    // Initialize with zeros
    void zeros();

    // Initialize with ones
    void ones();

    // Random initialization
    void random(float min = 0.0f, float max = 1.0f);

    // --- Utility functions ---

    // Get device type
    Device device() const { return device_; }

    // Get shape
    const std::vector<size_t>& shape() const { return shape_; }

    // Get total number of elements
    size_t numel() const { return num_elements_; }

    // Get raw data pointer (for CUDA kernels or library calls)
    float* data() const { return data_ptr_.get(); }

    // Check if tensor is empty
    bool empty() const { return num_elements_ == 0; }

private:
    std::shared_ptr<float> data_ptr_;
    std::vector<size_t> shape_;
    size_t num_elements_;
    Device device_;
};