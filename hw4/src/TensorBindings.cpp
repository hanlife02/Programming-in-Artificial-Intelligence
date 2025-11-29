#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Tensor.h"
#include "ActivationFunctions.h"
#include "Modules.h"
#include <cuda_runtime.h>
#include <cstring>
#include <optional>
#include <optional>

namespace py = pybind11;

namespace {

std::string device_to_string(Device device) {
    switch (device) {
        case Device::CPU:
            return "cpu";
        case Device::GPU:
            return "gpu";
        default:
            throw std::runtime_error("Unknown device type");
    }
}

Device parse_device(const std::string& device_str) {
    if (device_str == "cpu" || device_str == "CPU") {
        return Device::CPU;
    }
    if (device_str == "gpu" || device_str == "GPU") {
        return Device::GPU;
    }
    throw std::invalid_argument("Unsupported device type: " + device_str);
}

void copy_from_list(Tensor& tensor, const std::vector<float>& values) {
    if (values.size() != tensor.size()) {
        throw std::invalid_argument("Input data size does not match tensor size");
    }

    if (values.empty()) {
        return;
    }

    if (tensor.device() == Device::CPU) {
        std::memcpy(tensor.data(), values.data(), values.size() * sizeof(float));
    } else {
        cudaError_t error = cudaMemcpy(
            tensor.data(),
            values.data(),
            values.size() * sizeof(float),
            cudaMemcpyHostToDevice
        );
        if (error != cudaSuccess) {
            throw std::runtime_error(
                "Failed to copy data to GPU tensor: " + std::string(cudaGetErrorString(error))
            );
        }
    }
}

std::vector<float> tensor_to_vector(const Tensor& tensor) {
    std::vector<float> host_data(tensor.size(), 0.0f);
    if (tensor.size() == 0) {
        return host_data;
    }

    if (tensor.device() == Device::CPU) {
        std::memcpy(host_data.data(), tensor.data(), tensor.size() * sizeof(float));
    } else {
        cudaError_t error = cudaMemcpy(
            host_data.data(),
            tensor.data(),
            tensor.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        if (error != cudaSuccess) {
            throw std::runtime_error(
                "Failed to copy data from GPU tensor: " + std::string(cudaGetErrorString(error))
            );
        }
    }
    return host_data;
}

Tensor tensor_from_numpy_array(const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                               const std::string& device_str) {
    auto info = array.request();
    if (info.ndim == 0) {
        throw std::invalid_argument("Cannot create Tensor from scalar numpy array");
    }
    std::vector<int> shape(info.ndim);
    for (py::ssize_t i = 0; i < info.ndim; ++i) {
        shape[i] = static_cast<int>(info.shape[i]);
    }
    Device device = parse_device(device_str);
    Tensor tensor(shape, device);
    size_t bytes = info.size * sizeof(float);
    if (tensor.device() == Device::CPU) {
        std::memcpy(tensor.data(), info.ptr, bytes);
    } else {
        cudaError_t error = cudaMemcpy(tensor.data(), info.ptr, bytes, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error(
                "Failed to copy numpy data to GPU tensor: " + std::string(cudaGetErrorString(error))
            );
        }
    }
    return tensor;
}

py::array tensor_to_numpy_array(const Tensor& tensor) {
    auto host = tensor_to_vector(tensor);
    std::vector<py::ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
    std::vector<py::ssize_t> strides(shape.size(), 0);
    py::ssize_t stride = sizeof(float);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    py::array array(py::buffer_info(
        nullptr,
        sizeof(float),
        py::format_descriptor<float>::format(),
        shape.size(),
        shape,
        strides
    ));
    if (!host.empty()) {
        std::memcpy(array.mutable_data(), host.data(), host.size() * sizeof(float));
    }
    return array;
}

}  // namespace

PYBIND11_MODULE(mytensor, m) {
    m.doc() = "Python bindings for the custom Tensor class";

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](const std::vector<int>& shape, const std::string& device) {
            return Tensor(shape, parse_device(device));
        }), py::arg("shape"), py::arg("device") = "cpu")
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def_property_readonly("shape", [](const Tensor& tensor) {
            return tensor.shape();
        })
        .def_property_readonly("device", [](const Tensor& tensor) {
            return device_to_string(tensor.device());
        })
        .def_property_readonly("size", &Tensor::size)
        .def("set_data", [](Tensor& tensor, const std::vector<float>& values) {
            copy_from_list(tensor, values);
        })
        .def("to_list", [](const Tensor& tensor) {
            return tensor_to_vector(tensor);
        })
        .def("to_numpy", [](const Tensor& tensor) {
            return tensor_to_numpy_array(tensor);
        })
        .def_static("from_numpy", [](const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                                     const std::string& device) {
            return tensor_from_numpy_array(array, device);
        })
        .def("__repr__", [](const Tensor& tensor) {
            py::object py_shape = py::cast(tensor.shape());
            return "<Tensor shape=" + py::str(py_shape).cast<std::string>() +
                   ", device=" + device_to_string(tensor.device()) + ">";
        });

    m.def("from_numpy", [](const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                           const std::string& device) {
        return tensor_from_numpy_array(array, device);
    }, py::arg("array"), py::arg("device") = "cpu");

    m.def("relu_forward", [](const Tensor& input) {
        return ActivationFunctions::ReLU::forward(input);
    });

    m.def("relu_backward", [](const Tensor& grad_output, const Tensor& input) {
        return ActivationFunctions::ReLU::backward(grad_output, input);
    });

    m.def("sigmoid_forward", [](const Tensor& input) {
        return ActivationFunctions::Sigmoid::forward(input);
    });

    m.def("sigmoid_backward", [](const Tensor& grad_output, const Tensor& output) {
        return ActivationFunctions::Sigmoid::backward(grad_output, output);
    });

    m.def("linear_forward", [](const Tensor& input, const Tensor& weights, std::optional<Tensor> bias) {
        const Tensor* bias_ptr = bias ? &(*bias) : nullptr;
        return fully_connected_forward(input, weights, bias_ptr);
    }, py::arg("input"), py::arg("weights"), py::arg("bias") = std::nullopt);

    m.def("linear_backward", [](const Tensor& input, const Tensor& grad_output, const Tensor& weights) {
        return fully_connected_backward(input, grad_output, weights);
    });

    m.def("conv2d_forward",
          [](const Tensor& input, const Tensor& weights, std::optional<Tensor> bias,
             std::pair<int, int> stride, std::pair<int, int> padding) {
              auto w_shape = weights.shape();
              if (w_shape.size() != 4) {
                  throw std::invalid_argument("weights must be 4D (out_channels, in_channels, kH, kW)");
              }
              Conv2DConfig cfg;
              cfg.kernel_h = w_shape[2];
              cfg.kernel_w = w_shape[3];
              cfg.stride_h = stride.first;
              cfg.stride_w = stride.second;
              cfg.pad_h = padding.first;
              cfg.pad_w = padding.second;
              cfg.include_bias = bias.has_value();
              const Tensor* bias_ptr = bias ? &(*bias) : nullptr;
              return conv2d_forward(input, weights, bias_ptr, cfg);
          },
          py::arg("input"), py::arg("weights"), py::arg("bias") = std::nullopt,
          py::arg("stride") = std::make_pair(1, 1),
          py::arg("padding") = std::make_pair(1, 1));

    m.def("conv2d_backward",
          [](const Tensor& input, const Tensor& grad_output, const Tensor& weights,
             std::pair<int, int> stride, std::pair<int, int> padding) {
              auto w_shape = weights.shape();
              if (w_shape.size() != 4) {
                  throw std::invalid_argument("weights must be 4D (out_channels, in_channels, kH, kW)");
              }
              Conv2DConfig cfg;
              cfg.kernel_h = w_shape[2];
              cfg.kernel_w = w_shape[3];
              cfg.stride_h = stride.first;
              cfg.stride_w = stride.second;
              cfg.pad_h = padding.first;
              cfg.pad_w = padding.second;
              cfg.include_bias = true;
              return conv2d_backward(input, grad_output, weights, cfg);
          },
          py::arg("input"), py::arg("grad_output"), py::arg("weights"),
          py::arg("stride") = std::make_pair(1, 1),
          py::arg("padding") = std::make_pair(1, 1));

    m.def("max_pool_forward",
          [](const Tensor& input, std::pair<int, int> kernel, std::pair<int, int> stride) {
              Pool2DConfig cfg;
              cfg.kernel_h = kernel.first;
              cfg.kernel_w = kernel.second;
              cfg.stride_h = stride.first;
              cfg.stride_w = stride.second;
              auto result = max_pool_forward(input, cfg);
              return py::make_tuple(result.output, result.mask);
          },
          py::arg("input"),
          py::arg("kernel") = std::make_pair(2, 2),
          py::arg("stride") = std::make_pair(2, 2));

    m.def("max_pool_backward",
          [](const Tensor& grad_output, const Tensor& mask,
             const std::vector<int>& input_shape,
             std::pair<int, int> kernel, std::pair<int, int> stride) {
              Pool2DConfig cfg;
              cfg.kernel_h = kernel.first;
              cfg.kernel_w = kernel.second;
              cfg.stride_h = stride.first;
              cfg.stride_w = stride.second;
              return max_pool_backward(grad_output, mask, cfg, input_shape);
          },
          py::arg("grad_output"), py::arg("mask"), py::arg("input_shape"),
          py::arg("kernel") = std::make_pair(2, 2),
          py::arg("stride") = std::make_pair(2, 2));

    m.def("softmax_forward", [](const Tensor& input) {
        return softmax_forward(input);
    });

    m.def("softmax_backward", [](const Tensor& grad_output, const Tensor& softmax_output) {
        return softmax_backward(grad_output, softmax_output);
    });

    m.def("cross_entropy_loss", [](const Tensor& probs, const std::vector<int>& labels) {
        return cross_entropy_loss_forward(probs, labels);
    });

    m.def("cross_entropy_backward", [](const Tensor& probs, const std::vector<int>& labels) {
        return cross_entropy_loss_backward(probs, labels);
    });
}
