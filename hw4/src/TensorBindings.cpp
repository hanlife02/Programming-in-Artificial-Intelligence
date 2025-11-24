#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Tensor.h"
#include <cuda_runtime.h>
#include <cstring>

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
            auto host = tensor_to_vector(tensor);
            py::array_t<float> array(host.size());
            if (!host.empty()) {
                std::memcpy(array.mutable_data(), host.data(), host.size() * sizeof(float));
            }
            return array;
        })
        .def("__repr__", [](const Tensor& tensor) {
            py::object py_shape = py::cast(tensor.shape());
            return "<Tensor shape=" + py::str(py_shape).cast<std::string>() +
                   ", device=" + device_to_string(tensor.device()) + ">";
        });
}
