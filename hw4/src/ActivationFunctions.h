#pragma once

#include "Tensor.h"
#include <cmath>

namespace ActivationFunctions {

class ReLU {
public:
    static Tensor forward(const Tensor& input);
    static void forward(float* input, float* output, size_t size);

    static Tensor backward(const Tensor& grad_output, const Tensor& input);
    static void backward(const float* grad_output, const float* input, float* grad_input, size_t size);
};

class Sigmoid {
public:
    static Tensor forward(const Tensor& input);
    static void forward(float* input, float* output, size_t size);

    static Tensor backward(const Tensor& grad_output, const Tensor& output);
    static void backward(const float* grad_output, const float* output, float* grad_input, size_t size);
};

}  // namespace ActivationFunctions
