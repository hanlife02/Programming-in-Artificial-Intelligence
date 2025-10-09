#pragma once

#include "Tensor.h"
#include <cmath>

namespace ActivationFunctions {

// ReLU
class ReLU {
public:
    // Forward computation: ReLU(x) = max(x, 0)
    static Tensor forward(const Tensor& input);
    static void forward(float* input, float* output, size_t size);

    // Backward computation: dL/dx = dL/dy if x > 0, else 0
    static Tensor backward(const Tensor& grad_output, const Tensor& input);
    static void backward(const float* grad_output, const float* input, float* grad_input, size_t size);
};

// Sigmoid
class Sigmoid {
public:
    // Forward computation: sigmoid(x) = 1 / (1 + exp(-x))
    static Tensor forward(const Tensor& input);
    static void forward(float* input, float* output, size_t size);

    // Backward computation: dL/dx = dL/dy * y * (1 - y)
    static Tensor backward(const Tensor& grad_output, const Tensor& output);
    static void backward(const float* grad_output, const float* output, float* grad_input, size_t size);
};

}