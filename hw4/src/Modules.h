#pragma once

#include "Tensor.h"
#include <tuple>
#include <vector>

struct LinearConfig {
    bool include_bias = true;
};

struct Conv2DConfig {
    int kernel_h = 3;
    int kernel_w = 3;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    bool include_bias = true;
};

struct Pool2DConfig {
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;
};

struct PoolForwardResult {
    Tensor output;
    Tensor mask;
};

Tensor fully_connected_forward(const Tensor& input,
                               const Tensor& weights,
                               const Tensor* bias = nullptr);

std::tuple<Tensor, Tensor, Tensor> fully_connected_backward(const Tensor& input,
                                                            const Tensor& grad_output,
                                                            const Tensor& weights);

Tensor conv2d_forward(const Tensor& input,
                      const Tensor& weights,
                      const Tensor* bias,
                      const Conv2DConfig& config);

std::tuple<Tensor, Tensor, Tensor> conv2d_backward(const Tensor& input,
                                                   const Tensor& grad_output,
                                                   const Tensor& weights,
                                                   const Conv2DConfig& config);

PoolForwardResult max_pool_forward(const Tensor& input, const Pool2DConfig& config);

Tensor max_pool_backward(const Tensor& grad_output,
                         const Tensor& mask,
                         const Pool2DConfig& config,
                         const std::vector<int>& input_shape);

Tensor softmax_forward(const Tensor& input);

Tensor softmax_backward(const Tensor& grad_output, const Tensor& softmax_output);

float cross_entropy_loss_forward(const Tensor& probs, const std::vector<int>& labels);

Tensor cross_entropy_loss_backward(const Tensor& probs, const std::vector<int>& labels);
