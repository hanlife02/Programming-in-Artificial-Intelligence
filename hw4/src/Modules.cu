#include "Modules.h"
#include "ActivationFunctions.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

namespace {

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* context) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error in ") + context);
    }
}

struct CublasHandle {
    cublasHandle_t handle{};
    CublasHandle() { check_cublas(cublasCreate(&handle), "cublasCreate"); }
    ~CublasHandle() { cublasDestroy(handle); }
};

std::vector<int> expect_shape(const Tensor& tensor, size_t dims, const char* name) {
    if (tensor.shape().size() != dims) {
        throw std::invalid_argument(std::string(name) + " must have " + std::to_string(dims) + " dimensions");
    }
    return tensor.shape();
}

void ensure_device(Device device, const Tensor& a, const Tensor* b = nullptr, const Tensor* c = nullptr) {
    if (a.device() != device) {
        throw std::invalid_argument("Tensor device mismatch for operation");
    }
    if (b && b->device() != device) {
        throw std::invalid_argument("Tensor device mismatch for operation");
    }
    if (c && c->device() != device) {
        throw std::invalid_argument("Tensor device mismatch for operation");
    }
}

Tensor clone_tensor(const Tensor& tensor) {
    Tensor copy(tensor.shape(), tensor.device());
    if (tensor.device() == Device::CPU) {
        std::memcpy(copy.data(), tensor.data(), tensor.size() * sizeof(float));
    } else {
        check_cuda(cudaMemcpy(copy.data(), tensor.data(), tensor.size() * sizeof(float), cudaMemcpyDeviceToDevice),
                   "clone_tensor");
    }
    return copy;
}

// CPU helpers
Tensor fully_connected_forward_cpu(const Tensor& input, const Tensor& weights, const Tensor* bias) {
    auto in_shape = expect_shape(input, 2, "input");
    auto w_shape = expect_shape(weights, 2, "weights");
    int batch = in_shape[0];
    int in_features = in_shape[1];
    int out_features = w_shape[0];

    if (w_shape[1] != in_features) {
        throw std::invalid_argument("Weights second dimension must match input features");
    }
    if (bias) {
        auto b_shape = expect_shape(*bias, 1, "bias");
        if (b_shape[0] != out_features) {
            throw std::invalid_argument("Bias size must match output features");
        }
    }

    Tensor output({batch, out_features}, Device::CPU);
    const float* input_ptr = input.data();
    const float* weight_ptr = weights.data();
    const float* bias_ptr = bias ? bias->data() : nullptr;
    float* out_ptr = output.data();
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_features; ++o) {
            float sum = 0.f;
            for (int i = 0; i < in_features; ++i) {
                sum += input_ptr[b * in_features + i] * weight_ptr[o * in_features + i];
            }
            if (bias_ptr) {
                sum += bias_ptr[o];
            }
            out_ptr[b * out_features + o] = sum;
        }
    }
    return output;
}

std::tuple<Tensor, Tensor, Tensor> fully_connected_backward_cpu(const Tensor& input,
                                                                const Tensor& grad_output,
                                                                const Tensor& weights) {
    auto in_shape = expect_shape(input, 2, "input");
    auto grad_shape = expect_shape(grad_output, 2, "grad_output");
    auto w_shape = expect_shape(weights, 2, "weights");

    int batch = in_shape[0];
    int in_features = in_shape[1];
    int out_features = grad_shape[1];
    if (grad_shape[0] != batch) {
        throw std::invalid_argument("grad_output batch mismatch");
    }
    if (w_shape[0] != out_features || w_shape[1] != in_features) {
        throw std::invalid_argument("weights shape mismatch");
    }

    Tensor grad_input({batch, in_features}, Device::CPU);
    Tensor grad_weights({out_features, in_features}, Device::CPU);
    Tensor grad_bias({out_features}, Device::CPU);

    const float* input_ptr = input.data();
    const float* grad_ptr = grad_output.data();
    const float* weight_ptr = weights.data();
    float* grad_in_ptr = grad_input.data();
    float* grad_w_ptr = grad_weights.data();
    float* grad_b_ptr = grad_bias.data();

    std::fill(grad_in_ptr, grad_in_ptr + grad_input.size(), 0.f);
    std::fill(grad_w_ptr, grad_w_ptr + grad_weights.size(), 0.f);
    std::fill(grad_b_ptr, grad_b_ptr + grad_bias.size(), 0.f);

    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_features; ++o) {
            float go = grad_ptr[b * out_features + o];
            grad_b_ptr[o] += go;
            for (int i = 0; i < in_features; ++i) {
                grad_in_ptr[b * in_features + i] += go * weight_ptr[o * in_features + i];
                grad_w_ptr[o * in_features + i] += go * input_ptr[b * in_features + i];
            }
        }
    }

    return {grad_input, grad_weights, grad_bias};
}

}  // namespace

Tensor fully_connected_forward(const Tensor& input,
                               const Tensor& weights,
                               const Tensor* bias) {
    if (input.device() == Device::CPU) {
        return fully_connected_forward_cpu(input, weights, bias);
    }

    ensure_device(Device::GPU, input, &weights, bias);
    auto in_shape = expect_shape(input, 2, "input");
    auto w_shape = expect_shape(weights, 2, "weights");
    int batch_size = in_shape[0];
    int in_features = in_shape[1];
    int out_features = w_shape[0];
    if (w_shape[1] != in_features) {
        throw std::invalid_argument("Weights second dimension must match input features");
    }

    Tensor output({batch_size, out_features}, Device::GPU);
    CublasHandle handle;
    const float alpha = 1.0f, beta = 0.0f;

    check_cublas(
        cublasSgemm(handle.handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    out_features, batch_size, in_features,
                    &alpha,
                    weights.data(), in_features,
                    input.data(), in_features,
                    &beta,
                    output.data(), out_features),
        "fully_connected_forward");

    if (bias) {
        auto b_shape = expect_shape(*bias, 1, "bias");
        if (b_shape[0] != out_features) {
            throw std::invalid_argument("Bias size must match output features");
        }
        float* ones = nullptr;
        check_cuda(cudaMalloc(&ones, batch_size * sizeof(float)), "malloc ones");
        thrust::device_ptr<float> ones_ptr(ones);
        thrust::fill(ones_ptr, ones_ptr + batch_size, 1.0f);
        const float alpha_bias = 1.0f;
        const float beta_output = 1.0f;
        check_cublas(
            cublasSgemm(handle.handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        out_features, batch_size, 1,
                        &alpha_bias,
                        bias->data(), out_features,
                        ones, 1,
                        &beta_output,
                        output.data(), out_features),
            "fc bias add");
        cudaFree(ones);
    }

    return output;
}

std::tuple<Tensor, Tensor, Tensor> fully_connected_backward(const Tensor& input,
                                                            const Tensor& grad_output,
                                                            const Tensor& weights) {
    if (input.device() == Device::CPU) {
        return fully_connected_backward_cpu(input, grad_output, weights);
    }

    ensure_device(Device::GPU, input, &grad_output, &weights);
    auto in_shape = expect_shape(input, 2, "input");
    auto grad_shape = expect_shape(grad_output, 2, "grad_output");
    auto w_shape = expect_shape(weights, 2, "weights");

    int batch_size = in_shape[0];
    int in_features = in_shape[1];
    int out_features = grad_shape[1];
    if (grad_shape[0] != batch_size) {
        throw std::invalid_argument("grad_output batch mismatch");
    }
    if (w_shape[0] != out_features || w_shape[1] != in_features) {
        throw std::invalid_argument("weights shape mismatch");
    }

    Tensor grad_input({batch_size, in_features}, Device::GPU);
    Tensor grad_weights({out_features, in_features}, Device::GPU);
    Tensor grad_bias({out_features}, Device::GPU);

    CublasHandle handle;
    const float alpha = 1.0f, beta_zero = 0.0f;

    check_cublas(
        cublasSgemm(handle.handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    in_features, batch_size, out_features,
                    &alpha,
                    weights.data(), in_features,
                    grad_output.data(), out_features,
                    &beta_zero,
                    grad_input.data(), in_features),
        "fc grad input");

    check_cublas(
        cublasSgemm(handle.handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    in_features, out_features, batch_size,
                    &alpha,
                    input.data(), in_features,
                    grad_output.data(), out_features,
                    &beta_zero,
                    grad_weights.data(), in_features),
        "fc grad weights");

    float* ones = nullptr;
    check_cuda(cudaMalloc(&ones, batch_size * sizeof(float)), "malloc ones backward");
    thrust::device_ptr<float> ones_ptr(ones);
    thrust::fill(ones_ptr, ones_ptr + batch_size, 1.0f);
    check_cublas(
        cublasSgemv(handle.handle,
                    CUBLAS_OP_N,
                    out_features, batch_size,
                    &alpha,
                    grad_output.data(), out_features,
                    ones, 1,
                    &beta_zero,
                    grad_bias.data(), 1),
        "fc grad bias");
    cudaFree(ones);

    return {grad_input, grad_weights, grad_bias};
}

namespace {

__global__ void im2col_kernel(const int n, const float* data_im,
                              int batch, int channels, int height, int width,
                              int kernel_h, int kernel_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int height_col, int width_col,
                              float* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int k_w = (index / (width_col * height_col)) % kernel_w;
        int k_h = (index / (width_col * height_col * kernel_w)) % kernel_h;
        int channel = (index / (width_col * height_col * kernel_w * kernel_h)) % channels;
        int batch_idx = index / (width_col * height_col * kernel_w * kernel_h * channels);

        int h_in = h_out * stride_h - pad_h + k_h;
        int w_in = w_out * stride_w - pad_w + k_w;

        int data_col_index = index;
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            int data_im_index = (((batch_idx * channels + channel) * height + h_in) * width) + w_in;
            data_col[data_col_index] = data_im[data_im_index];
        } else {
            data_col[data_col_index] = 0.f;
        }
    }
}

__global__ void col2im_kernel(const int n, const float* data_col,
                              int batch, int channels, int height, int width,
                              int kernel_h, int kernel_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int height_col, int width_col,
                              float* data_im) {
    CUDA_KERNEL_LOOP(index, n) {
        int w = index % width;
        int h = (index / width) % height;
        int channel = (index / (width * height)) % channels;
        int batch_idx = index / (channels * height * width);

        int c_offset = channel * kernel_h * kernel_w;
        float val = 0.f;
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            int h_col = h + pad_h - k_h;
            if (h_col < 0 || h_col % stride_h != 0) {
                continue;
            }
            h_col /= stride_h;
            if (h_col < 0 || h_col >= height_col) {
                continue;
            }
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int w_col = w + pad_w - k_w;
                if (w_col < 0 || w_col % stride_w != 0) {
                    continue;
                }
                w_col /= stride_w;
                if (w_col < 0 || w_col >= width_col) {
                    continue;
                }
                int col_index =
                    (((batch_idx * (channels * kernel_h * kernel_w) +
                       (c_offset + k_h * kernel_w + k_w)) *
                      height_col + h_col) *
                     width_col + w_col);
                val += data_col[col_index];
            }
        }
        data_im[index] = val;
    }
}

void im2col_gpu(const float* data_im, int batch, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int height_col, int width_col,
                float* data_col) {
    int channels_col = channels * kernel_h * kernel_w;
    int n = batch * channels_col * height_col * width_col;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    im2col_kernel<<<blocks, threads>>>(n, data_im, batch, channels, height, width,
                                       kernel_h, kernel_w, pad_h, pad_w,
                                       stride_h, stride_w, height_col, width_col,
                                       data_col);
}

void col2im_gpu(const float* data_col, int batch, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int height_col, int width_col,
                float* data_im) {
    int n = batch * channels * height * width;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    col2im_kernel<<<blocks, threads>>>(n, data_col, batch, channels, height, width,
                                       kernel_h, kernel_w, pad_h, pad_w,
                                       stride_h, stride_w, height_col, width_col,
                                       data_im);
}

__global__ void matrix_to_nchw_kernel(const float* matrix, float* tensor,
                                      int batch, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = height * width;
    int total = batch * channels * spatial;
    if (index >= total) {
        return;
    }
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int n = index / (channels * height * width);

    int col = n * spatial + h * width + w;
    tensor[index] = matrix[c * (batch * spatial) + col];
}

__global__ void nchw_to_matrix_kernel(const float* tensor, float* matrix,
                                      int batch, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = height * width;
    int total = batch * channels * spatial;
    if (index >= total) {
        return;
    }
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int n = index / (channels * height * width);
    int col = n * spatial + h * width + w;
    matrix[c * (batch * spatial) + col] = tensor[index];
}

void matrix_to_nchw(const float* matrix, float* tensor,
                    int batch, int channels, int height, int width) {
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    matrix_to_nchw_kernel<<<blocks, threads>>>(matrix, tensor, batch, channels, height, width);
}

void nchw_to_matrix(const float* tensor, float* matrix,
                    int batch, int channels, int height, int width) {
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    nchw_to_matrix_kernel<<<blocks, threads>>>(tensor, matrix, batch, channels, height, width);
}

Tensor conv2d_forward_cpu(const Tensor& input,
                          const Tensor& weights,
                          const Tensor* bias,
                          const Conv2DConfig& config) {
    auto in_shape = expect_shape(input, 4, "input");
    auto w_shape = expect_shape(weights, 4, "weights");
    int batch = in_shape[0];
    int in_channels = in_shape[1];
    int height = in_shape[2];
    int width = in_shape[3];
    int out_channels = w_shape[0];
    int kernel_h = config.kernel_h;
    int kernel_w = config.kernel_w;
    if (w_shape[1] != in_channels || w_shape[2] != kernel_h || w_shape[3] != kernel_w) {
        throw std::invalid_argument("Conv weights shape mismatch");
    }
    int out_h = (height + 2 * config.pad_h - kernel_h) / config.stride_h + 1;
    int out_w = (width + 2 * config.pad_w - kernel_w) / config.stride_w + 1;

    Tensor output({batch, out_channels, out_h, out_w}, Device::CPU);
    const float* in_ptr = input.data();
    const float* w_ptr = weights.data();
    const float* bias_ptr = (bias && config.include_bias) ? bias->data() : nullptr;
    float* out_ptr = output.data();

    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias_ptr ? bias_ptr[oc] : 0.f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * config.stride_h - config.pad_h + kh;
                                int iw = ow * config.stride_w - config.pad_w + kw;
                                if (ih < 0 || ih >= height || iw < 0 || iw >= width) {
                                    continue;
                                }
                                int input_idx = (((n * in_channels + ic) * height) + ih) * width + iw;
                                int weight_idx = (((oc * in_channels + ic) * kernel_h) + kh) * kernel_w + kw;
                                sum += in_ptr[input_idx] * w_ptr[weight_idx];
                            }
                        }
                    }
                    int out_idx = (((n * out_channels + oc) * out_h) + oh) * out_w + ow;
                    out_ptr[out_idx] = sum;
                }
            }
        }
    }
    return output;
}

std::tuple<Tensor, Tensor, Tensor> conv2d_backward_cpu(const Tensor& input,
                                                       const Tensor& grad_output,
                                                       const Tensor& weights,
                                                       const Conv2DConfig& config) {
    auto in_shape = expect_shape(input, 4, "input");
    auto grad_shape = expect_shape(grad_output, 4, "grad_output");
    auto w_shape = expect_shape(weights, 4, "weights");
    int batch = in_shape[0];
    int in_channels = in_shape[1];
    int height = in_shape[2];
    int width = in_shape[3];
    int out_channels = w_shape[0];
    int kernel_h = config.kernel_h;
    int kernel_w = config.kernel_w;
    int out_h = grad_shape[2];
    int out_w = grad_shape[3];

    Tensor grad_input({batch, in_channels, height, width}, Device::CPU);
    Tensor grad_weights({out_channels, in_channels, kernel_h, kernel_w}, Device::CPU);
    Tensor grad_bias({out_channels}, Device::CPU);

    std::fill(grad_input.data(), grad_input.data() + grad_input.size(), 0.f);
    std::fill(grad_weights.data(), grad_weights.data() + grad_weights.size(), 0.f);
    std::fill(grad_bias.data(), grad_bias.data() + grad_bias.size(), 0.f);

    const float* input_ptr = input.data();
    const float* grad_ptr = grad_output.data();
    const float* weight_ptr = weights.data();

    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float grad_val = grad_ptr[(((n * out_channels + oc) * out_h) + oh) * out_w + ow];
                    grad_bias.data()[oc] += grad_val;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * config.stride_h - config.pad_h + kh;
                                int iw = ow * config.stride_w - config.pad_w + kw;
                                if (ih < 0 || ih >= height || iw < 0 || iw >= width) {
                                    continue;
                                }
                                int input_idx = (((n * in_channels + ic) * height) + ih) * width + iw;
                                int weight_idx = (((oc * in_channels + ic) * kernel_h) + kh) * kernel_w + kw;
                                grad_weights.data()[weight_idx] += input_ptr[input_idx] * grad_val;
                                grad_input.data()[input_idx] += weight_ptr[weight_idx] * grad_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return {grad_input, grad_weights, grad_bias};
}

Tensor conv2d_forward_gpu(const Tensor& input,
                          const Tensor& weights,
                          const Tensor* bias,
                          const Conv2DConfig& config) {
    ensure_device(Device::GPU, input, &weights, bias);
    auto in_shape = expect_shape(input, 4, "input");
    auto w_shape = expect_shape(weights, 4, "weights");
    int batch = in_shape[0];
    int in_channels = in_shape[1];
    int height = in_shape[2];
    int width = in_shape[3];
    int out_channels = w_shape[0];
    int kernel_h = config.kernel_h;
    int kernel_w = config.kernel_w;
    if (w_shape[1] != in_channels || w_shape[2] != kernel_h || w_shape[3] != kernel_w) {
        throw std::invalid_argument("Conv weights shape mismatch");
    }

    int pad_h = config.pad_h;
    int pad_w = config.pad_w;
    int stride_h = config.stride_h;
    int stride_w = config.stride_w;
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int kernel_dim = in_channels * kernel_h * kernel_w;
    int n_cols = batch * height_col * width_col;

    Tensor output({batch, out_channels, height_col, width_col}, Device::GPU);

    float* col_buffer = nullptr;
    check_cuda(cudaMalloc(&col_buffer, kernel_dim * n_cols * sizeof(float)), "conv col buffer");
    im2col_gpu(input.data(), batch, in_channels, height, width,
               kernel_h, kernel_w, pad_h, pad_w,
               stride_h, stride_w, height_col, width_col,
               col_buffer);

    float* output_matrix = nullptr;
    check_cuda(cudaMalloc(&output_matrix, out_channels * n_cols * sizeof(float)), "conv output matrix");
    CublasHandle handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasSgemm(handle.handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n_cols, out_channels, kernel_dim,
                    &alpha,
                    col_buffer, n_cols,
                    weights.data(), kernel_dim,
                    &beta,
                    output_matrix, n_cols),
        "conv2d matmul");

    if (bias && config.include_bias) {
        float* ones = nullptr;
        check_cuda(cudaMalloc(&ones, n_cols * sizeof(float)), "conv ones");
        thrust::device_ptr<float> ones_ptr(ones);
        thrust::fill(ones_ptr, ones_ptr + n_cols, 1.0f);
        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f;
        check_cublas(
            cublasSgemm(handle.handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n_cols, out_channels, 1,
                        &alpha_bias,
                        ones, n_cols,
                        bias->data(), 1,
                        &beta_bias,
                        output_matrix, n_cols),
            "conv bias add");
        cudaFree(ones);
    }

    matrix_to_nchw(output_matrix, output.data(), batch, out_channels, height_col, width_col);

    cudaFree(output_matrix);
    cudaFree(col_buffer);
    return output;
}

std::tuple<Tensor, Tensor, Tensor> conv2d_backward_gpu(const Tensor& input,
                                                       const Tensor& grad_output,
                                                       const Tensor& weights,
                                                       const Conv2DConfig& config) {
    ensure_device(Device::GPU, input, &grad_output, &weights);
    auto in_shape = expect_shape(input, 4, "input");
    auto grad_shape = expect_shape(grad_output, 4, "grad_output");
    auto w_shape = expect_shape(weights, 4, "weights");
    int batch = in_shape[0];
    int in_channels = in_shape[1];
    int height = in_shape[2];
    int width = in_shape[3];
    int out_channels = w_shape[0];
    int kernel_h = config.kernel_h;
    int kernel_w = config.kernel_w;

    int pad_h = config.pad_h;
    int pad_w = config.pad_w;
    int stride_h = config.stride_h;
    int stride_w = config.stride_w;

    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int kernel_dim = in_channels * kernel_h * kernel_w;
    int n_cols = batch * height_col * width_col;

    Tensor grad_input({batch, in_channels, height, width}, Device::GPU);
    Tensor grad_weights({out_channels, in_channels, kernel_h, kernel_w}, Device::GPU);
    Tensor grad_bias({out_channels}, Device::GPU);

    float* col_buffer = nullptr;
    check_cuda(cudaMalloc(&col_buffer, kernel_dim * n_cols * sizeof(float)), "conv backward col buffer");
    im2col_gpu(input.data(), batch, in_channels, height, width,
               kernel_h, kernel_w, pad_h, pad_w,
               stride_h, stride_w, height_col, width_col,
               col_buffer);

    float* grad_output_matrix = nullptr;
    check_cuda(cudaMalloc(&grad_output_matrix, out_channels * n_cols * sizeof(float)), "grad output matrix");
    nchw_to_matrix(grad_output.data(), grad_output_matrix, batch, out_channels, height_col, width_col);

    CublasHandle handle;
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;

    check_cublas(
        cublasSgemm(handle.handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    kernel_dim, out_channels, n_cols,
                    &alpha,
                    col_buffer, n_cols,
                    grad_output_matrix, n_cols,
                    &beta_zero,
                    grad_weights.data(), kernel_dim),
        "conv grad weights");

    float* ones = nullptr;
    check_cuda(cudaMalloc(&ones, n_cols * sizeof(float)), "conv grad bias ones");
    thrust::device_ptr<float> ones_ptr(ones);
    thrust::fill(ones_ptr, ones_ptr + n_cols, 1.0f);

    check_cublas(
        cublasSgemv(handle.handle,
                    CUBLAS_OP_T,
                    n_cols, out_channels,
                    &alpha,
                    grad_output_matrix, n_cols,
                    ones, 1,
                    &beta_zero,
                    grad_bias.data(), 1),
        "conv grad bias");
    cudaFree(ones);

    float* grad_col = nullptr;
    check_cuda(cudaMalloc(&grad_col, kernel_dim * n_cols * sizeof(float)), "grad col buffer");
    check_cublas(
        cublasSgemm(handle.handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    n_cols, kernel_dim, out_channels,
                    &alpha,
                    grad_output_matrix, n_cols,
                    weights.data(), kernel_dim,
                    &beta_zero,
                    grad_col, n_cols),
        "conv grad col");

    col2im_gpu(grad_col, batch, in_channels, height, width,
               kernel_h, kernel_w, pad_h, pad_w,
               stride_h, stride_w, height_col, width_col,
               grad_input.data());

    cudaFree(grad_col);
    cudaFree(grad_output_matrix);
    cudaFree(col_buffer);
    return {grad_input, grad_weights, grad_bias};
}

}  // namespace

Tensor conv2d_forward(const Tensor& input,
                      const Tensor& weights,
                      const Tensor* bias,
                      const Conv2DConfig& config) {
    if (input.device() == Device::CPU) {
        return conv2d_forward_cpu(input, weights, bias, config);
    }
    return conv2d_forward_gpu(input, weights, bias, config);
}

std::tuple<Tensor, Tensor, Tensor> conv2d_backward(const Tensor& input,
                                                   const Tensor& grad_output,
                                                   const Tensor& weights,
                                                   const Conv2DConfig& config) {
    if (input.device() == Device::CPU) {
        return conv2d_backward_cpu(input, grad_output, weights, config);
    }
    return conv2d_backward_gpu(input, grad_output, weights, config);
}

namespace {

__global__ void max_pool_forward_kernel(const float* in_data, float* out_data, float* out_mask,
                                        int nthreads, int num, int channels,
                                        int in_h, int in_w, int out_h, int out_w,
                                        int kernel_h, int kernel_w, int stride_h, int stride_w) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / (channels * out_h * out_w);
        int c = (index / (out_h * out_w)) % channels;
        int ph = (index / out_w) % out_h;
        int pw = index % out_w;

        int hstart = ph * stride_h;
        int wstart = pw * stride_w;

        float max_val = -FLT_MAX;
        int max_idx = -1;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = hstart + kh;
                int w = wstart + kw;
                if (h >= in_h || w >= in_w) {
                    continue;
                }
                int in_index = ((n * channels + c) * in_h + h) * in_w + w;
                float val = in_data[in_index];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_index;
                }
            }
        }

        out_data[index] = max_val;
        if (out_mask != nullptr) {
            out_mask[index] = static_cast<float>(max_idx);
        }
    }
}

__global__ void max_pool_backward_kernel(const float* grad_out, const float* mask,
                                         float* grad_in, int nthreads) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int in_index = static_cast<int>(mask[index]);
        if (in_index >= 0) {
            atomicAdd(&grad_in[in_index], grad_out[index]);
        }
    }
}

PoolForwardResult max_pool_forward_cpu_impl(const Tensor& input, const Pool2DConfig& config) {
    auto in_shape = expect_shape(input, 4, "input");
    int num = in_shape[0];
    int channels = in_shape[1];
    int height = in_shape[2];
    int width = in_shape[3];
    int out_h = (height - config.kernel_h) / config.stride_h + 1;
    int out_w = (width - config.kernel_w) / config.stride_w + 1;

    PoolForwardResult result{
        Tensor({num, channels, out_h, out_w}, Device::CPU),
        Tensor({num, channels, out_h, out_w}, Device::CPU)
    };

    const float* in_ptr = input.data();
    float* out_ptr = result.output.data();
    float* mask_ptr = result.mask.data();

    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < out_h; ++ph) {
                for (int pw = 0; pw < out_w; ++pw) {
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int kh = 0; kh < config.kernel_h; ++kh) {
                        for (int kw = 0; kw < config.kernel_w; ++kw) {
                            int h = ph * config.stride_h + kh;
                            int w = pw * config.stride_w + kw;
                            if (h >= height || w >= width) continue;
                            int in_index = ((n * channels + c) * height + h) * width + w;
                            float val = in_ptr[in_index];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = in_index;
                            }
                        }
                    }
                    int out_index = ((n * channels + c) * out_h + ph) * out_w + pw;
                    out_ptr[out_index] = max_val;
                    mask_ptr[out_index] = static_cast<float>(max_idx);
                }
            }
        }
    }
    return result;
}

Tensor max_pool_backward_cpu_impl(const Tensor& grad_output,
                                  const Tensor& mask,
                                  const Pool2DConfig& config,
                                  const std::vector<int>& input_shape) {
    int num = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];
    Tensor grad_input(input_shape, Device::CPU);
    std::fill(grad_input.data(), grad_input.data() + grad_input.size(), 0.f);

    const float* grad_ptr = grad_output.data();
    const float* mask_ptr = mask.data();
    int total = grad_output.size();
    float* grad_in_ptr = grad_input.data();

    for (int idx = 0; idx < total; ++idx) {
        int in_index = static_cast<int>(mask_ptr[idx]);
        if (in_index >= 0 && in_index < grad_input.size()) {
            grad_in_ptr[in_index] += grad_ptr[idx];
        }
    }
    return grad_input;
}

}  // namespace

PoolForwardResult max_pool_forward(const Tensor& input, const Pool2DConfig& config) {
    if (input.device() == Device::CPU) {
        return max_pool_forward_cpu_impl(input, config);
    }

    auto in_shape = expect_shape(input, 4, "input");
    int num = in_shape[0];
    int channels = in_shape[1];
    int height = in_shape[2];
    int width = in_shape[3];
    int out_h = (height - config.kernel_h) / config.stride_h + 1;
    int out_w = (width - config.kernel_w) / config.stride_w + 1;

    PoolForwardResult result{
        Tensor({num, channels, out_h, out_w}, Device::GPU),
        Tensor({num, channels, out_h, out_w}, Device::GPU)
    };

    int nthreads = num * channels * out_h * out_w;
    int threads = 256;
    int blocks = (nthreads + threads - 1) / threads;
    max_pool_forward_kernel<<<blocks, threads>>>(
        input.data(), result.output.data(), result.mask.data(),
        nthreads, num, channels, height, width, out_h, out_w,
        config.kernel_h, config.kernel_w, config.stride_h, config.stride_w);
    cudaDeviceSynchronize();

    return result;
}

Tensor max_pool_backward(const Tensor& grad_output,
                         const Tensor& mask,
                         const Pool2DConfig& config,
                         const std::vector<int>& input_shape) {
    if (grad_output.device() != mask.device()) {
        throw std::invalid_argument("grad_output and mask must be on the same device");
    }
    if (grad_output.device() == Device::CPU) {
        return max_pool_backward_cpu_impl(grad_output, mask, config, input_shape);
    }

    Tensor grad_input(input_shape, Device::GPU);
    check_cuda(cudaMemset(grad_input.data(), 0, grad_input.size() * sizeof(float)), "pool grad memset");
    int nthreads = grad_output.size();
    int threads = 256;
    int blocks = (nthreads + threads - 1) / threads;
    max_pool_backward_kernel<<<blocks, threads>>>(grad_output.data(), mask.data(), grad_input.data(), nthreads);
    cudaDeviceSynchronize();
    return grad_input;
}

namespace {

__global__ void row_max_kernel(const float* data, float* row_max, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    float max_val = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x) {
        float val = data[row * cols + col];
        if (val > max_val) {
            max_val = val;
        }
    }
    shared[tid] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        row_max[row] = shared[0];
    }
}

__global__ void subtract_max_kernel(const float* input, const float* row_max, float* output,
                                    int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (index >= total) {
        return;
    }
    int row = index / cols;
    output[index] = input[index] - row_max[row];
}

__global__ void exp_kernel(float* data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] = expf(data[idx]);
    }
}

__global__ void row_sum_kernel(const float* data, float* row_sum, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    float sum = 0.f;
    for (int col = tid; col < cols; col += blockDim.x) {
        sum += data[row * cols + col];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        row_sum[row] = shared[0];
    }
}

__global__ void normalize_kernel(float* data, const float* row_sum, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (index >= total) {
        return;
    }
    int row = index / cols;
    float denom = row_sum[row];
    data[index] = (denom > 0.f) ? (data[index] / denom) : 0.f;
}

__global__ void softmax_row_dot_kernel(const float* grad_output, const float* softmax_output,
                                       float* row_dot, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    float sum = 0.f;
    for (int col = tid; col < cols; col += blockDim.x) {
        sum += grad_output[row * cols + col] * softmax_output[row * cols + col];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        row_dot[row] = shared[0];
    }
}

__global__ void softmax_backward_kernel(const float* grad_output, const float* softmax_output,
                                        const float* row_dot, float* grad_input,
                                        int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (index >= total) {
        return;
    }
    int row = index / cols;
    grad_input[index] = softmax_output[index] * (grad_output[index] - row_dot[row]);
}

__global__ void cross_entropy_loss_kernel(const float* probs, const int* labels,
                                          float* loss_buffer, int batch_size, int num_classes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size) {
        return;
    }
    int label = labels[index];
    float prob = probs[index * num_classes + label];
    prob = fmaxf(prob, 1e-12f);
    loss_buffer[index] = -logf(prob);
}

__global__ void softmax_cross_entropy_backward_kernel(const float* probs, const int* labels,
                                                      float* grad_input, int batch_size, int num_classes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;
    if (index >= total) {
        return;
    }
    int row = index / num_classes;
    int col = index % num_classes;
    int label = labels[row];
    float grad = probs[index];
    if (col == label) {
        grad -= 1.f;
    }
    grad_input[index] = grad / batch_size;
}

Tensor softmax_forward_cpu_impl(const Tensor& input) {
    auto in_shape = expect_shape(input, 2, "input");
    int batch = in_shape[0];
    int classes = in_shape[1];
    Tensor output(in_shape, Device::CPU);
    const float* in_ptr = input.data();
    float* out_ptr = output.data();
    for (int b = 0; b < batch; ++b) {
        float max_val = -FLT_MAX;
        for (int c = 0; c < classes; ++c) {
            max_val = std::max(max_val, in_ptr[b * classes + c]);
        }
        float sum = 0.f;
        for (int c = 0; c < classes; ++c) {
            float val = std::exp(in_ptr[b * classes + c] - max_val);
            out_ptr[b * classes + c] = val;
            sum += val;
        }
        for (int c = 0; c < classes; ++c) {
            out_ptr[b * classes + c] /= sum;
        }
    }
    return output;
}

Tensor softmax_backward_cpu_impl(const Tensor& grad_output, const Tensor& softmax_output) {
    auto shape = expect_shape(grad_output, 2, "grad_output");
    Tensor grad_input(shape, Device::CPU);
    int batch = shape[0];
    int classes = shape[1];
    const float* grad_ptr = grad_output.data();
    const float* soft_ptr = softmax_output.data();
    float* grad_in_ptr = grad_input.data();
    for (int b = 0; b < batch; ++b) {
        float dot = 0.f;
        for (int c = 0; c < classes; ++c) {
            dot += grad_ptr[b * classes + c] * soft_ptr[b * classes + c];
        }
        for (int c = 0; c < classes; ++c) {
            grad_in_ptr[b * classes + c] = soft_ptr[b * classes + c] * (grad_ptr[b * classes + c] - dot);
        }
    }
    return grad_input;
}

float cross_entropy_loss_forward_cpu_impl(const Tensor& probs, const std::vector<int>& labels) {
    auto shape = expect_shape(probs, 2, "probs");
    int batch = shape[0];
    int classes = shape[1];
    if (static_cast<int>(labels.size()) != batch) {
        throw std::invalid_argument("Label count must match batch size");
    }
    const float* ptr = probs.data();
    float loss = 0.f;
    for (int b = 0; b < batch; ++b) {
        int label = labels[b];
        if (label < 0 || label >= classes) {
            throw std::invalid_argument("Label out of bounds");
        }
        float prob = std::max(ptr[b * classes + label], 1e-12f);
        loss += -std::log(prob);
    }
    return loss / batch;
}

Tensor cross_entropy_loss_backward_cpu_impl(const Tensor& probs, const std::vector<int>& labels) {
    auto shape = expect_shape(probs, 2, "probs");
    int batch = shape[0];
    int classes = shape[1];
    if (static_cast<int>(labels.size()) != batch) {
        throw std::invalid_argument("Label count must match batch size");
    }
    Tensor grad(shape, Device::CPU);
    float* grad_ptr = grad.data();
    std::memcpy(grad_ptr, probs.data(), probs.size() * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        int label = labels[b];
        if (label < 0 || label >= classes) {
            throw std::invalid_argument("Label out of bounds");
        }
        grad_ptr[b * classes + label] -= 1.f;
    }
    float scale = 1.f / batch;
    for (size_t i = 0; i < grad.size(); ++i) {
        grad_ptr[i] *= scale;
    }
    return grad;
}

}  // namespace

Tensor softmax_forward(const Tensor& input) {
    if (input.device() == Device::CPU) {
        return softmax_forward_cpu_impl(input);
    }

    auto shape = expect_shape(input, 2, "input");
    int batch = shape[0];
    int classes = shape[1];
    Tensor output(shape, Device::GPU);
    float* d_row_max = nullptr;
    float* d_row_sum = nullptr;
    check_cuda(cudaMalloc(&d_row_max, batch * sizeof(float)), "softmax row max");
    check_cuda(cudaMalloc(&d_row_sum, batch * sizeof(float)), "softmax row sum");

    int threads_per_block = 256;
    size_t shared_mem = threads_per_block * sizeof(float);
    row_max_kernel<<<batch, threads_per_block, shared_mem>>>(input.data(), d_row_max, batch, classes);

    int total = batch * classes;
    int blocks = (total + threads_per_block - 1) / threads_per_block;
    subtract_max_kernel<<<blocks, threads_per_block>>>(input.data(), d_row_max, output.data(), batch, classes);
    exp_kernel<<<blocks, threads_per_block>>>(output.data(), total);
    row_sum_kernel<<<batch, threads_per_block, shared_mem>>>(output.data(), d_row_sum, batch, classes);
    normalize_kernel<<<blocks, threads_per_block>>>(output.data(), d_row_sum, batch, classes);

    cudaFree(d_row_max);
    cudaFree(d_row_sum);
    return output;
}

Tensor softmax_backward(const Tensor& grad_output, const Tensor& softmax_output) {
    if (grad_output.device() != softmax_output.device()) {
        throw std::invalid_argument("grad_output and softmax_output must be on same device");
    }
    if (grad_output.device() == Device::CPU) {
        return softmax_backward_cpu_impl(grad_output, softmax_output);
    }

    auto shape = expect_shape(grad_output, 2, "grad_output");
    int batch = shape[0];
    int classes = shape[1];
    Tensor grad_input(shape, Device::GPU);

    float* row_dot = nullptr;
    check_cuda(cudaMalloc(&row_dot, batch * sizeof(float)), "softmax backward row dot");

    int threads_per_block = 256;
    size_t shared_mem = threads_per_block * sizeof(float);
    softmax_row_dot_kernel<<<batch, threads_per_block, shared_mem>>>(
        grad_output.data(), softmax_output.data(), row_dot, batch, classes);

    int total = batch * classes;
    int blocks = (total + threads_per_block - 1) / threads_per_block;
    softmax_backward_kernel<<<blocks, threads_per_block>>>(
        grad_output.data(), softmax_output.data(), row_dot, grad_input.data(), batch, classes);

    cudaFree(row_dot);
    return grad_input;
}

float cross_entropy_loss_forward(const Tensor& probs, const std::vector<int>& labels) {
    if (probs.device() == Device::CPU) {
        return cross_entropy_loss_forward_cpu_impl(probs, labels);
    }

    auto shape = expect_shape(probs, 2, "probs");
    int batch = shape[0];
    int classes = shape[1];
    if (static_cast<int>(labels.size()) != batch) {
        throw std::invalid_argument("Label count must match batch size");
    }
    Tensor temp_buffer({batch}, Device::GPU);

    int* d_labels = nullptr;
    check_cuda(cudaMalloc(&d_labels, batch * sizeof(int)), "cross entropy labels");
    check_cuda(cudaMemcpy(d_labels, labels.data(), batch * sizeof(int), cudaMemcpyHostToDevice),
               "copy labels");

    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    cross_entropy_loss_kernel<<<blocks, threads>>>(probs.data(), d_labels, temp_buffer.data(), batch, classes);

    std::vector<float> host_losses(batch, 0.f);
    check_cuda(cudaMemcpy(host_losses.data(), temp_buffer.data(), batch * sizeof(float), cudaMemcpyDeviceToHost),
               "copy losses");

    cudaFree(d_labels);
    float total_loss = 0.f;
    for (float val : host_losses) {
        total_loss += val;
    }
    return total_loss / batch;
}

Tensor cross_entropy_loss_backward(const Tensor& probs, const std::vector<int>& labels) {
    if (probs.device() == Device::CPU) {
        return cross_entropy_loss_backward_cpu_impl(probs, labels);
    }

    auto shape = expect_shape(probs, 2, "probs");
    int batch = shape[0];
    int classes = shape[1];
    if (static_cast<int>(labels.size()) != batch) {
        throw std::invalid_argument("Label count must match batch size");
    }

    Tensor grad(shape, Device::GPU);
    int total = batch * classes;
    check_cuda(cudaMemcpy(grad.data(), probs.data(), total * sizeof(float), cudaMemcpyDeviceToDevice),
               "grad copy");

    int* d_labels = nullptr;
    check_cuda(cudaMalloc(&d_labels, batch * sizeof(int)), "cross entropy back labels");
    check_cuda(cudaMemcpy(d_labels, labels.data(), batch * sizeof(int), cudaMemcpyHostToDevice),
               "copy labels back");

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softmax_cross_entropy_backward_kernel<<<blocks, threads>>>(
        probs.data(), d_labels, grad.data(), batch, classes);
    cudaFree(d_labels);
    return grad;
}
