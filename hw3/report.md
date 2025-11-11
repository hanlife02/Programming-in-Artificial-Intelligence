# HW3 Report

## Fully Connected Layer
- **核心函数设计**：`forward_fc` 利用 `cublasSgemm` 将输入按 `(in_features, batch)` 与权重矩阵相乘，再通过一次额外 `cublasSgemm` 完成 bias broadcasting，从而完全复用高度优化的 GEMM。`backward_fc` 则分别用三次 `cublasSgemm / Sgemv` 计算 `∂X`、`∂W`、`∂b`，避免手写线程级逻辑。
- **正确性测试**：`test_forward_fc` 与 `test_backward_fc` 构造 2×3 输入和 4 输出神经元的微型网络， forward 期望输出为 `[4.5, 5.5, …, 6.5]`，backward 中对 `∂X` 直接比较 `0.4, 0.5, 0.70000005, 1.2, 1.3, 1.9000001` 等解析梯度；`∂W` 和 `∂b` 同样用解析值校验。
- **测试结果**：运行 `./hw3/build/main.exe` 时输出 `Test1 passed!`、`Test2 passed!`，表明前后向结果与解析解一致。

## Max Pooling Layer
- **核心函数设计**：`max_pool_forward_kernel` 以 `CUDA_KERNEL_LOOP` 遍历输出，每个线程扫描其 2×2 window 选出最大值并保存 argmax mask；`max_pool_backward_kernel` 使用 `atomicAdd` 将梯度 scatter 回输入。`max_pool_forward_layer/backward_layer` 负责配置网格并在 backward 里先 `cudaMemset`。
- **正确性测试**：`test_max_pool_forward`/`test_max_pool_backward` 分别对 1×1×4×4 输入验证输出与梯度，期望值手动计算（例如 forward 得到 `[6,8,14,16]`）。
- **测试结果**：程序输出 `MaxPool forward test passed!` 与 `MaxPool backward test passed!`。

## Softmax + Cross Entropy
- **核心函数设计**：`row_max_kernel`/`row_sum_kernel` 做行内归约，`subtract_max_kernel`、`exp_kernel`、`normalize_kernel` 负责数值稳定的减最大值、指数与归一；`cross_entropy_loss_kernel` 和 `softmax_cross_entropy_backward_kernel` 直接在 GPU 上计算 loss 与梯度，避免 host 参与。
- **正确性测试**：`test_softmax_forward` 与 `test_softmax_cross_entropy` 使用批大小 2、类别数 3 的固定 logits 和标签，期望概率、loss、梯度由公式求得并硬编码比较。
- **测试结果**：执行程序得到 `Softmax forward test passed!` 及两次 `Softmax + CrossEntropy test passed!`（一次来自 softmax-only 测试内的 loss 校验）。

## Conv2D (im2col + GEMM)
- **核心函数设计**：`im2col_kernel` 将输入 patch 展平，`conv2d_forward` 先 `im2col` 再 `cublasSgemm` 完成 `W * X_patch`，并通过一次 `ones` GEMM 加 bias。反向传播中，`conv2d_backward` 使用 `cublasSgemm` 计算 `∂W`、`∂b`，再用 `col2im_kernel` 将 `∂X_patch` 聚合回输入梯度，实现 `col2im`。
- **问题修复**：梯度回传阶段的 `∂X` 需要乘以 `W^T`，原先误将 `weights` 视作 `(out_channels, kernel_dim)` 参与乘法，导致 `Grad input mismatch at index 0: expected 0.8, got 0.6`。本次修复将 `cublasSgemm` 第二个操作数改为 `CUBLAS_OP_T`，保证 `grad_output_matrix` 与 `weights^T` 相乘，`col2im` 后即可得到正确的输入梯度。
- **正确性测试**：`test_conv_im2col` 构造 1×1×4×4 输入与 3×3 kernel，给出手工推导的 forward 输出以及 `∂X/∂W/∂b` 期望值，逐元素比较。
- **测试结果**：修复后再次运行程序，`Grad input mismatch` 不再出现，并重新看到 `Conv im2col test passed!`，证明 im2col+GEMM+col2im 全链路正确。

## 运行方式
```bash
nvcc hw3/main.cu -o hw3/build/main.exe -lcublas -std=c++17 -arch=sm_75 -Wno-deprecated-gpu-targets
./hw3/build/main.exe
```
终端输出的全部 `passed` 提示即为测试结果汇总。
