# 构建说明 (Build Instructions)

本项目使用 CMake 构建系统，支持 CUDA 编程。

## 前置要求

- CMake 3.18 或更高版本
- CUDA Toolkit 10.0 或更高版本
- 支持 C++11 的编译器 (gcc/g++ 或 MSVC)
- 兼容的 NVIDIA GPU (可选，用于 GPU 测试)

## 快速开始

### 1. 创建构建目录
```bash
mkdir build
cd build
```

### 2. 配置项目
```bash
# Release 构建 (推荐)
cmake .. -DCMAKE_BUILD_TYPE=Release

# 或 Debug 构建 (用于调试)
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### 3. 编译项目
```bash
cmake --build .

# 或使用并行编译 (加速构建)
cmake --build . --parallel
```

### 4. 运行测试
```bash
# 运行所有测试
ctest

# 或使用自定义目标
cmake --build . --target run_all_tests

# 运行单个测试
cmake --build . --target run_tensor_tests
cmake --build . --target run_activation_tests

# 直接运行可执行文件
./test_tensor
./test_activations
```

## 详细构建选项

### 指定 CUDA 架构
如果您知道目标 GPU 的计算能力，可以指定特定架构：
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75"  # RTX 20xx 系列
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80"  # RTX 30xx 系列
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"  # RTX 40xx 系列
```

### Windows 平台
```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### 清理构建
```bash
# 删除构建文件
rm -rf build/*

# 或重新创建构建目录
cd ..
rm -rf build
mkdir build
cd build
```

## 项目结构

```
hw2/
├── CMakeLists.txt              # CMake 配置文件
├── BUILD.md                    # 构建说明 (本文件)
├── Tensor.h                    # Tensor 类头文件
├── Tensor.cu                   # Tensor 类实现
├── ActivationFunctions.h       # 激活函数头文件
├── ActivationFunctions.cu      # 激活函数实现
├── test_tensor.cu              # Tensor 测试程序
└── test_activations.cu         # 激活函数测试程序
```

## 生成的文件

构建完成后，在 `build/` 目录下会生成：
- `libTensor.a` - Tensor 静态库
- `libActivationFunctions.a` - 激活函数静态库
- `test_tensor` - Tensor 测试可执行文件
- `test_activations` - 激活函数测试可执行文件

## 常见问题

### 1. CUDA 未找到
```
CMake Error: Could not find CUDA
```
**解决方案**：确保安装了 CUDA Toolkit 并设置了环境变量。

### 2. 架构不兼容
```
nvcc fatal: Unsupported gpu architecture
```
**解决方案**：检查您的 GPU 计算能力，并在 CMakeLists.txt 中调整 `CMAKE_CUDA_ARCHITECTURES`。

### 3. C++ 标准错误
```
error: unsupported c++ standard
```
**解决方案**：确保编译器支持 C++11 或更高版本。

## 性能优化

- 使用 Release 构建获得最佳性能
- 根据目标 GPU 设置正确的 CUDA 架构
- 启用并行编译加速构建过程

## 测试说明

- `test_tensor`：测试 Tensor 类的基础功能、内存管理、设备迁移等
- `test_activations`：测试 ReLU 和 Sigmoid 激活函数的正向和反向传播

测试程序会自动检测 GPU 可用性，如果没有 GPU 则只运行 CPU 测试。