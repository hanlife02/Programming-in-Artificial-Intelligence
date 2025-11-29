# HW4 实验报告

## 1. 构建与运行方式
1. **CMake 构建**
   ```powershell
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build
   conda activate env
   python setup.py build_ext --inplace
   ```
2. **运行单测**
   ```powershell
   python test_modules.py      # 7个算子 对比 torch.nn.functional
   python test_mytensor.py     # Tensor 导入与基础读写检测
   ```
3. **MNIST 示例**
   ```powershell
   python mnist_utils.py
   ```

## 2. 运行与验证结果
- `python test_modules.py`
  ```
  .......
  ----------------------------------------------------------------------
  Ran 7 tests in 0.038s
  OK
  ```
- `python test_mytensor.py`
  ```
  Creating Tensor([2, 3], 'cpu') ...
  Tensor: <Tensor shape=[2, 3], device=cpu>
  Shape: [2, 3]
  Device: cpu
  Size: 6
  Values: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  ```
- `python mnist_utils.py`
  ```
  Loaded MNIST batch: <Tensor shape=[32, 1, 28, 28], device=cpu> labels tensor shape: [32]
  ```

## 3. 实验总结
- 完成 Tensor 的 Pybind11 绑定，并支持与 NumPy 互转。
- 复用前两次作业的 7 个算子，封装成可在 Python 调用的模块，接口与 PyTorch 基本对齐。
- 在 Python 端提供 MNIST 转自定义 Tensor 的工具，为后续训练铺路。
- 通过与 `torch.nn.functional` 的对照单测，提前验证核心算子数值正确性，减少后续搭建计算图的调试成本。


## 4. 正确性检测实现说明
- `test_mytensor.py` 中的 `add_build_to_path()` 会自动定位 CMake 构建输出目录并加入 `sys.path`，同时在 Windows 环境下补充 CUDA DLL 路径，确保 Pybind11 扩展可被顺利导入。`main()` 函数实例化 `Tensor([2, 3], "cpu")`，通过 `set_data()` 写入示例数据，再调用 `shape`、`device`、`size`、`to_list()` 等接口验证跨语言的数据读写流程是否一致，从而覆盖 Tensor 创建、宿主侧赋值与 NumPy 回读的基础功能。运行 `python test_mytensor.py` 的打印结果即为该验证的直接证据。
- `test_modules.py` 使用 `unittest` 组织 7 个核心算子的对比测试，在 `setUp()` 中固定随机种子并调用 `mytensor.from_numpy()` 把 Torch 张量转换成自定义 Tensor。随后 `test_sigmoid`、`test_relu`、`test_linear`、`test_conv2d`、`test_max_pool`、`test_softmax`、`test_cross_entropy` 分别调用绑定函数的前向实现，与 `torch.nn.functional` 的对应输出进行 `np.testing.assert_allclose` 或 `assertAlmostEqual` 的逐项比对，确保数值结果与 PyTorch 保持一致，从而在搭建更复杂网络前锁定算子级正确性。
