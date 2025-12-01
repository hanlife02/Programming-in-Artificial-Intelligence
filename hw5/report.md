## Task 1: Operators
- `PowerScalar`: 前向用 `np.power`，梯度返回 `scalar * a ** (scalar-1)`，支持 `scalar=0` 时返回零张量。
- `EWiseDiv`/`DivScalar`: 逐点除法和标量除法，梯度分别为 `out_grad / rhs`、`-out_grad * lhs / rhs^2` 与 `out_grad / scalar`。
- `Transpose`: 前向根据 `axes` 交换维度，梯度需要计算原交换的逆排列并再次 `transpose`。
- `Reshape`: 前向 `np.reshape`，梯度把上游 reshape 回原 shape。
- `BroadcastTo`: 前向 `np.broadcast_to`，反向通过对被扩展的轴求和并 reshape，恢复输入形状。
- `Summation`: 前向 `np.sum(axis=self.axes)`，梯度先把 `out_grad` reshape 成含 1 的形状，再 `broadcast_to` 输入尺寸。
- `MatMul`: 前向 `np.matmul`；梯度对左输入为 `out_grad @ rhs^T`，对右输入为 `lhs^T @ out_grad`，并用 reduce_sum 折回原 shape（含批次或广播）。
- `Negate`: 前向 `-a`，梯度 `-out_grad`。
- `Log`: 前向 `np.log`，梯度 `out_grad / input`。
- `Exp`: 前向 `np.exp`，梯度 `out_grad * exp(input)`（用 Tensor 版本保持图）。
- `ReLU`: 前向 `np.maximum(a,0)`；梯度使用缓存的输入数值构造 `(input>0)` 掩码，与 `out_grad` 相乘。
- 测试命令：`python hw5/test_task1_forward.py`、`python hw5/test_task1_backward.py`。

## Task 2: Autodiff
- `find_topo_sort`/`topo_sort_dfs`: 使用后序 DFS，按输入边递归，保证叶子节点先入序列、根最后。
- `compute_gradient_of_variables`: 
  1. 为输出节点准备 `out_grad`，按拓扑序逆序遍历；
  2. 累加同一节点的所有贡献（保持 Tensor 运算以保留图），写入 `node.grad`；
  3. 若节点有算子，则调用 `gradient_as_tuple` 把总梯度分发给输入，只对 `requires_grad=True` 的输入累计；
  4. 不需要梯度的节点被直接跳过，允许重复 backward 和高阶梯度。
- 测试命令：`python hw5/test_task2_topo_sort.py`、`python hw5/test_task2_auto_diff.py`。
