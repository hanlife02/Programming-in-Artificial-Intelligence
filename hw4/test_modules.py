import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from test_mytensor import add_build_to_path


def import_extension():
    add_build_to_path()
    import mytensor  # type: ignore

    return mytensor


class ModuleParityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        self.mytensor = import_extension()

    def to_tensor(self, tensor: torch.Tensor):
        array = tensor.detach().cpu().numpy().astype(np.float32)
        return self.mytensor.from_numpy(array, "cpu")

    def assertTensorClose(self, ours, reference, atol=1e-4):  # noqa: N802
        ours_np = np.array(ours.to_numpy(), copy=False)
        ref_np = reference.detach().cpu().numpy()
        np.testing.assert_allclose(ours_np, ref_np, atol=atol, rtol=1e-4)

    def test_sigmoid(self):
        x = torch.randn(4, 5, dtype=torch.float32)
        ref = torch.sigmoid(x)
        ours = self.mytensor.sigmoid_forward(self.to_tensor(x))
        self.assertTensorClose(ours, ref)

    def test_relu(self):
        x = torch.randn(3, 7, dtype=torch.float32) - 0.5
        ref = torch.relu(x)
        ours = self.mytensor.relu_forward(self.to_tensor(x))
        self.assertTensorClose(ours, ref)

    def test_linear(self):
        x = torch.randn(2, 4, dtype=torch.float32)
        weight = torch.randn(3, 4, dtype=torch.float32)
        bias = torch.randn(3, dtype=torch.float32)
        ref = F.linear(x, weight, bias)
        ours = self.mytensor.linear_forward(self.to_tensor(x), self.to_tensor(weight), self.to_tensor(bias))
        self.assertTensorClose(ours, ref)

    def test_conv2d(self):
        x = torch.randn(2, 3, 8, 8, dtype=torch.float32)
        weight = torch.randn(4, 3, 3, 3, dtype=torch.float32)
        bias = torch.randn(4, dtype=torch.float32)
        ref = F.conv2d(x, weight, bias=bias, stride=1, padding=1)
        ours = self.mytensor.conv2d_forward(
            self.to_tensor(x), self.to_tensor(weight), self.to_tensor(bias),
            stride=(1, 1), padding=(1, 1))
        self.assertTensorClose(ours, ref)

    def test_max_pool(self):
        x = torch.randn(1, 2, 6, 6, dtype=torch.float32)
        ref = F.max_pool2d(x, kernel_size=2, stride=2)
        out, _mask = self.mytensor.max_pool_forward(self.to_tensor(x), kernel=(2, 2), stride=(2, 2))
        self.assertTensorClose(out, ref)

    def test_softmax(self):
        logits = torch.randn(5, 10, dtype=torch.float32)
        ref = F.softmax(logits, dim=1)
        ours = self.mytensor.softmax_forward(self.to_tensor(logits))
        self.assertTensorClose(ours, ref)

    def test_cross_entropy(self):
        logits = torch.randn(4, 6, dtype=torch.float32)
        labels = torch.tensor([0, 2, 3, 5], dtype=torch.long)
        ref_loss = F.cross_entropy(logits, labels).item()
        probs = self.mytensor.softmax_forward(self.to_tensor(logits))
        loss = self.mytensor.cross_entropy_loss(probs, labels.tolist())
        self.assertAlmostEqual(loss, ref_loss, places=4)


if __name__ == "__main__":
    unittest.main()
