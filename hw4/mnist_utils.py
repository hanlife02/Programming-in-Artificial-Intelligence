from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms

from test_mytensor import add_build_to_path


def load_mytensor():
    add_build_to_path()
    import mytensor

    return mytensor


def load_mnist_as_tensors(
    root: str | Path,
    train: bool = True,
    limit: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[object, object]:
    """
    Read MNIST with torchvision, convert to numpy, then to custom Tensor.

    Returns:
        (images_tensor, labels_numpy)
    """

    dataset = datasets.MNIST(
        root=str(root),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    data = dataset.data.float() / 255.0
    targets = dataset.targets

    if limit is not None:
        data = data[:limit]
        targets = targets[:limit]

    images = data.unsqueeze(1).numpy().astype(np.float32)
    labels = targets.numpy().astype(np.int64)

    mytensor = load_mytensor()
    image_tensor = mytensor.from_numpy(images, device)
    label_tensor = mytensor.from_numpy(labels.astype(np.float32), device)

    return image_tensor, label_tensor


if __name__ == "__main__":
    imgs, lbls = load_mnist_as_tensors("./data", limit=32)
    print("Loaded MNIST batch:", imgs, "labels tensor shape:", lbls.shape)
