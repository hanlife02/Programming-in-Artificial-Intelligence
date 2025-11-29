import os
import sys
from pathlib import Path


def add_build_to_path():
    """Ensure the locally built extension can be imported."""
    script_path = Path(__file__).resolve()
    potential_roots = [
        script_path.parent,
        script_path.parent.parent,
    ]

    added = False
    for root in potential_roots:
        build_dir = root / "build"
        candidates = [
            build_dir,
            build_dir / "Release",
            build_dir / "Debug",
            build_dir / "RelWithDebInfo",
        ]
        for candidate in candidates:
            if candidate.exists():
                sys.path.append(str(candidate))
                added = True
        if added:
            break

    cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")
    if hasattr(os, "add_dll_directory") and cuda_bin.exists():
        os.add_dll_directory(str(cuda_bin))

    if not added:
        raise FileNotFoundError(
            f"No build outputs found relative to {script_path}. Run CMake build first."
        )


def main():
    add_build_to_path()

    import mytensor  # type: ignore  # imported only after sys.path tweak

    print("Creating Tensor([2, 3], 'cpu') ...")
    tensor = mytensor.Tensor([2, 3], "cpu")

    tensor.set_data([float(i) for i in range(tensor.size)])

    print("Tensor:", tensor)
    print("Shape:", tensor.shape)
    print("Device:", tensor.device)
    print("Size:", tensor.size)
    print("Values:", tensor.to_list())


if __name__ == "__main__":
    main()
