"""
Sanity check: confirm PyTorch can use CUDA and run a tiny GPU operation.
"""

import torch


def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("\nCUDA is NOT available to PyTorch.")
        print("Common causes: wrong torch build, driver mismatch, or missing CUDA runtime.")
        return

    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda runtime:", torch.version.cuda)

    # Tiny GPU compute test
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    print("compute ok; y device:", y.device)
    print("allocated (MB):", round(torch.cuda.memory_allocated() / 1024**2, 2))


if __name__ == "__main__":
    main()