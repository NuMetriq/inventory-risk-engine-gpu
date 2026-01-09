"""
Sanity check: confirm PyTorch can use CUDA and run a tiny GPU operation.
"""

import torch
from numetriq_ire.utils.gpu import get_device, print_gpu_summary


def main():
    print_gpu_summary()
    device = get_device()
    x = torch.randn(2000,2000, device=device)
    y = x @ x
    print("compute ok; y device:", y.device)


if __name__ == "__main__":
    main()