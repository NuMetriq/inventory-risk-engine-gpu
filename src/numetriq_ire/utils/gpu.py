from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_summary() -> dict:
    """Return a small dict describing the current GPU/CUDA environment."""
    summary = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if summary["cuda_available"] and summary["device_count"] > 0:
        summary.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            }
        )
    return summary


def print_gpu_summary() -> None:
    s = gpu_summary()
    for k, v in s.items():
        print(f"{k}: {v}")