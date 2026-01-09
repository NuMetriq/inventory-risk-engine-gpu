"""
Sanity check: confirm Python can see an NVIDIA GPU.
This will later be extended for CUDA / PyTorch checks.
"""

import subprocess


def check_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("nvidia-smi output:\n")
        print(result.stdout)
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. NVIDIA drivers may not be installed.")
    except subprocess.CalledProcessError as e:
        print("ERROR: nvidia-smi failed.")
        print(e.stderr)


if __name__ == "__main__":
    check_nvidia_smi()