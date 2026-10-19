#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CUDA Configuration Check and PyTorch Installation Guide

This script detects your CUDA version, checks PyTorch installation status,
and provides clear installation instructions for GPU support.

Usage:
    python scripts/check_cuda_version.py
"""

import subprocess
import sys
import re
import platform
from typing import Optional, Tuple, List

# Set UTF-8 encoding for Windows console
if platform.system() == "Windows":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def run_command(cmd: List[str]) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0, result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, ""


def detect_nvidia_gpu() -> Tuple[Optional[str], Optional[List[str]]]:
    """Detect NVIDIA GPU and CUDA driver version using nvidia-smi."""
    success, output = run_command(['nvidia-smi'])

    if not success:
        return None, None

    # Extract CUDA version
    cuda_match = re.search(r'CUDA Version:\s*(\d+\.\d+)', output)
    cuda_version = cuda_match.group(1) if cuda_match else None

    # Extract GPU names
    gpu_matches = re.findall(r'(NVIDIA|GeForce|Quadro|Tesla|RTX|GTX|A\d+|H\d+)[^\n|]*', output)
    gpus = []
    for match in gpu_matches:
        gpu = match.strip()
        # Clean up the GPU name
        if gpu and not gpu.startswith('NVIDIA Driver') and not gpu.startswith('NVIDIA-SMI'):
            gpus.append(gpu)

    # Remove duplicates while preserving order
    unique_gpus = []
    seen = set()
    for gpu in gpus:
        if gpu not in seen:
            unique_gpus.append(gpu)
            seen.add(gpu)

    return cuda_version, unique_gpus if unique_gpus else None


def check_pytorch_installation() -> Tuple[bool, Optional[str], bool]:
    """
    Check PyTorch installation status.

    Returns:
        (is_installed, version, has_cuda_support)
    """
    try:
        import torch
        version = torch.__version__
        has_cuda = torch.cuda.is_available()
        return True, version, has_cuda
    except ImportError:
        return False, None, False


def print_status_summary(
    cuda_version: Optional[str],
    gpus: Optional[List[str]],
    torch_installed: bool,
    torch_version: Optional[str],
    torch_has_cuda: bool
):
    """Print a comprehensive status summary."""
    print("\n" + "="*60)
    print("     CUDA Configuration Check for CODAvision")
    print("="*60 + "\n")

    # System Information
    print("System Information:")
    print("-" * 60)

    if cuda_version and gpus:
        print(f"  GPU(s): {', '.join(gpus)}")
        print(f"  CUDA Driver Version: {cuda_version}")
    elif cuda_version:
        print(f"  CUDA Driver Version: {cuda_version}")
        print("  GPU: NVIDIA GPU detected")
    else:
        print("  GPU: No NVIDIA GPU detected")
        print("  CUDA: Not available")

    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")

    # PyTorch Information
    print("\nPyTorch Installation:")
    print("-" * 60)

    if torch_installed:
        print(f"  PyTorch Version: {torch_version}")

        # Check if CUDA-enabled version
        if '+cu' in torch_version:
            cuda_build = re.search(r'\+cu(\d+)', torch_version)
            if cuda_build:
                print(f"  CUDA Build: {cuda_build.group(1)} (CUDA {int(cuda_build.group(1))/10:.1f})")
        elif '+cpu' in torch_version:
            print("  Build: CPU-only")
        else:
            print("  Build: Standard (likely CPU-only)")

        if torch_has_cuda:
            print("  CUDA Available: Yes ✓")
            # Check number of GPUs
            try:
                import torch
                num_gpus = torch.cuda.device_count()
                print(f"  Detected GPUs: {num_gpus}")
                for i in range(num_gpus):
                    print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            except:
                pass
        else:
            print("  CUDA Available: No ✗")
    else:
        print("  PyTorch: Not installed")

    print("\n" + "="*60)

    # Status and Recommendations
    if not cuda_version:
        print("\nStatus: No NVIDIA GPU detected")
        print("-" * 60)
        print("\nYour system does not have an NVIDIA GPU or CUDA drivers.")
        print("CODAvision can use CPU for computations.")
        print("\nTo install CODAvision:")
        print("  uv sync --locked")
        print("\nNote: Training will be significantly slower on CPU.")

    elif torch_installed and torch_has_cuda:
        print("\nStatus: GPU support is working! ✓")
        print("-" * 60)
        print("\nYour PyTorch installation has CUDA support and can use your GPU.")
        print("This checks PyTorch only; TensorFlow GPU support is separate.")

        if '+cu' not in torch_version:
            print("\nNote: Your PyTorch version string doesn't show CUDA build info,")
            print("but CUDA is available. This is normal for some installations.")

        print("\nTo verify GPU support:")
        print('  python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"')

    elif torch_installed and not torch_has_cuda:
        print("\nStatus: PyTorch installed but GPU support not working ✗")
        print("-" * 60)
        print("\nYou have PyTorch installed, but it cannot access your GPU.")
        print("Check the PyTorch build, NVIDIA driver, and GPU visibility.")

        print("\nVerify installation:")
        print('  python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"')

    else:  # PyTorch not installed
        print("\nStatus: PyTorch not installed")
        print("-" * 60)
        print("\nPyTorch is not installed. To use GPU acceleration:")

        print("\nVerify installation:")
        print('  python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"')

    # Installation commands must match the shared framework dependencies.
    if cuda_version:
        print("\n" + "="*60)
        print("Installation with current dependencies:")
        print("-" * 60)
        print("From the checkout, use the locked CUDA 12.8 environment:")
        print("  uv sync --locked")
        print("  uv run --locked CODAvision")
        print("The NVIDIA driver must support CUDA 12.8; see README.md for requirements.")
    if platform.system() == "Windows":
        print("Native Windows requires Python 3.10, TensorFlow 2.10.1, and CUDA 11.2/cuDNN 8.1.")
        print("Install the legacy TensorFlow runtime separately; see README.md (Step 3).")
        print("PyTorch continues to use its own CUDA 12.8 wheels.")

    # Troubleshooting
    print("\n" + "="*60)
    print("Troubleshooting:")
    print("-" * 60)
    print("\nCommon Issues:")
    print("  1. 'CUDA available: False'")
    print("     → Check the PyTorch build, NVIDIA driver, and GPU visibility.")
    print("\n  2. 'nvidia-smi' not found")
    print("     → Install NVIDIA drivers from nvidia.com/drivers")
    print("\n  3. Out of memory errors")
    print("     → Reduce batch size in CODAvision settings")
    print("\n  4. Multiple CUDA versions")
    print("     → Use the documented environment; avoid mixing CUDA package versions.")

    print("\nFor detailed instructions, see:")
    print("  - CLAUDE.md (Framework Selection section)")
    print("  - README.md (Installation section)")
    print("\n" + "="*60 + "\n")


def main():
    """Main function to check CUDA configuration and provide guidance."""
    # Detect NVIDIA GPU and CUDA
    cuda_version, gpus = detect_nvidia_gpu()

    # Check PyTorch installation
    torch_installed, torch_version, torch_has_cuda = check_pytorch_installation()

    # Print comprehensive status
    print_status_summary(
        cuda_version,
        gpus,
        torch_installed,
        torch_version,
        torch_has_cuda
    )


if __name__ == "__main__":
    main()
