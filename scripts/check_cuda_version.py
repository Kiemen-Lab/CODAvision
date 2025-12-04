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


def recommend_cuda_version(driver_version: Optional[str]) -> str:
    """Recommend appropriate CUDA wheel version based on driver."""
    if not driver_version:
        return "cu118"  # Default to most compatible

    try:
        major = int(float(driver_version))
        if major >= 13:
            return "cu124"
        elif major >= 12:
            return "cu121"
        else:
            return "cu118"
    except (ValueError, TypeError):
        return "cu118"


def get_install_command(cuda_version: str) -> Tuple[str, str]:
    """Get installation commands for the specified CUDA version."""
    base_url = f"https://download.pytorch.org/whl/{cuda_version}"

    if platform.system() == "Windows":
        pytorch_cmd = f"pip install torch torchvision --index-url {base_url}"
        codavision_cmd = "pip install -e ."
    else:
        pytorch_cmd = f"pip install torch torchvision --index-url {base_url}"
        codavision_cmd = "pip install -e ."

    return pytorch_cmd, codavision_cmd


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
        print("CODAvision will use CPU for computations.")
        print("\nTo install CODAvision with CPU-only PyTorch:")
        print("  pip install -e .")
        print("\nNote: Training will be significantly slower on CPU.")

    elif torch_installed and torch_has_cuda:
        print("\nStatus: GPU support is working! ✓")
        print("-" * 60)
        print("\nYour PyTorch installation has CUDA support and can use your GPU.")
        print("CODAvision is ready to use GPU acceleration.")

        if '+cu' not in torch_version:
            print("\nNote: Your PyTorch version string doesn't show CUDA build info,")
            print("but CUDA is available. This is normal for some installations.")

        print("\nTo verify GPU support:")
        print('  python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"')

    elif torch_installed and not torch_has_cuda:
        print("\nStatus: PyTorch installed but GPU support not working ✗")
        print("-" * 60)
        print("\nYou have PyTorch installed, but it cannot access your GPU.")
        print("This usually means you have the CPU-only version.")

        recommended_cuda = recommend_cuda_version(cuda_version)
        pytorch_cmd, codavision_cmd = get_install_command(recommended_cuda)

        print(f"\nRecommended: Install PyTorch with CUDA {recommended_cuda.replace('cu', '')} support")
        print("\nInstallation commands:")
        print(f"  1. Uninstall current PyTorch:")
        print("     pip uninstall torch torchvision -y")
        print(f"\n  2. Install CUDA-enabled PyTorch:")
        print(f"     {pytorch_cmd}")
        print(f"\n  3. Install CODAvision:")
        print(f"     {codavision_cmd}")

        print("\nVerify installation:")
        print('  python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"')

    else:  # PyTorch not installed
        print("\nStatus: PyTorch not installed")
        print("-" * 60)
        print("\nPyTorch is not installed. To use GPU acceleration:")

        recommended_cuda = recommend_cuda_version(cuda_version)
        pytorch_cmd, codavision_cmd = get_install_command(recommended_cuda)

        print(f"\nRecommended: PyTorch with CUDA {recommended_cuda.replace('cu', '')} support")
        print("\nInstallation commands:")
        print(f"  1. Install CUDA-enabled PyTorch:")
        print(f"     {pytorch_cmd}")
        print(f"\n  2. Install CODAvision:")
        print(f"     {codavision_cmd}")

        print("\nVerify installation:")
        print('  python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"')

    # Additional CUDA version options
    if cuda_version:
        print("\n" + "="*60)
        print("Alternative CUDA Versions:")
        print("-" * 60)
        print("\nYou can choose different CUDA versions:")

        for cu_ver, cu_name in [("cu118", "11.8"), ("cu121", "12.1"), ("cu124", "12.4")]:
            pytorch_cmd, _ = get_install_command(cu_ver)
            print(f"\n  CUDA {cu_name}:")
            print(f"    {pytorch_cmd}")

        print("\nNote: CUDA 11.8 wheels are forward compatible with newer drivers")
        print("and work well with CUDA 12.x and 13.x driver versions.")

    # Troubleshooting
    print("\n" + "="*60)
    print("Troubleshooting:")
    print("-" * 60)
    print("\nCommon Issues:")
    print("  1. 'CUDA available: False'")
    print("     → You have CPU-only PyTorch. Reinstall with CUDA support.")
    print("\n  2. 'nvidia-smi' not found")
    print("     → Install NVIDIA drivers from nvidia.com/drivers")
    print("\n  3. Out of memory errors")
    print("     → Reduce batch size in CODAvision settings")
    print("\n  4. Multiple CUDA versions")
    print("     → Use the CUDA version matching your driver (check nvidia-smi)")

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
