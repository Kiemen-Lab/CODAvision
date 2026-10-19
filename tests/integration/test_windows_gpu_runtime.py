"""Regression for native Windows AdamW and CUDA runtime coexistence."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows CUDA regression")
def test_windows_gpu_runtime():
    # Opt in so ordinary Windows CPU-only test runs do not require an NVIDIA GPU.
    if os.environ.get("CODAVISION_TEST_GPU") != "1":
        pytest.skip("Set CODAVISION_TEST_GPU=1 to require both GPU frameworks")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "verify_gpu_runtime.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TF GPU forward/backward PASS: DeepLabV3PlusTrainer" in result.stdout
    assert "TF GPU forward/backward PASS: UNetTrainer" in result.stdout
    assert "Torch GPU forward/backward PASS:" in result.stdout
