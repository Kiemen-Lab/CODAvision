"""
Test script to verify robust framework detection.

This script tests that the framework detection gracefully handles
broken installations (like PyTorch DLL errors on Windows).
"""

import sys
import logging
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')  # Set console to UTF-8

# Setup logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("Testing Framework Detection")
print("=" * 60)

# Test 1: Framework detection
print("\n[Test 1] Testing detect_framework_availability()...")
try:
    from base.models.base import detect_framework_availability

    avail = detect_framework_availability()

    print(f"  TensorFlow available: {avail['tensorflow']}")
    if avail['tensorflow']:
        print(f"    Version: {avail['tensorflow_version']}")
        print(f"    GPU support: {avail['tensorflow_gpu']}")

    print(f"  PyTorch available: {avail['pytorch']}")
    if avail['pytorch']:
        print(f"    Version: {avail['pytorch_version']}")
        print(f"    GPU support: {avail['pytorch_gpu']}")

    print("  [PASS] Framework detection completed without crashing")

except Exception as e:
    print(f"  [FAIL] Framework detection failed: {e}")
    sys.exit(1)

# Test 2: Framework configuration
print("\n[Test 2] Testing framework configuration...")
try:
    from base.config import get_framework_config

    config = get_framework_config()
    print(f"  Configured framework: {config['framework']}")
    print(f"  [PASS] Framework configuration successful")

except Exception as e:
    print(f"  [FAIL] Framework configuration failed: {e}")
    sys.exit(1)

# Test 3: Model creation with automatic fallback
print("\n[Test 3] Testing model_call() with framework selection...")
try:
    from base.models.backbones import model_call

    # This should use TensorFlow if PyTorch is broken
    print("  Creating model with default framework...")
    model = model_call(
        name="DeepLabV3_plus",
        IMAGE_SIZE=512,
        NUM_CLASSES=8,
        l2_regularization_weight=0.0001
    )
    print(f"  [PASS] Model created successfully: {type(model)}")

except Exception as e:
    print(f"  [FAIL] Model creation failed: {e}")
    sys.exit(1)

# Test 4: Explicit TensorFlow selection
print("\n[Test 4] Testing explicit TensorFlow framework selection...")
try:
    from base.models.backbones import model_call

    model = model_call(
        name="DeepLabV3_plus",
        IMAGE_SIZE=512,
        NUM_CLASSES=8,
        l2_regularization_weight=0.0001,
        framework="tensorflow"
    )
    print(f"  [PASS] TensorFlow model created successfully: {type(model)}")

except Exception as e:
    print(f"  [FAIL] TensorFlow model creation failed: {e}")
    sys.exit(1)

# Test 5: PyTorch with graceful error handling
print("\n[Test 5] Testing PyTorch framework selection (should fail gracefully if broken)...")
try:
    from base.models.backbones import model_call

    model = model_call(
        name="DeepLabV3_plus",
        IMAGE_SIZE=512,
        NUM_CLASSES=8,
        l2_regularization_weight=0.0001,
        framework="pytorch"
    )
    print(f"  [PASS] PyTorch model created successfully: {type(model)}")

except ImportError as e:
    print(f"  [PASS] PyTorch not available (expected): {str(e).split(chr(10))[0]}")

except Exception as e:
    print(f"  [FAIL] Unexpected error type: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! Framework detection is robust.")
print("=" * 60)
