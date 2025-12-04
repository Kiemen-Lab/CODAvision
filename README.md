# CODAvision
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2025.04.11.648464-blue)](https://www.biorxiv.org/content/10.1101/2025.04.11.648464v1)

CODAvision is an open-source Python package designed for semantic segmentation of biomedical images through a user-friendly interface.

---

## 📑 Table of Contents

1. [System Requirements](#-1-system-requirements)
   - [Hardware](#️-hardware)
   - [Software](#-software)
2. [Installation Guide](#️-2-installation-guide)
   - [Step 1: Install Miniconda](#-step-1-install-miniconda)
   - [Step 2: Create and Activate CODAvision Environment](#-step-2-create-and-activate-codavision-environment)
   - [Step 3: Install CUDA Toolkit and cuDNN](#-step-3-install-cuda-toolkit-and-cudnn)
   - [Step 4: Install CODAvision](#-step-4-install-codavision)
   - [Step 5: Launch CODAvision GUI](#️-step-5-launch-codavision-gui)
3. [Demo](#-3-demo)
   - [Sample Dataset](#-sample-dataset)
   - [Instructions to Run on Sample Data](#-instructions-to-run-on-sample-data)
   - [Expected Output](#-expected-output)
   - [Expected Runtime](#-expected-runtime)

---

## 📋 1. System Requirements

### 🧰 Hardware

- **Minimum Requirements:**
  - Computer with ≥16 GB RAM
  - NVIDIA GPU with ≥8 GB VRAM
  - Operating System: Windows 10/11 or macOS 11
  - Storage: ≥2.5 GB free space
  - CUDA Toolkit (≥11.2) and cuDNN (≥8.1) installed

- **Tested Configuration:**
  - Workstation with 128 GB RAM
  - NVIDIA GeForce RTX 4090 GPU
  - Operating System: Windows 11

### 🖥️ Software

- [CODAvision Repository](https://github.com/Kiemen-Lab/CODAvision)
- Python IDE (e.g., PyCharm, Visual Studio, Spyder)
- Image Annotation Tool (choose one):
  - [Aperio ImageScope](https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope)
  - [QuPath](https://qupath.github.io)
    
    > ⚠️ **Note for QuPath Users:**  
    > To use the GUI-guided workflow in CODAvision with annotations created in QuPath, you must first export the annotations for each image as GeoJSON files via `File > Export Objects as GeoJSON`.  
    > These GeoJSON files must then be converted into XML format, which is compatible with CODAvision.  
    > You can perform this conversion using the scripts provided in the following repository: [GeoJSON2XML](https://github.com/Kiemen-Lab/GeoJSON2XML).

---

## ⚙️ 2. Installation Guide

### 📥 Step 1: Install Miniconda

Download and install Miniconda by following the instructions provided [here](https://docs.anaconda.com/miniconda/).

### 🐍 Step 2: Create and Activate CODAvision Environment

```bash
  conda create -n CODAvision python=3.9.19
  conda activate CODAvision
```

### 🔧 Step 3: GPU Setup (Optional but Recommended)

CODAvision supports both **PyTorch** and **TensorFlow** deep learning frameworks with GPU acceleration.

#### Option A: PyTorch with NVIDIA GPU (Recommended)

**Check your CUDA version first:**
```bash
python scripts/check_cuda_version.py
```

This helper script will detect your GPU, identify your CUDA version, and provide specific installation commands.

**Or install manually** (choose your CUDA version):
```bash
# For CUDA 11.8 (most compatible - works with CUDA 11.x, 12.x, 13.x drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4 (latest)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### Option B: TensorFlow with NVIDIA GPU

Install CUDA toolkit and cuDNN for TensorFlow GPU support:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

#### Option C: CPU-Only (No GPU)

Skip this step if you don't have an NVIDIA GPU or want to use CPU only.

### 📦 Step 4: Install CODAvision

Install the CODAvision package using pip:

```bash
  pip install -e git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision
```

> ⚠️ **Note:**  
> Ensure Git is installed. If not, download it from [here](https://git-scm.com/downloads/win).  
> After installation, restart your IDE and reactivate the environment:

```bash
  conda activate CODAvision
```

#### Verify GPU Support

After installation, verify that GPU acceleration is working:

```bash
# Check PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Or use the helper script for comprehensive status
python scripts/check_cuda_version.py
```

**Expected output with GPU**: `CUDA available: True` (PyTorch) or list of GPU devices (TensorFlow)

> ### ⚠️ **macOS Users**

#### Apple Silicon (M1/M2/M3/M4) with GPU Support

For Apple Silicon Macs with Metal GPU acceleration:

**PyTorch** (recommended - GPU support works automatically):
```bash
pip install -e git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision
```

**TensorFlow with Metal GPU**:
```bash
pip install -e "git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision[macos-silicon]"
```

#### Intel Mac or CPU-Only

Standard installation works for Intel Macs or CPU-only setup:
```bash
pip install -e git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision
```

> 💡 **Note**: PyTorch automatically detects and uses MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon.

### 🔄 Framework Selection

CODAvision supports both PyTorch and TensorFlow. By default, PyTorch is used if available.

**To switch frameworks:**

```bash
# Use PyTorch (default)
export CODAVISION_FRAMEWORK=pytorch

# Use TensorFlow
export CODAVISION_FRAMEWORK=tensorflow

# On Windows (PowerShell)
$env:CODAVISION_FRAMEWORK="pytorch"
```

**Framework features:**
- Both support DeepLabV3+ and UNet architectures
- PyTorch: Faster training on some systems, better Apple Silicon support
- TensorFlow: More mature, longer development history
- Models trained with one framework cannot be used with the other

For detailed configuration options, see [CLAUDE.md](CLAUDE.md).

### 🖼️ Step 5: Launch CODAvision GUI

After completing the installation, run the `CODAvision.py` script to launch the GUI and begin data parameterization.

**⏱️ Typical Installation Time:** Approximately 10–15 minutes on a standard desktop computer.

---

## 🎬 3. Demo

### 📂 Sample Dataset

Access the sample dataset [here](https://drive.google.com/drive/folders/1dkF10ojFylRl1OrcjRcgz0JIey1-zJwB?usp=drive_link).

### 📝 Instructions to Run on Sample Data

Acess the demo instructions [here](https://drive.google.com/file/d/1ZtL0MrC_uGJmYUgUi4EBto6gyXNsg3Hh/view?usp=drive_link) 

### 📊 Expected Output

Acess the expected output [here](https://drive.google.com/drive/folders/1D3xujNXFZjP76CYznlfZtLYrKdyaKGDU?usp=sharing).

### ⏳ Expected Runtime

- **GPU-Powered Workstation:** Approximately 2–3 hours for model training and image processing.
- **Desktop Computer with no GPU:** Image processing and training time may extend up to 10 hours.

---

For a more comprehensive guidance on annotation dataset creation [CODAvision Protocol](https://www.biorxiv.org/content/10.1101/2025.04.11.648464v1).

---
