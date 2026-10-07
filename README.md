# CODAvision
[![Nature Protocols](https://img.shields.io/badge/Nature%20Protocols-s41596--026--01404--3-purple)](https://www.nature.com/articles/s41596-026-01404-3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

CODAvision is an open-source Python package designed for semantic segmentation of biomedical images through a user-friendly interface.

---

## Table of Contents

1. [System Requirements](#-1-system-requirements)
   - [Hardware](#️-hardware)
   - [Software](#-software)
2. [Installation Guide](#️-2-installation-guide)
   - [Install with uv](#install-with-uv)
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
4. [Adding Custom Model Architectures](#-4-adding-custom-model-architectures)

---

## 1. System Requirements

### 🧰 Hardware

- **Minimum Requirements:**
  - Computer with ≥16 GB RAM
  - NVIDIA GPU with ≥8 GB VRAM (Windows/Linux only)
  - Operating System: Windows 10/11, macOS (Apple Silicon, CPU development), or Linux
  - Storage: ≥2.5 GB free space

- **Tested Configuration:**
  - Workstation with 128 GB RAM
  - NVIDIA GeForce RTX 4090 GPU
  - Operating System: Windows 11

### 🖥️ Software

- [CODAvision Repository](https://github.com/Kiemen-Lab/CODAvision)
- Python IDE (optional, e.g., PyCharm, Visual Studio, Spyder)
- Image Annotation Tool (choose one):
  - [Aperio ImageScope](https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope)
  - [QuPath](https://qupath.github.io)
    
    > ⚠️ **Note for QuPath Users:**  
    > To use the GUI-guided workflow in CODAvision with annotations created in QuPath, you must first export the annotations for each image as GeoJSON files via `File > Export Objects as GeoJSON`.  
    > These GeoJSON files must then be converted into XML format, which is compatible with CODAvision.  
    > You can perform this conversion using the scripts provided in the following repository: [GeoJSON2XML](https://github.com/Kiemen-Lab/GeoJSON2XML).

---

## ⚙️ 2. Installation Guide

### Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.10.10 or
newer** (check with `uv --version`; upgrade with `uv self update`), then clone
CODAvision and install the locked dependencies:

```bash
git clone https://github.com/Kiemen-Lab/CODAvision.git
cd CODAvision
uv sync --locked
uv run --locked CODAvision
```

> **Windows with Anaconda:** an older `uv.exe` in Anaconda's base environment can
> come first on `PATH` even when the CODAvision conda environment is active, which
> fails with `Required uv version >=0.10.10 does not match the running version`.
> Run `where.exe uv` and `uv --version` to check. If the wrong copy is used, call
> the environment's copy instead, e.g. `& "$env:CONDA_PREFIX\Scripts\uv.exe" sync --locked`
> in PowerShell, or define `Set-Alias uv "$env:CONDA_PREFIX\Scripts\uv.exe"` for the session.

Both TensorFlow and PyTorch are installed by default. Linux x86_64 uses
TensorFlow 2.21 / tf-keras 2.21 and PyTorch 2.11 / torchvision 0.26 with CUDA
12.8. Native Windows uses TensorFlow 2.10.1 / Keras 2.10 and the same PyTorch
versions; first install CUDA 11.2 / cuDNN 8.1 using Steps 2–3 below (or
[environment-windows.yml](environment-windows.yml)) and keep Conda activated.
Apple Silicon macOS uses TensorFlow 2.21 / tf-keras 2.21 and PyTorch from
PyPI for CPU development, without CUDA or `tensorflow-metal`.

`.python-version` selects Python 3.10. To use Python 3.11 on Linux or macOS,
run `uv python pin 3.11` before syncing so later `uv run` commands keep using
3.11. Native Windows requires Python 3.10; Python 3.12+ and Intel macOS are
not included in this environment.

For development, use `uv sync --locked --extra dev` and
`uv run --locked --extra dev pytest`. Use `--extra test` for minimal test tools.

To install with Conda and pip instead, follow the steps below.

---

### Step 1: Install Miniconda

Download and install Miniconda by following the instructions provided [here](https://docs.anaconda.com/miniconda/).

---

### Step 2: Create and Activate CODAvision Environment

**For Windows and Linux:**
```bash
conda create -n CODAvision python=3.10
conda activate CODAvision
```

**For macOS:**

- **Apple Silicon (M1/M2/M3/M4), CPU development** — use Python 3.10 or 3.11:
```bash
conda create -n CODAvision python=3.10
conda activate CODAvision
```
---

### Step 3: Install CUDA Toolkit and cuDNN

**For Windows only:**

Ensure that CUDA drivers are installed as per the instructions [here](https://www.tensorflow.org/install/pip#windows-native). Then, install the CUDA Toolkit and cuDNN:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Keep Conda activated when running CODAvision so TensorFlow can find these
libraries. PyTorch supplies its own CUDA 12.8 runtime. CODAvision disables
AdamW JIT compilation on Windows because this Conda runtime does not include
the XLA compiler tools; custom AdamW code must also use `jit_compile=False`.

**For Linux users:** TensorFlow's CUDA libraries are installed with CODAvision;
a compatible NVIDIA driver is still required.

**For macOS users:** Skip this step.

---

### Step 4: Install CODAvision

> ⚠️ **Note:**  
> Ensure Git is installed. If not, download it from [here](https://git-scm.com/downloads).

**For Windows and Linux:**
```bash
pip install -e git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision --extra-index-url https://download.pytorch.org/whl/cu128
```

**For macOS:**

- **Apple Silicon (M1/M2/M3/M4), CPU development:**
```bash
pip install -e "git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision"
```

This installs TensorFlow, tf-keras, PyTorch, and other dependencies. Do not install `keras` separately (it's included).

After installation, restart your IDE and reactivate the environment:
```bash
conda activate CODAvision
```

>💡 **Alternative installation option:**
> You can also clone the repository first and install dependencies locally:  
> ```bash
> git clone https://github.com/Kiemen-Lab/CODAvision.git
> cd CODAvision
> pip install -e .
> ```

> 💡 **PyTorch GPU Support (Windows/Linux):**
> For NVIDIA GPU acceleration with PyTorch, install CUDA-enabled PyTorch *before* installing CODAvision:
> ```bash
> pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
> pip install -e .
> ```

TensorFlow 2.21 uses legacy Keras through `tf-keras`. In notebooks and custom
models, import `base` before accessing Keras and use `from tensorflow import keras`,
or set `TF_USE_LEGACY_KERAS=1` before starting Python. Existing HDF5 checkpoints
retain their historical `.keras` filenames.

---

### 🖼️ Step 5: Launch CODAvision GUI

After completing the installation, run the following command to launch the GUI:
```bash
CODAvision
```

**⏱️ Typical Installation Time:** Approximately 10–15 minutes on a standard desktop computer.

---

## 🎬 3. Demo

### 📂 Sample Dataset

Access the sample dataset [here](https://drive.google.com/drive/folders/1dkF10ojFylRl1OrcjRcgz0JIey1-zJwB?usp=drive_link).

### 📝 Instructions to Run on Sample Data

Access the demo instructions [here](https://drive.google.com/file/d/1ZtL0MrC_uGJmYUgUi4EBto6gyXNsg3Hh/view?usp=drive_link).

### 📊 Expected Output

Access the expected output [here](https://drive.google.com/drive/folders/1D3xujNXFZjP76CYznlfZtLYrKdyaKGDU?usp=sharing).

### ⏳ Expected Runtime

- **GPU-Powered Workstation:** Approximately 2–3 hours for model training and image processing.
- **Desktop Computer with no GPU:** Image processing and training time may extend up to 10 hours.

---

## 🔧 4. Adding Custom Model Architectures

### Adding Custom Model Architectures

CODAvision uses a flexible plugin-based architecture that allows you to easily integrate new segmentation models.

To add your own model architecture:

1. Review the comprehensive guide in [MODEL_PLUGIN_ARCHITECTURE.md](docs/MODEL_PLUGIN_ARCHITECTURE.md)
2. Follow the abstract base class pattern to ensure compatibility
3. Register your model in the factory function
4. Your model will automatically appear in the GUI and training pipeline

The plugin architecture supports:
- TensorFlow/Keras models (DeepLabV3+, UNet)
- PyTorch models (DeepLabV3+) with Keras-compatible adapter
- Multi-framework model registry
- Seamless integration with existing workflows

---

For comprehensive guidance on annotation dataset creation, see the [CODAvision Protocol](https://www.nature.com/articles/s41596-026-01404-3).

---

