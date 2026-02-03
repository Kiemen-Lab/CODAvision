# CODAvision
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2025.04.11.648464-blue)](https://www.biorxiv.org/content/10.1101/2025.04.11.648464v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

CODAvision is an open-source Python package designed for semantic segmentation of biomedical images through a user-friendly interface.

---

## Table of Contents

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
4. [Adding Custom Model Architectures](#-4-adding-custom-model-architectures)

---

## 1. System Requirements

### 🧰 Hardware

- **Minimum Requirements:**
  - Computer with ≥16 GB RAM
  - NVIDIA GPU with ≥8 GB VRAM (Windows/Linux only)
  - Operating System: Windows 10/11, macOS 11+, or Linux
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

### Step 1: Install Miniconda

Download and install Miniconda by following the instructions provided [here](https://docs.anaconda.com/miniconda/).

---

### Step 2: Create and Activate CODAvision Environment

**For Windows and Linux:**
```bash
conda create -n CODAvision python=3.9
conda activate CODAvision
```

**For macOS:**

- **Apple Silicon with GPU support (M1/M2/M3/M4)** — requires Python 3.10+:
```bash
conda create -n CODAvision python=3.10
conda activate CODAvision
```
---

### Step 3: Install CUDA Toolkit and cuDNN

**For Windows and Linux only:**

Ensure that CUDA drivers are installed as per the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Then, install the CUDA Toolkit and cuDNN:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

**For macOS users:** Skip this step.

---

### Step 4: Install CODAvision

> ⚠️ **Note:**  
> Ensure Git is installed. If not, download it from [here](https://git-scm.com/downloads).

**For Windows and Linux:**
```bash
pip install -e git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision
```

**For macOS:**

- **Apple Silicon with GPU acceleration (M1/M2/M3/M4):**
```bash
pip install -e "git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision[macos-silicon]"
```

This installs `tensorflow-macos`, `tensorflow-metal`, and other dependencies. Do not install `keras` separately (it's included).

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

> 💡 **PyTorch GPU Support (Optional):**
> PyTorch is installed automatically with CPU support. For NVIDIA GPU acceleration with PyTorch, install CUDA-enabled PyTorch *before* installing CODAvision:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -e .
> ```
---

### 🖼️ Step 5: Launch CODAvision GUI

After completing the installation, run the following command to launch the GUI:
```bash
python CODAvision.py
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

For comprehensive guidance on annotation dataset creation, see the [CODAvision Protocol](https://www.biorxiv.org/content/10.1101/2025.04.11.648464v1).

---

