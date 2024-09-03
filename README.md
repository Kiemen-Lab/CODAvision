 # ANACODA

ANACODA is an open-source Python package designed for microanatomical tissue labeling. It facilitates deep learning-based semantic segmentation of Whole Slide Images (WSI) through a user-friendly interface.

## Quick Install

1. **Download and Install Miniconda**

   Follow the instructions [here](https://docs.anaconda.com/miniconda/) to download and install Miniconda.

2. **Create and Activate the ANACODA Environment**

    Open your terminal or command prompt and run the following commands:
    
    ```sh
    conda create -n ANACODA python=3.9.19
    
    conda activate ANACODA

3. **Install CUDA Toolkit and cuDNN**
  
    ANACODA requires the CUDA Toolkit and cuDNN to run on NVIDIA GPUs. Follow the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to install the CUDA drivers. After installing the drivers, install the CUDA Toolkit and cuDNN using conda:

    ```sh
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

4. **Install ANACODA**
  
    Install the ANACODA package using pip:
    
    ```sh
    pip install git+https://github.com/Valentinamatos/CODA_python.git
