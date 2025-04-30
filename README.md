# CODAvision

CODAvision is an open-source Python package designed for microanatomical tissue labeling. It facilitates deep learning-based semantic segmentation of Whole Slide Images (WSI) through a user-friendly interface.

## 🚀 Quick Install

1. **📥 Download and Install Miniconda**

   Follow the instructions [📄 here](https://docs.anaconda.com/miniconda/) to download and install Miniconda.

2. **⚙️ Create and Activate the CODAvision Environment**

    Open your terminal or command prompt and run the following commands:
    
    ```sh
    conda create -n CODAvision python=3.9.19
    
    conda activate CODAvision
    ```

3. **🖥️ Install CUDA Toolkit and cuDNN**
  
    CODAvision requires the CUDA Toolkit and cuDNN to run on NVIDIA GPUs. Follow the instructions [📄 here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to install the CUDA drivers. After installing the drivers, install the CUDA Toolkit and cuDNN using conda:

    ```sh
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    ```

4. **📦 Install CODAvision**

    Install the CODAvision package using pip:

    ```sh
    pip install -e git+https://github.com/Kiemen-Lab/CODAvision.git#egg=CODAvision
    ```

> ⚠️ **Warning:**  
> You might need to install Git from the following [🔗 link](https://git-scm.com/downloads/win) to be able to run the `pip install` git link command.

> ⚠️ **Warning:**  
> After installing the package, restart your IDE and activate the environment again using `conda activate CODAvision` to ensure all the dependencies are properly loaded.
