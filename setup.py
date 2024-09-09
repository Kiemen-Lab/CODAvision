from setuptools import setup, find_packages

setup(
    name='ANACODA',
    version='0.1.0',
    description='ANACODA is an open-source Python package designed for microanatomical tissue labeling.',
    author='Valentina Matos',
    url='https://github.com/Valentinamatos/CODA_python',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'pillow==10.4.0',
        'tensorflow==2.10.1',
        'tensorflow-gpu==2.10.0',
        'keras==2.10.0',
        'opencv-python==4.10.0.84',
        'matplotlib==3.9.2',
        'scipy==1.13.1',
        'scikit-image==0.24.0',
        'xmltodict==0.13.0',
        'pandas==2.2.2',
        'seaborn==0.13.2',
        'tifffile==2024.8.28',
        'jupyter==1.1.1',
        'pip==24.2',
        'attrs==24.2.0',
        'wheel==0.43.0',
        'tornado==6.4.1',
        'jinja2==3.1.4',
        'setuptools==72.1.0',
        'packaging==24.1',
        'zipp==3.20.1',
        'openslide-python==1.3.1'
        'openpyxl==3.1.2'
    ],
    package_data={
        '': ['*.ipynb', '*.qss'],
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)