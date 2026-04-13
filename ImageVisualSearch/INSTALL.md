# ImageVisualSearch - Installation Guide

Complete step-by-step installation instructions for all operating systems.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Windows Setup](#windows-setup)
3. [Linux/Ubuntu Setup](#linuxubuntu-setup)
4. [macOS Setup](#macos-setup)
5. [Creating Conda Environment](#creating-conda-environment)
6. [Installation Verification](#installation-verification)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 8GB RAM (16GB recommended)
- Optional: NVIDIA GPU with CUDA 11.8+ for GPU acceleration

### Install Python

**Windows**: Download from [python.org](https://www.python.org/downloads/)  
**Linux/Mac**: Usually pre-installed, or use package manager

Verify installation:
```bash
python --version
pip --version
```

## Windows Setup

### Step 1: Install Tesseract OCR

Tesseract is required for text extraction from images.

**Using UB Mannheim Installer (Recommended):**

1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Look for `tesseract-ocr-w64-setup-v5.x.x.exe` (latest version)
3. Run installer with default settings
4. Default installation path: `C:\Program Files\Tesseract-OCR`

**Verify Installation:**
```bash
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

### Step 2: Install Git (Optional but Recommended)

Download from: https://git-scm.com/download/win

### Step 3: Clone or Extract Project

**Using Git:**
```bash
git clone https://github.com/yourusername/image-visual-search.git
cd ImageVisualSearch
```

**Or extract ZIP file manually**

### Step 4: Create Virtual Environment

Using venv:
```bash
python -m venv venv
venv\Scripts\activate
```

Or using Anaconda (see [Creating Conda Environment](#creating-conda-environment) section below)

### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Configure Environment Variables

```bash
copy .env.example .env
```

Edit `.env` and update:
```
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
DEVICE=cpu  # or 'cuda' if you have NVIDIA GPU
```

### Step 7: Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import pytesseract; print('Tesseract OK')"
```

## Linux/Ubuntu Setup

### Step 1: Install System Dependencies

**Ubuntu 20.04/22.04:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    tesseract-ocr \
    libtesseract-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install -y \
    python3.10 \
    python3-pip \
    git \
    gcc-c++ \
    tesseract-devel
```

**Verify Tesseract:**
```bash
tesseract --version
```

### Step 2: Clone Project

```bash
git clone https://github.com/yourusername/image-visual-search.git
cd ImageVisualSearch
```

### Step 3: Create Virtual Environment

Using venv:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

Or using Conda (see section below)

### Step 4: Install Python Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 5: Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```
TESSERACT_PATH=/usr/bin/tesseract
DEVICE=cpu  # or 'cuda' for GPU
```

### Step 6: Verification

```bash
python -c "import torch; print('PyTorch OK')"
python -c "import cv2; print('OpenCV OK')"
```

## macOS Setup

### Step 1: Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Tesseract and Dependencies

```bash
brew install tesseract opencv libomp
brew install python@3.10  # If not using system Python
```

### Step 3: Clone Project

```bash
git clone https://github.com/yourusername/image-visual-search.git
cd ImageVisualSearch
```

### Step 4: Create Virtual Environment

Using venv:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

Or using Conda (see section below)

### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```
TESSERACT_PATH=/usr/local/bin/tesseract
DEVICE=cpu  # or 'mps' for Apple Silicon GPU
```

### Step 7: Verify Installation

```bash
python -c "import torch; print('PyTorch OK')"
tesseract --version
```

## Creating Conda Environment

If you prefer to use Conda (recommended for easier dependency management):

### Option 1: Create from environment.yml

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate visual-search-env

# Verify
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### Option 2: Create Manually

```bash
# Create environment with Python 3.10
conda create -n visual-search-env python=3.10

# Activate environment
conda activate visual-search-env

# Install PyTorch (choose one based on your system)
# CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU (NVIDIA):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Deactivate Environment

```bash
conda deactivate
```

## Installation Verification

Create a test script to verify all components are working:

```bash
python verify_installation.py
```

Or manually verify:

```python
#!/usr/bin/env python3
"""Verify ImageVisualSearch installation"""

import sys

def verify_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

def verify_tesseract():
    try:
        import pytesseract
        from PIL import Image
        print("✓ pytesseract and PIL (Tesseract)")
        return True
    except Exception as e:
        print(f"✗ Tesseract: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ImageVisualSearch Installation Verification")
    print("=" * 50)
    print()
    
    modules = [
        "torch",
        "torchvision",
        "ultralytics",
        "cv2",
        "numpy",
        "scipy",
        "sklearn",
        "faiss",
        "PIL",
        "gradio",
        "pytest",
    ]
    
    failed = []
    
    for module in modules:
        if not verify_import(module):
            failed.append(module)
    
    print()
    
    if not verify_tesseract():
        failed.append("tesseract")
    
    print()
    print("=" * 50)
    if failed:
        print(f"✗ {len(failed)} module(s) failed to import:")
        for module in failed:
            print(f"  - {module}")
        sys.exit(1)
    else:
        print("✓ All dependencies verified successfully!")
        sys.exit(0)
```

## Setup Project Structure

Run the automated setup script:

```bash
python setup_project.py
```

This will create all required directories and `__init__.py` files.

## Running Tests

Verify everything works with tests:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=modules --cov-report=html
```

## GPU Setup (Optional)

If you have an NVIDIA GPU and want to use GPU acceleration:

### Install NVIDIA CUDA

1. Download from: https://developer.nvidia.com/cuda-downloads
2. Install CUDA 11.8 or later
3. Download cuDNN from: https://developer.nvidia.com/cudnn

### Install GPU-enabled PyTorch

```bash
# Using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Using conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Verify GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Issue: Tesseract not found

**Windows:**
```bash
# Check if path is correct in .env
where tesseract  # Should show the path
```

**Linux/Mac:**
```bash
which tesseract
```

**Solution:** Update `TESSERACT_PATH` in `.env` file

### Issue: Permission Denied on Linux/Mac

```bash
chmod +x setup_project.py
./setup_project.py
```

### Issue: DLL Load Failed (Windows)

This usually happens with CUDA/GPU packages:
1. Install Visual Studio Build Tools
2. Reinstall torch with correct CUDA version

### Issue: Out of Memory

Reduce batch size in config or use CPU:
```bash
DEVICE=cpu
```

### Issue: Module Import Errors

Reinstall dependencies:
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

### Issue: PyTorch Version Mismatch

```bash
# Uninstall old version
pip uninstall torch torchvision torchaudio -y

# Install correct version for your system
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

## Post-Installation

1. Create `.env` from `.env.example`
2. Download pre-trained models:
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```
3. Set up your data directories in `data/`
4. Read documentation in `README.md`

## Getting Help

- Check [README.md](README.md) for usage examples
- Review test files in `tests/` directory
- Check existing issues on GitHub
- Create a new issue with detailed error messages

---

**Last Updated**: 2024  
**Version**: 1.0.0
