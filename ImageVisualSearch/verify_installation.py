"""
Project Verification Script
Checks if all project components are properly set up
"""

import sys
import os
from pathlib import Path

def verify_project_structure():
    """Verify project folder structure"""
    print("=" * 60)
    print("ImageVisualSearch - Project Structure Verification")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/reference_db",
        "data/test_images",
        "models/yolo",
        "models/resnet",
        "models/ocr",
        "modules",
        "utils",
        "outputs/results",
        "outputs/reports",
        "ui",
        "tests",
    ]
    
    required_files = [
        "config.py",
        "main.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        ".env.example",
        "environment.yml",
        "INSTALL.md",
        "modules/__init__.py",
        "modules/detection.py",
        "modules/ocr_engine.py",
        "modules/similarity.py",
        "modules/retrieval.py",
        "utils/__init__.py",
        "utils/preprocessing.py",
        "utils/helpers.py",
        "ui/app.py",
        "tests/__init__.py",
        "tests/test_detection.py",
        "tests/test_ocr.py",
        "tests/test_similarity.py",
    ]
    
    print("Checking Directories:")
    print("-" * 60)
    dirs_ok = 0
    for directory in required_dirs:
        dir_path = project_root / directory
        status = "✓" if dir_path.exists() else "✗"
        print(f"{status} {directory}")
        if dir_path.exists():
            dirs_ok += 1
    
    print()
    print("Checking Files:")
    print("-" * 60)
    files_ok = 0
    for file in required_files:
        file_path = project_root / file
        status = "✓" if file_path.exists() else "✗"
        print(f"{status} {file}")
        if file_path.exists():
            files_ok += 1
    
    print()
    print("=" * 60)
    print(f"Directories: {dirs_ok}/{len(required_dirs)}")
    print(f"Files: {files_ok}/{len(required_files)}")
    print("=" * 60)
    print()
    
    if dirs_ok == len(required_dirs) and files_ok == len(required_files):
        print("✓ Project structure verification PASSED")
        return True
    else:
        print("✗ Project structure verification FAILED")
        return False


def verify_python_environment():
    """Verify Python environment and dependencies"""
    print("Python Environment Verification:")
    print("-" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print()
    
    # Try importing key modules
    modules_to_check = [
        "numpy",
        "cv2",
        "PIL",
        "torch",
        "torchvision",
    ]
    
    print("Checking Python Packages:")
    print("-" * 60)
    
    packages_ok = 0
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            version = __import__(module_name).__version__ if hasattr(__import__(module_name), '__version__') else "installed"
            print(f"✓ {module_name}: {version}")
            packages_ok += 1
        except ImportError:
            print(f"✗ {module_name}: NOT INSTALLED")
    
    print()
    optional_modules = [
        "ultralytics",
        "pytesseract",
        "faiss",
        "gradio",
    ]
    
    print("Checking Optional Packages:")
    print("-" * 60)
    for module_name in optional_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}: installed")
        except ImportError:
            print(f"- {module_name}: not installed (optional)")
    
    print()
    
    return packages_ok >= 3  # At least numpy, cv2, PIL should be installed


if __name__ == "__main__":
    structure_ok = verify_project_structure()
    print()
    env_ok = verify_python_environment()
    
    print()
    print("=" * 60)
    if structure_ok and env_ok:
        print("✓ Verification PASSED - Project is ready to use!")
        sys.exit(0)
    else:
        print("✗ Verification FAILED - Some issues need to be fixed")
        sys.exit(1)
