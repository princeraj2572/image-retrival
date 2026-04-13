#!/usr/bin/env python3
"""
Project Setup Script for ImageVisualSearch
Creates all necessary folders and __init__.py files
"""

import os
import sys
from pathlib import Path


def create_dir_structure():
    """Create the complete project folder structure."""
    
    # Define the root directory
    base_dir = Path(__file__).parent / "ImageVisualSearch"
    
    # Define all folders to create
    folders = [
        # Data folders
        "data",
        "data/raw",
        "data/processed",
        "data/reference_db",
        "data/test_images",
        # Model folders
        "models",
        "models/yolo",
        "models/resnet",
        "models/ocr",
        # Module folders
        "modules",
        "utils",
        # Output folders
        "outputs",
        "outputs/results",
        "outputs/reports",
        # UI folder
        "ui",
        # Tests folder
        "tests",
    ]
    
    # Folders that need __init__.py files
    python_packages = [
        "modules",
        "utils",
        "tests",
    ]
    
    print("=" * 60)
    print("ImageVisualSearch Project Setup")
    print("=" * 60)
    print()
    
    created_count = 0
    
    # Create all directories
    for folder in folders:
        folder_path = base_dir / folder
        try:
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created folder: {folder}")
                created_count += 1
            else:
                print(f"→ Folder already exists: {folder}")
        except Exception as e:
            print(f"✗ Error creating {folder}: {e}")
            return False
    
    print()
    
    # Create __init__.py files for Python packages
    init_count = 0
    for package in python_packages:
        init_path = base_dir / package / "__init__.py"
        try:
            if not init_path.exists():
                init_path.touch()
                print(f"✓ Created __init__.py: {package}/__init__.py")
                init_count += 1
            else:
                print(f"→ __init__.py already exists: {package}/__init__.py")
        except Exception as e:
            print(f"✗ Error creating __init__.py in {package}: {e}")
            return False
    
    print()
    print("=" * 60)
    print(f"Summary: {created_count} folder(s) created, {init_count} __init__.py file(s) created")
    print("=" * 60)
    print()
    print("Project structure setup completed successfully! ✓")
    print(f"Project root: {base_dir}")
    
    return True


if __name__ == "__main__":
    success = create_dir_structure()
    sys.exit(0 if success else 1)
