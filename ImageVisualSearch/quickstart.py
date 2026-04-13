#!/usr/bin/env python3
"""
Quick Start Script for ImageVisualSearch
Sets up the project in a few simple steps
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)
    print()


def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"➜ {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"✓ {description} - Done!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - Failed!")
        return False


def main():
    """Main quick start routine"""
    
    print_header("ImageVisualSearch - Quick Start")
    
    project_root = Path(__file__).parent
    
    # Step 1: Create project structure
    print_header("Step 1: Creating Project Structure")
    
    from setup_project import create_dir_structure
    if create_dir_structure():
        print("✓ Project structure created successfully!")
    else:
        print("✗ Failed to create project structure")
        return False
    
    # Step 2: Verify installation
    print_header("Step 2: Verifying Installation")
    
    try:
        from verify_installation import verify_project_structure, verify_python_environment
        
        structure_ok = verify_project_structure()
        env_ok = verify_python_environment()
        
        if not structure_ok or not env_ok:
            print()
            print("⚠ Some verification checks failed")
            print("Please review the output above")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
    
    # Step 3: Installation instructions
    print_header("Step 3: Next Steps")
    
    print("Your project is now set up! Follow these steps to get started:")
    print()
    print("1. Create a Python virtual environment:")
    print("   • Using venv:")
    print("     Windows:   python -m venv venv && venv\\Scripts\\activate")
    print("     Linux/Mac: python3 -m venv venv && source venv/bin/activate")
    print()
    print("   • Using conda:")
    print("     conda env create -f environment.yml")
    print("     conda activate visual-search-env")
    print()
    print("2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Configure environment:")
    print("   • Copy .env.example to .env")
    print("   • Edit .env and update paths (especially TESSERACT_PATH)")
    print()
    print("4. Install Tesseract OCR (if needed):")
    print("   • Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   • Linux: sudo apt-get install tesseract-ocr")
    print("   • Mac: brew install tesseract")
    print()
    print("5. Test the installation:")
    print("   python verify_installation.py")
    print()
    print("6. Launch the web UI:")
    print("   python ui/app.py")
    print()
    print("7. Open your browser and go to:")
    print("   http://localhost:7860")
    print()
    
    print_header("Project Information")
    
    print(f"Project Root: {project_root}")
    print(f"Python Version: {sys.version}")
    print(f"Path: {sys.executable}")
    print()
    print("Key Files:")
    print(f"  • README.md - Main documentation")
    print(f"  • INSTALL.md - Installation guide")
    print(f"  • PROJECT_SUMMARY.md - Project summary")
    print(f"  • config.py - Configuration")
    print(f"  • requirements.txt - Dependencies")
    print()
    print("Available Make Commands:")
    print("  make help       - Show all commands")
    print("  make install    - Install dependencies")
    print("  make test       - Run tests")
    print("  make run-ui     - Launch web UI")
    print("  make verify     - Verify installation")
    print()
    
    print_header("Setup Complete!")
    
    print("✓ ImageVisualSearch is ready to use!")
    print()
    print("For more information, see:")
    print("  • README.md for project overview")
    print("  • INSTALL.md for detailed installation")
    print("  • PROJECT_SUMMARY.md for project structure")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
