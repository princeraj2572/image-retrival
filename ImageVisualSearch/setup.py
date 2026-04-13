"""
Setup configuration for ImageVisualSearch package
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="image-visual-search",
    version="1.0.0",
    description="Image-Based Visual Search and Information Retrieval System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/image-visual-search",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="computer-vision image-retrieval search object-detection ocr",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/image-visual-search/issues",
        "Source": "https://github.com/yourusername/image-visual-search",
        "Documentation": "https://github.com/yourusername/image-visual-search#readme",
    },
)
