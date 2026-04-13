# ImageVisualSearch

A comprehensive Python computer vision project for image-based visual search and information retrieval using deep learning models.

## Overview

ImageVisualSearch is a complete system that combines multiple computer vision techniques to enable intelligent image searching, object detection, text recognition, and similarity matching. It leverages state-of-the-art models including YOLOv8 for object detection, ResNet for feature extraction, and Tesseract for OCR.

## Features

✓ **Object Detection** - YOLOv8-based object detection  
✓ **OCR Engine** - Text extraction from images using Tesseract  
✓ **Image Similarity** - Deep learning-based image similarity matching  
✓ **Image Retrieval** - Efficient image retrieval from custom databases  
✓ **Web UI** - Gradio-based interactive web interface  
✓ **Modular Architecture** - Clean, extensible codebase structure  

## Project Structure

```
ImageVisualSearch/
├── data/                    # Data storage directory
│   ├── raw/                # Raw input images
│   ├── processed/          # Processed images
│   ├── reference_db/       # Reference image database
│   └── test_images/        # Test data
├── models/                 # Pre-trained models
│   ├── yolo/              # YOLO model weights
│   ├── resnet/            # ResNet model weights
│   └── ocr/               # OCR model data
├── modules/               # Core functionality
│   ├── detection.py       # Object detection module
│   ├── ocr_engine.py      # OCR functionality
│   ├── similarity.py      # Similarity matching
│   └── retrieval.py       # Image retrieval system
├── utils/                 # Utility functions
│   ├── preprocessing.py   # Image preprocessing
│   └── helpers.py         # Helper functions
├── outputs/               # Output storage
│   ├── results/          # Detection results
│   └── reports/          # Analysis reports
├── ui/                    # Web interface
│   └── app.py            # Gradio application
├── tests/                # Test suite
├── config.py             # Configuration
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── README.md            # This file
├── .env.example         # Environment variables template
└── environment.yml      # Conda environment file
```

## Installation

### Quick Start (3 steps)

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/image-visual-search.git
cd ImageVisualSearch
```

2. **Create environment (Choose one method):**

**Option A - Conda:**
```bash
conda env create -f environment.yml
conda activate visual-search-env
```

**Option B - venv:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Detailed Installation

For detailed installation instructions including OS-specific setup and Tesseract OCR installation, see [INSTALL.md](INSTALL.md).

## Configuration

1. **Copy environment template:**
```bash
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

2. **Edit `.env` and configure paths/API keys:**
```
TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe
SEARCH_API_KEY=your_api_key
DEVICE=cpu  # or cuda for GPU acceleration
```

## Usage

### Basic Python Usage

```python
from modules import ObjectDetector, OCREngine, SimilarityMatcher
from utils import ImagePreprocessor

# Initialize modules
detector = ObjectDetector()
ocr = OCREngine()
matcher = SimilarityMatcher()

# Process an image
image_path = "data/test_images/sample.jpg"
preprocessor = ImagePreprocessor()
processed_img = preprocessor.preprocess(image_path)

# Run detection
detections = detector.detect(processed_img)

# Extract text
text = ocr.extract_text(processed_img)

# Find similar images
similar_images = matcher.find_similar(image_path, top_k=5)
```

### Using Web UI

```bash
python ui/app.py
```

Then open `http://localhost:7860` in your browser.

### Command Line

```bash
python main.py --image data/test_images/sample.jpg --task detect
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_detection.py -v

# Run with coverage
pytest tests/ --cov=modules --cov-report=html
```

## Key Dependencies

- **torch 2.1.0** - Deep learning framework
- **ultralytics 8.0.196** - YOLOv8 implementation
- **opencv-python 4.8.1.78** - Computer vision library
- **pytesseract 0.3.10** - OCR interface
- **faiss-cpu 1.7.4** - Similarity search
- **gradio 4.7.1** - Web UI framework
- **scikit-learn 1.3.2** - ML utilities

## System Requirements

- **Python**: 3.8+
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional (NVIDIA CUDA 11.8+)
- **Storage**: 10GB for models and data

## Configuration Options

Key configuration parameters in `config.py`:

```python
CONFIDENCE_THRESHOLD = 0.5          # Detection confidence
SIMILARITY_THRESHOLD_HIGH = 0.85    # High similarity threshold
SIMILARITY_THRESHOLD_MEDIUM = 0.65  # Medium similarity threshold
SIMILARITY_THRESHOLD_LOW = 0.40     # Low similarity threshold
DEVICE = "cpu"                      # cpu or cuda
YOLO_MODEL = "yolov8n.pt"          # Model size
```

## Performance Tips

1. **GPU Acceleration**: Set `DEVICE=cuda` in `.env` for faster processing
2. **Model Selection**: Use `yolov8n` for speed, `yolov8l` for accuracy
3. **Batch Processing**: Process multiple images together for efficiency
4. **Feature Caching**: Cache image features to avoid recomputation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tesseract not found | Install Tesseract and set `TESSERACT_PATH` in `.env` |
| CUDA out of memory | Reduce batch size or switch to `DEVICE=cpu` |
| Import errors | Run `pip install -r requirements.txt` |
| Model download fails | Check internet connection and disk space |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- ResNet by Microsoft Research
- Tesseract OCR by Google
- FAISS by Facebook Research
- Gradio by HuggingFace

## Contact

For questions and support:
- **Email**: your.email@example.com
- **GitHub Issues**: [Project Issues](https://github.com/yourusername/image-visual-search/issues)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{imagevisualsearch2024,
  title={ImageVisualSearch: Image-Based Visual Search and Information Retrieval System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/image-visual-search}
}
```

---

**Last Updated**: 2024  
**Version**: 1.0.0
