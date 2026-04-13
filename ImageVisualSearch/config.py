"""
Configuration module for ImageVisualSearch
Contains global configuration parameters and paths
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DB_DIR = DATA_DIR / "reference_db"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_DIR = MODELS_DIR / "yolo"
RESNET_MODEL_DIR = MODELS_DIR / "resnet"
OCR_MODEL_DIR = MODELS_DIR / "ocr"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REFERENCE_DB_DIR, TEST_IMAGES_DIR,
                  YOLO_MODEL_DIR, RESNET_MODEL_DIR, OCR_MODEL_DIR, RESULTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Detection and Matching Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
SIMILARITY_THRESHOLD_HIGH = float(os.getenv("SIMILARITY_THRESHOLD_HIGH", 0.85))
SIMILARITY_THRESHOLD_MEDIUM = float(os.getenv("SIMILARITY_THRESHOLD_MEDIUM", 0.65))
SIMILARITY_THRESHOLD_LOW = float(os.getenv("SIMILARITY_THRESHOLD_LOW", 0.40))

# Model Configuration
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")

# Device Configuration (cpu or cuda)
DEVICE = os.getenv("DEVICE", "cpu")

# Tesseract Configuration
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

# API Configuration
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = os.getenv("LOG_FILE", str(LOG_DIR / "app.log"))

# Gradio UI Configuration
UI_PORT = int(os.getenv("UI_PORT", 7860))
UI_SHARE = os.getenv("UI_SHARE", "False").lower() == "true"

# Image Processing Configuration
IMAGE_SIZE = 640
BATCH_SIZE = 32
NUM_WORKERS = 4

print(f"Configuration loaded from: {PROJECT_ROOT}")
print(f"Device: {DEVICE}")
