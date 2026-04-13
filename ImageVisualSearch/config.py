"""
Configuration module for ImageVisualSearch
Loads all settings using python-dotenv and provides centralized configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Centralized configuration class for ImageVisualSearch project
    Manages paths, model settings, thresholds, and API keys
    """
    
    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    REFERENCE_DB_DIR = DATA_DIR / "reference_db"
    TEST_DIR = DATA_DIR / "test_images"
    MODEL_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"
    RESULTS_DIR = OUTPUT_DIR / "results"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    LOGS_DIR = BASE_DIR / "logs"
    
    # ==================== TESSERACT OCR ====================
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
    
    # ==================== YOLO OBJECT DETECTION ====================
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    YOLO_IMG_SIZE = 640
    YOLO_CLASSES = [
        "person", "car", "building", "animal", 
        "food", "electronics", "furniture", "plant"
    ]
    
    # ==================== IMAGE SIMILARITY & EMBEDDINGS ====================
    SIMILARITY_HIGH = float(os.getenv("SIMILARITY_THRESHOLD_HIGH", 0.85))
    SIMILARITY_MEDIUM = float(os.getenv("SIMILARITY_THRESHOLD_MEDIUM", 0.65))
    SIMILARITY_LOW = float(os.getenv("SIMILARITY_THRESHOLD_LOW", 0.40))
    EMBEDDING_DIM = 2048
    RESNET_MODEL = "resnet50"
    FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"
    
    # ==================== SEARCH API ====================
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")
    MAX_SEARCH_RESULTS = 5
    
    # ==================== DEVICE CONFIGURATION ====================
    DEVICE = os.getenv("DEVICE", "cpu")
    USE_GPU = DEVICE.lower() == "cuda"
    
    # ==================== DATASET & TRAINING ====================
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42
    
    # ==================== PREPROCESSING ====================
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    MAX_IMAGE_SIZE = 2048
    MIN_IMAGE_SIZE = 50
    
    # ==================== LOGGING ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "app.log"
    
    # ==================== UI/GRADIO ====================
    UI_PORT = int(os.getenv("UI_PORT", 7860))
    UI_SHARE = os.getenv("UI_SHARE", "False").lower() == "true"
    
    @classmethod
    def create_dirs(cls) -> None:
        """
        Create all project directories if they don't exist
        
        Returns:
            None
        """
        dirs = [
            cls.RAW_DIR, 
            cls.PROCESSED_DIR, 
            cls.REFERENCE_DB_DIR,
            cls.TEST_DIR, 
            cls.MODEL_DIR, 
            cls.RESULTS_DIR, 
            cls.REPORTS_DIR,
            cls.LOGS_DIR,
            cls.FEATURE_CACHE_DIR
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_all_settings(cls) -> dict:
        """
        Get all configuration settings as a dictionary
        
        Returns:
            dict: All configuration key-value pairs
        """
        return {
            key: getattr(cls, key) 
            for key in dir(cls) 
            if not key.startswith('_') and key.isupper()
        }
    
    @classmethod
    def to_dict(cls) -> dict:
        """
        Convert configuration to dictionary format
        Useful for logging and debugging
        
        Returns:
            dict: Configuration as dictionary
        """
        config_dict = {}
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                value = getattr(cls, key)
                config_dict[key] = str(value)
        return config_dict


# Initialize directories on module import
Config.create_dirs()
