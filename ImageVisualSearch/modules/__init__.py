"""
ImageVisualSearch modules package
Contains core functionality for detection, OCR, similarity, and retrieval
"""

from .detection import ObjectDetector
from .ocr_engine import OCREngine
from .similarity import SimilarityMatcher
from .retrieval import ImageRetriever

__all__ = [
    "ObjectDetector",
    "OCREngine",
    "SimilarityMatcher",
    "ImageRetriever",
]

__version__ = "1.0.0"
