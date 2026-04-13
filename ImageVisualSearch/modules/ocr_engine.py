"""
OCR (Optical Character Recognition) Engine Module
Extracts text from images using Tesseract and PyTesseract
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR Engine for text extraction from images
    Uses Tesseract with pytesseract
    
    Attributes:
        tesseract_path (str): Path to Tesseract executable
        language (str): Language(s) to use for OCR
    """
    
    def __init__(self, tesseract_path: str = "", language: str = "eng"):
        """
        Initialize OCREngine
        
        Args:
            tesseract_path: Path to tesseract.exe (Windows) or tesseract (Linux/Mac)
            language: Language(s) for OCR (eng, fra, deu, etc.)
        """
        try:
            import pytesseract
            from pytesseract import Output
            
            if tesseract_path:
                pytesseract.pytesseract.pytesseract_cmd = tesseract_path
            
            self.pytesseract = pytesseract
            self.Output = Output
            self.language = language
            
            logger.info(f"OCREngine initialized with language: {language}")
            
        except ImportError:
            logger.error("pytesseract package required. Install with: pip install pytesseract")
            raise
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract all text from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Extracted text string
        """
        try:
            # Convert BGR to RGB for Tesseract
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract text
            text = self.pytesseract.image_to_string(rgb_image, lang=self.language)
            
            logger.debug(f"Extracted text length: {len(text)}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def extract_text_with_confidence(self, image: np.ndarray) -> tuple:
        """
        Extract text with confidence scores
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (text, confidence_score)
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get data with confidence
            data = self.pytesseract.image_to_data(
                rgb_image,
                lang=self.language,
                output_type=self.Output.DICT
            )
            
            # Extract text and average confidence
            text = " ".join([w for w in data['text'] if w.strip()])
            confidences = [int(c) for c in data['confidence'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return text.strip(), avg_confidence / 100.0
            
        except Exception as e:
            logger.error(f"Text extraction with confidence failed: {e}")
            raise
    
    def detect_text_regions(self, image: np.ndarray) -> List[dict]:
        """
        Detect text regions and bounding boxes
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of text regions with bounding boxes
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            data = self.pytesseract.image_to_data(
                rgb_image,
                lang=self.language,
                output_type=self.Output.DICT
            )
            
            regions = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    region = {
                        'text': data['text'][i],
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i],
                        'confidence': int(data['conf'][i]) / 100.0
                    }
                    regions.append(region)
            
            logger.debug(f"Detected {len(regions)} text regions")
            return regions
            
        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            raise
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, h=10, templateWindowSize=7,
                                           searchWindowSize=21)
        
        return denoised
    
    def extract_text_from_regions(self, image: np.ndarray, regions: List[dict]) -> List[str]:
        """
        Extract text from specific regions
        
        Args:
            image: Input image
            regions: List of region dictionaries with x, y, w, h
            
        Returns:
            List of extracted texts
        """
        texts = []
        
        for region in regions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            roi = image[y:y+h, x:x+w]
            
            text = self.extract_text(roi)
            texts.append(text)
        
        return texts
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [
            'eng', 'fra', 'deu', 'ita', 'spa', 'por', 'rus', 'jpn',
            'kor', 'chi_sim', 'chi_tra', 'ara', 'hin'
        ]
