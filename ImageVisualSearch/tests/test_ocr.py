"""
Test suite for OCR Engine module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Mock tests for demonstration
class TestOCREngine:
    """Tests for OCREngine class"""
    
    @pytest.fixture
    def sample_image_with_text(self):
        """Create a sample image with text"""
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Hello World", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 0), 2)
        return img
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine instance for testing"""
        try:
            from modules.ocr_engine import OCREngine
            return OCREngine(language="eng")
        except Exception:
            pytest.skip("OCR dependencies not available")
    
    def test_ocr_initialization(self, ocr_engine):
        """Test OCR engine initialization"""
        assert ocr_engine is not None
        assert ocr_engine.language == "eng"
    
    def test_extract_text_returns_string(self, ocr_engine, sample_image_with_text):
        """Test that extract_text returns a string"""
        result = ocr_engine.extract_text(sample_image_with_text)
        assert isinstance(result, str)
    
    def test_extract_text_with_confidence(self, ocr_engine, sample_image_with_text):
        """Test text extraction with confidence"""
        text, confidence = ocr_engine.extract_text_with_confidence(sample_image_with_text)
        
        assert isinstance(text, str)
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
    
    def test_detect_text_regions(self, ocr_engine, sample_image_with_text):
        """Test text region detection"""
        regions = ocr_engine.detect_text_regions(sample_image_with_text)
        
        assert isinstance(regions, list)
        for region in regions:
            assert 'text' in region
            assert 'x' in region
            assert 'y' in region
            assert 'w' in region
            assert 'h' in region
            assert 'confidence' in region
    
    def test_supported_languages(self, ocr_engine):
        """Test getting supported languages"""
        languages = ocr_engine.get_supported_languages()
        
        assert isinstance(languages, list)
        assert 'eng' in languages
        assert len(languages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
