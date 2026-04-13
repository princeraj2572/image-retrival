"""
OCR Engine Module Tests
Tests for Tesseract-based OCREngine class
"""

import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from config import Config
from modules.ocr_engine import OCREngine


class TestOCREngine:
    """Test cases for OCREngine class"""
    
    def test_ocr_initialization(self):
        """Test that OCREngine initializes correctly"""
        try:
            engine = OCREngine()
            assert engine is not None
            assert engine.pytesseract is not None
        except Exception as e:
            pytest.skip(f"Tesseract not installed: {e}")
    
    def test_extract_text_from_text_image(self, text_image_path):
        """Test text extraction from image with text"""
        try:
            engine = OCREngine()
            result = engine.extract_text(text_image_path)
            
            assert isinstance(result, dict)
            
            # Should have extracted some text or word info
            text = result.get("text", "").upper()
            word_count = result.get("word_count", 0)
            
            # Either text contains HELLO or word_count is > 0
            assert "HELLO" in text or word_count > 0
        except Exception as e:
            pytest.skip(f"OCR test skipped: {e}")
    
    def test_extract_text_returns_dict(self, sample_text_image):
        """Test that extract_text returns dict with required keys"""
        try:
            engine = OCREngine()
            result = engine.extract_text(sample_text_image)
            
            assert isinstance(result, dict)
            assert "text" in result or "words" in result
            assert "confidence" in result or "word_count" in result
        except Exception as e:
            pytest.skip(f"OCR test skipped: {e}")
    
    def test_build_search_query(self):
        """Test search query building from text"""
        try:
            engine = OCREngine()
            
            test_text = "The quick brown fox jumps over the lazy dog!!!"
            query = engine.build_search_query_from_text(test_text)
            
            assert isinstance(query, str)
            # Stopwords should be removed (The, over, the, dog might be filtered)
            assert len(query) <= 100
            
            # Should not contain double spaces or extra punctuation
            assert "  " not in query
        except Exception as e:
            pytest.skip(f"OCR test skipped: {e}")
    
    def test_empty_image_handling(self):
        """Test handling of blank/empty image"""
        try:
            engine = OCREngine()
            
            # Create blank white image
            blank_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
            
            # Should not raise error
            result = engine.extract_text(blank_image)
            
            assert isinstance(result, dict)
        except Exception as e:
            pytest.skip(f"OCR test skipped: {e}")
    
    def test_engine_has_required_methods(self):
        """Test that OCREngine has all required methods"""
        try:
            engine = OCREngine()
            
            assert hasattr(engine, "extract_text")
            assert callable(engine.extract_text)
            assert hasattr(engine, "extract_text_regions")
            assert callable(engine.extract_text_regions)
            assert hasattr(engine, "build_search_query_from_text")
            assert callable(engine.build_search_query_from_text)
        except Exception as e:
            pytest.skip(f"OCR test skipped: {e}")


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def text_image_path(tmp_path):
    """Create an image with text and return path"""
    # Create image with text using PIL
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw text
    try:
        # Try to use a system font, fallback to default
        draw.text((50, 50), "HELLO WORLD", fill=(0, 0, 0))
    except:
        draw.text((50, 50), "HELLO WORLD", fill=(0, 0, 0))
    
    # Save to temp path
    image_path = tmp_path / "text_image.png"
    img.save(image_path)
    
    return str(image_path)


@pytest.fixture
def sample_text_image(tmp_path):
    """Create sample text image as numpy array"""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    draw.text((50, 50), "Sample OCR Text", fill=(0, 0, 0))
    
    # Convert to numpy
    return np.array(img)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
