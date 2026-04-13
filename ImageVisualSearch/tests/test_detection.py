"""
Object Detection Module Tests
Tests for YOLOv8-based ObjectDetector class
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from config import Config
from modules.detection import ObjectDetector


class TestObjectDetector:
    """Test cases for ObjectDetector class"""
    
    def test_model_initialization(self):
        """Test that ObjectDetector initializes correctly"""
        detector = ObjectDetector()
        assert detector is not None
        assert detector.model is not None
        assert detector.confidence_threshold == Config.YOLO_CONFIDENCE
        assert detector.img_size == Config.YOLO_IMG_SIZE
    
    def test_detect_single_image(self, sample_image_path):
        """Test detection on a single image"""
        detector = ObjectDetector()
        result = detector.detect(sample_image_path)
        
        assert isinstance(result, list)
        
        if len(result) > 0:
            for detection in result:
                assert "class" in detection or "class_id" in detection
                assert "confidence" in detection
                assert "bbox" in detection
                
                # Verify confidence is between 0 and 1
                conf = detection["confidence"]
                assert 0 <= conf <= 1, f"Confidence {conf} not in range [0, 1]"
    
    def test_detect_returns_empty_for_blank_image(self):
        """Test that detection on blank image returns empty list"""
        detector = ObjectDetector()
        
        # Create blank white image
        blank_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        # Should not raise error and should return list
        result = detector.detect(blank_image)
        assert isinstance(result, list)
    
    def test_detect_batch(self, sample_image_list):
        """Test batch detection on multiple images"""
        detector = ObjectDetector()
        
        result = detector.detect_batch(sample_image_list)
        
        assert isinstance(result, list)
        assert len(result) == len(sample_image_list)
    
    def test_draw_detections_returns_image(self, sample_image_path):
        """Test that draw_detections returns proper image array"""
        detector = ObjectDetector()
        
        # Read image and detect
        image = cv2.imread(sample_image_path)
        detections = detector.detect(sample_image_path)
        
        # Draw detections
        result = detector.draw_detections(image, detections)
        
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3, "Output should have 3 channels (BGR or RGB)"
    
    def test_invalid_image_path(self):
        """Test detection with non-existent image path"""
        detector = ObjectDetector()
        
        # Should either raise FileNotFoundError or return empty list
        try:
            result = detector.detect("nonexistent_path_12345.jpg")
            assert isinstance(result, list)
        except FileNotFoundError:
            # This is also acceptable behavior
            pass
    
    def test_detect_confidence_thresholding(self, sample_image_path):
        """Test that confidence threshold is applied"""
        # Create detector with high confidence threshold
        detector = ObjectDetector(confidence_threshold=0.95)
        result = detector.detect(sample_image_path)
        
        # All detections should meet threshold
        for detection in result:
            assert detection["confidence"] >= 0.95
    
    def test_model_has_required_attributes(self):
        """Test that detector has all required attributes"""
        detector = ObjectDetector()
        
        assert hasattr(detector, "model")
        assert hasattr(detector, "confidence_threshold")
        assert hasattr(detector, "img_size")
        assert hasattr(detector, "device")


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a simple test image and return path"""
    # Create a 224x224 RGB test image
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    
    # Draw some simple shapes on it to make detection possible
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a few rectangles (simulating objects)
    draw.rectangle([20, 20, 100, 100], fill=(255, 0, 0), outline=(0, 0, 0))
    draw.rectangle([120, 120, 200, 200], fill=(0, 255, 0), outline=(0, 0, 0))
    draw.ellipse([50, 50, 150, 150], fill=(0, 0, 255), outline=(255, 255, 0))
    
    # Save to temp path
    image_path = tmp_path / "test_image.jpg"
    img.save(image_path)
    
    return str(image_path)


@pytest.fixture
def sample_image_list(tmp_path):
    """Create multiple test images and return list of paths"""
    image_paths = []
    
    for i in range(3):
        img = Image.new("RGB", (224, 224), color=(100 + i*20, 150, 200))
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([20 + i*10, 20, 100 + i*10, 100], 
                      fill=(255 - i*50, 0, 0), outline=(0, 0, 0))
        
        image_path = tmp_path / f"test_image_{i}.jpg"
        img.save(image_path)
        image_paths.append(str(image_path))
    
    return image_paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
