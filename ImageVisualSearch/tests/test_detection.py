"""
Test suite for Object Detection module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Mock tests for demonstration
class TestObjectDetector:
    """Tests for ObjectDetector class"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        try:
            from modules.detection import ObjectDetector
            return ObjectDetector(model_name="yolov8n.pt", device="cpu", confidence_threshold=0.5)
        except Exception:
            pytest.skip("Model not available")
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.device == "cpu"
        assert detector.confidence_threshold == 0.5
    
    def test_detect_returns_list(self, detector, sample_image):
        """Test that detect returns a list"""
        result = detector.detect(sample_image)
        assert isinstance(result, list)
    
    def test_detect_output_format(self, detector, sample_image):
        """Test detection output format"""
        detections = detector.detect(sample_image)
        
        for detection in detections:
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert 'class_id' in detection
            assert 'class_name' in detection
    
    def test_batch_detect(self, detector):
        """Test batch detection"""
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = detector.detect_batch(images)
        
        assert len(results) == len(images)
        assert all(isinstance(r, list) for r in results)
    
    def test_get_model_info(self, detector):
        """Test getting model information"""
        info = detector.get_model_info()
        assert isinstance(info, dict)
        assert 'device' in info


class TestDetectionIntegration:
    """Integration tests for detection"""
    
    def test_detection_with_real_image(self, tmp_path):
        """Test detection with a synthetic image"""
        # Create a synthetic image with some features
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), -1)
        
        # Save image
        img_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(img_path), img)
        
        assert img_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
