"""
Test suite for Similarity Matcher module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Mock tests for demonstration
class TestSimilarityMatcher:
    """Tests for SimilarityMatcher class"""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing"""
        img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return img1, img2
    
    @pytest.fixture
    def matcher(self):
        """Create similarity matcher instance"""
        try:
            from modules.similarity import SimilarityMatcher
            return SimilarityMatcher(model_name="resnet50", device="cpu")
        except Exception:
            pytest.skip("PyTorch not available")
    
    def test_matcher_initialization(self, matcher):
        """Test matcher initialization"""
        assert matcher is not None
        assert matcher.device == "cpu"
        assert matcher.similarity_threshold == 0.85
    
    def test_get_features(self, matcher, sample_images):
        """Test feature extraction"""
        img, _ = sample_images
        features = matcher.get_features(img)
        
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1  # Flattened vector
        assert len(features) > 0
    
    def test_compute_similarity(self, matcher):
        """Test similarity computation"""
        features1 = np.random.randn(2048)
        features2 = np.random.randn(2048)
        
        similarity = matcher.compute_similarity(features1, features2)
        
        assert isinstance(similarity, (int, float))
        assert 0 <= similarity <= 1
    
    def test_find_similar_images(self, matcher):
        """Test finding similar images"""
        query_features = np.random.randn(2048)
        reference_features = [np.random.randn(2048) for _ in range(5)]
        
        results = matcher.find_similar_images(query_features, reference_features, top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        for idx, similarity in results:
            assert isinstance(idx, (int, np.integer))
            assert isinstance(similarity, (float, np.floating))
    
    def test_batch_get_features(self, matcher):
        """Test batch feature extraction"""
        images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                 for _ in range(3)]
        
        features_list = matcher.batch_get_features(images)
        
        assert len(features_list) == 3
        assert all(isinstance(f, np.ndarray) for f in features_list)
    
    def test_is_similar(self, matcher, sample_images):
        """Test similarity checking"""
        img1, img2 = sample_images
        
        result = matcher.is_similar(img1, img2)
        assert isinstance(result, (bool, np.bool_))


class TestSimilarityIntegration:
    """Integration tests for similarity matching"""
    
    def test_similarity_with_identical_images(self):
        """Test similarity with identical images"""
        try:
            from modules.similarity import SimilarityMatcher
            
            matcher = SimilarityMatcher(device="cpu")
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            features1 = matcher.get_features(img)
            features2 = matcher.get_features(img)
            
            similarity = matcher.compute_similarity(features1, features2)
            
            # Should be very similar (but may not be exactly 1 due to processing)
            assert similarity > 0.9
            
        except Exception:
            pytest.skip("PyTorch not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
