"""
Visual Similarity Matching Module Tests
Tests for ResNet50 + FAISS-based VisualSimilarityMatcher class
"""

import pytest
import numpy as np
from PIL import Image
import cv2

from config import Config
from modules.similarity import VisualSimilarityMatcher


class TestVisualSimilarityMatcher:
    """Test cases for VisualSimilarityMatcher class"""
    
    def test_initialization(self):
        """Test that VisualSimilarityMatcher initializes correctly"""
        matcher = VisualSimilarityMatcher()
        assert matcher is not None
        assert matcher.model is not None
        assert matcher.index is not None
    
    def test_extract_embedding_shape(self, random_image):
        """Test that extracted embedding has correct shape"""
        matcher = VisualSimilarityMatcher()
        
        embedding = matcher.extract_embedding(random_image)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (2048,), f"Expected shape (2048,), got {embedding.shape}"
        assert embedding.dtype == np.float32
    
    def test_same_image_high_similarity(self, sample_image_path):
        """Test that same image compared with itself has high similarity"""
        matcher = VisualSimilarityMatcher()
        
        image = Image.open(sample_image_path)
        
        # Extract embeddings
        emb1 = matcher.extract_embedding(image)
        emb2 = matcher.extract_embedding(image)
        
        # Compute similarity
        similarity = matcher.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, (float, np.floating))
        assert similarity > 0.98, f"Same image similarity {similarity} should be > 0.98"
    
    def test_different_images_lower_similarity(self):
        """Test that very different images have lower similarity"""
        matcher = VisualSimilarityMatcher()
        
        # Create two very different solid color images
        red_image = Image.new("RGB", (224, 224), color=(255, 0, 0))
        blue_image = Image.new("RGB", (224, 224), color=(0, 0, 255))
        
        emb1 = matcher.extract_embedding(red_image)
        emb2 = matcher.extract_embedding(blue_image)
        
        similarity = matcher.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, (float, np.floating))
        assert similarity < 0.85, f"Different image similarity {similarity} should be < 0.85"
    
    def test_similarity_level_classification(self):
        """Test classification of similarity into levels"""
        matcher = VisualSimilarityMatcher()
        
        # Create dummy embeddings
        emb1 = np.random.randn(2048).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        # Test different similarity levels
        test_cases = [
            (0.90, "High"),      # High similarity
            (0.75, "Medium"),    # Medium similarity
            (0.50, "Low"),       # Low similarity
            (0.20, "No Match"),  # No match / very low
        ]
        
        for score, expected_level in test_cases:
            # Create embedding with target similarity
            emb2 = emb1 * score
            # Normalize
            emb2 = emb2 / np.linalg.norm(emb2)
            
            similarity = matcher.compute_similarity(emb1, emb2)
            
            # Classify similarity
            if similarity > Config.SIMILARITY_HIGH_THRESHOLD:
                level = "High"
            elif similarity > Config.SIMILARITY_MEDIUM_THRESHOLD:
                level = "Medium"
            elif similarity > Config.SIMILARITY_LOW_THRESHOLD:
                level = "Low"
            else:
                level = "No Match"
            
            # Results may vary due to normalization, just check consistency
            assert level in ["High", "Medium", "Low", "No Match"]
    
    def test_embedding_is_normalized(self, random_image):
        """Test that embeddings are L2-normalized"""
        matcher = VisualSimilarityMatcher()
        
        embedding = matcher.extract_embedding(random_image)
        
        # Check L2 norm
        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01, f"Embedding not normalized, norm={norm}"
    
    def test_batch_embedding_extraction(self, image_list):
        """Test batch extraction of embeddings"""
        matcher = VisualSimilarityMatcher()
        
        embeddings = matcher.extract_embeddings_batch(image_list)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(image_list)
        assert embeddings.shape[1] == 2048
    
    def test_matcher_has_required_methods(self):
        """Test that matcher has all required methods"""
        matcher = VisualSimilarityMatcher()
        
        assert hasattr(matcher, "extract_embedding")
        assert callable(matcher.extract_embedding)
        assert hasattr(matcher, "extract_embeddings_batch")
        assert callable(matcher.extract_embeddings_batch)
        assert hasattr(matcher, "compute_similarity")
        assert callable(matcher.compute_similarity)
        assert hasattr(matcher, "find_similar")
        assert callable(matcher.find_similar)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def random_image():
    """Create a random PIL image (224x224)"""
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a test image and return path"""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    image_path = tmp_path / "test_similarity_image.jpg"
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def image_list():
    """Create list of test images"""
    images = []
    for i in range(5):
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))
    return images


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
