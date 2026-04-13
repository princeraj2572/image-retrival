"""
Image Similarity Matcher Module
Computes image similarity using deep learning features
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class SimilarityMatcher:
    """
    Image Similarity Matcher using ResNet features
    
    Attributes:
        model_name (str): Feature extraction model (resnet50, resnet101)
        device (str): Device to use (cpu or cuda)
        similarity_threshold (float): Threshold for considering images similar
    """
    
    def __init__(self, model_name: str = "resnet50", device: str = "cpu",
                 similarity_threshold: float = 0.85):
        """
        Initialize SimilarityMatcher
        
        Args:
            model_name: ResNet model to use
            device: Device to use (cpu or cuda)
            similarity_threshold: Similarity threshold (0-1)
        """
        try:
            import torchvision
            from torchvision import models, transforms
            import torch
            
            self.torch = torch
            self.device = device
            self.similarity_threshold = similarity_threshold
            
            # Load pre-trained ResNet model
            if model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
            else:
                self.model = models.resnet101(pretrained=True)
            
            # Remove classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(device)
            self.model.eval()
            
            # Image preprocessing
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info(f"SimilarityMatcher initialized with {model_name} on {device}")
            
        except ImportError:
            logger.error("torch and torchvision required. Install with: pip install torch torchvision")
            raise
    
    def get_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Feature vector (flattened)
        """
        try:
            from PIL import Image
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Preprocess
            tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with self.torch.no_grad():
                features = self.model(tensor)
            
            # Flatten and normalize
            features = features.squeeze().cpu().numpy()
            features = features / (np.linalg.norm(features) + 1e-8)
            
            logger.debug(f"Extracted feature vector with shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1)
        """
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        magnitude1 = np.linalg.norm(features1)
        magnitude2 = np.linalg.norm(features2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0, min(1, (similarity + 1) / 2))  # Normalize to [0, 1]
    
    def find_similar_images(self, query_features: np.ndarray,
                           reference_features: List[np.ndarray],
                           top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find similar images from reference set
        
        Args:
            query_features: Feature vector of query image
            reference_features: List of reference feature vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for idx, ref_features in enumerate(reference_features):
            similarity = self.compute_similarity(query_features, ref_features)
            similarities.append((idx, similarity))
        
        # Sort by similarity scores
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def batch_get_features(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features from multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of feature vectors
        """
        features_list = []
        for image in images:
            features = self.get_features(image)
            features_list.append(features)
        return features_list
    
    def is_similar(self, image1: np.ndarray, image2: np.ndarray) -> bool:
        """
        Check if two images are similar based on threshold
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            True if images are similar, False otherwise
        """
        features1 = self.get_features(image1)
        features2 = self.get_features(image2)
        
        similarity = self.compute_similarity(features1, features2)
        return similarity >= self.similarity_threshold
    
    def visualize_similarity(self, image1: np.ndarray, image2: np.ndarray,
                            output_path: Optional[str] = None):
        """
        Visualize two images side by side with similarity score
        
        Args:
            image1: First image
            image2: Second image
            output_path: Optional path to save visualization
        """
        features1 = self.get_features(image1)
        features2 = self.get_features(image2)
        similarity = self.compute_similarity(features1, features2)
        
        # Resize images to same height for side-by-side display
        h = min(image1.shape[0], image2.shape[0])
        w1 = int(image1.shape[1] * h / image1.shape[0])
        w2 = int(image2.shape[1] * h / image2.shape[0])
        
        img1_resized = cv2.resize(image1, (w1, h))
        img2_resized = cv2.resize(image2, (w2, h))
        
        # Concatenate images
        combined = np.hstack([img1_resized, img2_resized])
        
        # Add text
        text = f"Similarity: {similarity:.2%}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0) if similarity > self.similarity_threshold else (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, combined)
            logger.info(f"Similarity visualization saved to {output_path}")
        
        return combined
