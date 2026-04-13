"""
Image Preprocessing Module
Provides functions for image preprocessing and augmentation
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing and augmentation
    
    Attributes:
        target_size: Target image size (height, width)
        normalize: Whether to normalize image
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640), normalize: bool = True):
        """
        Initialize ImagePreprocessor
        
        Args:
            target_size: Target size for resizing (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Resize
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            
            # Normalize
            if self.normalize:
                image = image.astype('float32') / 255.0
            
            logger.debug(f"Preprocessed image shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch of preprocessed images
        """
        batch = []
        for image_path in image_paths:
            image = self.preprocess(image_path)
            if image is not None:
                batch.append(image)
        
        return np.array(batch)
    
    def augment(self, image: np.ndarray, augmentation_type: str = "random") -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation (flip, rotate, brightness, contrast, random)
            
        Returns:
            Augmented image
        """
        if augmentation_type == "flip":
            image = cv2.flip(image, 1)
        elif augmentation_type == "rotate":
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
            image = cv2.warpAffine(image, M, (cols, rows))
        elif augmentation_type == "brightness":
            alpha = np.random.uniform(0.7, 1.3)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        elif augmentation_type == "contrast":
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.uniform(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        elif augmentation_type == "random":
            augmentations = ["flip", "rotate", "brightness", "contrast"]
            selected = np.random.choice(augmentations)
            image = self.augment(image, selected)
        
        return image
    
    def denoise(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Denoise an image
        
        Args:
            image: Input image
            method: Denoising method (bilateral, non_local_means, morphological)
            
        Returns:
            Denoised image
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        if method == "bilateral":
            result = cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "non_local_means":
            if len(image.shape) == 3:
                result = cv2.fastNlMeansDenoisingColored(image, None, h=10,
                                                         templateWindowSize=7,
                                                         searchWindowSize=21)
            else:
                result = cv2.fastNlMeansDenoising(image, None, h=10,
                                                 templateWindowSize=7,
                                                 searchWindowSize=21)
        elif method == "morphological":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        else:
            result = image
        
        return result
    
    def convert_color(self, image: np.ndarray, target_format: str = "RGB") -> np.ndarray:
        """
        Convert image color format
        
        Args:
            image: Input image (assumes BGR for OpenCV)
            target_format: Target format (RGB, HSV, GRAY, LAB)
            
        Returns:
            Image in target color format
        """
        if target_format == "RGB":
            result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif target_format == "HSV":
            result = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif target_format == "GRAY":
            result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif target_format == "LAB":
            result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            result = image
        
        return result
    
    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization
        
        Args:
            image: Input image
            
        Returns:
            Image with equalized histogram
        """
        if len(image.shape) == 3:
            # Convert to HSV and equalize V channel
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            result = cv2.equalizeHist(image)
        
        return result
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Get information about an image
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with image properties
        """
        image = cv2.imread(image_path)
        
        if image is None:
            return {}
        
        return {
            'shape': image.shape,
            'size_mb': Path(image_path).stat().st_size / (1024 * 1024),
            'dtype': str(image.dtype),
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'height': image.shape[0],
            'width': image.shape[1]
        }
