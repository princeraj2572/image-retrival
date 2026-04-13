"""
Image Preprocessing Module
Provides comprehensive image preprocessing and augmentation for computer vision tasks
"""

import logging
from typing import Tuple, Optional, List, Dict
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

try:
    import albumentations as A
except ImportError:
    A = None

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Comprehensive image preprocessing and augmentation class
    Handles validation, preprocessing for different models, enhancement, and augmentation
    
    Attributes:
        target_size: Target image size for YOLO (height, width)
        normalize: Whether to normalize pixel values
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640), normalize: bool = True):
        """
        Initialize ImagePreprocessor
        
        Args:
            target_size: Target size for resizing (height, width). Default: (640, 640)
            normalize: Whether to normalize pixel values to [0, 1]. Default: True
        """
        self.target_size = target_size
        self.normalize = normalize
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate if a file is a valid image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if valid image, False otherwise
        """
        try:
            path = Path(image_path)
            
            # Check file exists
            if not path.exists():
                logger.warning(f"File not found: {image_path}")
                return False
            
            # Check file extension
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported format: {image_path}")
                return False
            
            # Try to open with PIL to validate
            with Image.open(image_path) as img:
                img.convert('RGB')
            
            # Verify with OpenCV
            cv2_img = cv2.imread(image_path)
            if cv2_img is None:
                logger.warning(f"Failed to read with OpenCV: {image_path}")
                return False
            
            logger.debug(f"Image validated: {image_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {str(e)}")
            return False
    
    def preprocess_for_yolo(self, image_path: str) -> np.ndarray:
        """
        Preprocess image specifically for YOLO object detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image (640x640, normalized to [0, 1])
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Resize to YOLO input size using INTER_LINEAR
            resized = cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                               interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 1]
            normalized = resized.astype('float32') / 255.0
            
            logger.debug(f"YOLO preprocessing complete: {image_path} -> shape {normalized.shape}")
            return normalized
            
        except Exception as e:
            logger.error(f"YOLO preprocessing failed for {image_path}: {str(e)}")
            raise
    
    def preprocess_for_ocr(self, image: np.ndarray, text_region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Preprocess image specifically for OCR (Tesseract)
        
        Args:
            image: Input image (BGR format from OpenCV)
            text_region: Optional tuple (x, y, w, h) to crop text region
            
        Returns:
            np.ndarray: Preprocessed image optimized for OCR
        """
        try:
            # Convert to uint8 if float
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            
            # Crop region if specified
            if text_region is not None:
                x, y, w, h = text_region
                image = image[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply Gaussian denoising
            denoised = cv2.fastNlMeansDenoising(equalized, None, h=10, 
                                              templateWindowSize=7, 
                                              searchWindowSize=21)
            
            # Upscale if image is too small
            height, width = denoised.shape
            if height < 50 or width < 50:
                scale = 2
                denoised = cv2.resize(denoised, 
                                    (width * scale, height * scale),
                                    interpolation=cv2.INTER_CUBIC)
            
            # Apply adaptive thresholding for better OCR
            processed = cv2.adaptiveThreshold(denoised, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            
            logger.debug(f"OCR preprocessing complete: shape {processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"OCR preprocessing failed: {str(e)}")
            raise
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using CLAHE and color space techniques
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            np.ndarray: Enhanced image
        """
        try:
            # Convert to uint8 if needed
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            
            # Apply Gaussian blur for noise reduction (kernel 3x3)
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel only
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            logger.debug("Image quality enhancement complete")
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {str(e)}")
            raise
    
    def augment_dataset(self, image: np.ndarray, augmentation_level: str = "medium") -> List[np.ndarray]:
        """
        Generate augmented versions of an image using Albumentations
        
        Args:
            image: Input image (BGR format)
            augmentation_level: Level of augmentation ("light", "medium", "heavy")
            
        Returns:
            List[np.ndarray]: List of 4 augmented images
        """
        try:
            if A is None:
                logger.warning("Albumentations not installed. Returning original image.")
                return [image]
            
            # Define augmentation pipeline
            if augmentation_level == "light":
                transform = A.Compose([
                    A.HorizontalFlip(p=0.3),
                    A.RandomBrightnessContrast(p=0.2),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            elif augmentation_level == "medium":
                transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.RandomScale(p=0.3),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            else:  # heavy
                transform = A.Compose([
                    A.HorizontalFlip(p=0.7),
                    A.RandomRotate90(p=0.7),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomScale(p=0.5),
                    A.Rotate(limit=45, p=0.5),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            
            # Generate augmented versions
            augmented_images = []
            for _ in range(4):
                augmented = transform(image=image, bboxes=[], labels=[])['image']
                augmented_images.append(augmented)
            
            logger.debug(f"Generated {len(augmented_images)} augmented images")
            return augmented_images
            
        except Exception as e:
            logger.error(f"Dataset augmentation failed: {str(e)}")
            raise
    
    def load_and_validate_dataset(self, dataset_dir: str) -> Dict[str, any]:
        """
        Load and validate all images in a dataset directory
        
        Args:
            dataset_dir: Path to dataset directory
            
        Returns:
            Dict with keys: "valid" (list of valid paths), "invalid" (list of invalid paths), "total" (int)
        """
        try:
            dataset_path = Path(dataset_dir)
            if not dataset_path.exists():
                logger.error(f"Dataset directory not found: {dataset_dir}")
                return {"valid": [], "invalid": [], "total": 0}
            
            valid_images = []
            invalid_images = []
            
            # Search for all image files
            for image_file in dataset_path.rglob('*'):
                if image_file.suffix.lower() in self.SUPPORTED_FORMATS:
                    if self.validate_image(str(image_file)):
                        valid_images.append(str(image_file))
                    else:
                        invalid_images.append(str(image_file))
            
            result = {
                "valid": valid_images,
                "invalid": invalid_images,
                "total": len(valid_images) + len(invalid_images)
            }
            
            logger.info(f"Dataset validation: {len(valid_images)} valid, {len(invalid_images)} invalid")
            return result
            
        except Exception as e:
            logger.error(f"Dataset loading failed: {str(e)}")
            raise
    
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
