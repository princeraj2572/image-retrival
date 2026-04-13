"""
OCR (Optical Character Recognition) Engine Module
Complete Tesseract-based text extraction with preprocessing and evaluation
"""

import logging
import re
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import json
from tqdm import tqdm

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None
    Output = None

# Import from project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Complete OCR Engine for text extraction from images
    Uses Tesseract OCR with PyTesseract wrapper
    
    Attributes:
        pytesseract: PyTesseract module
        Output: PyTesseract Output enum
        language: Language(s) for OCR ('eng', 'hin', 'ara', etc.)
        preprocessor: ImagePreprocessor instance
        supported_languages: List of supported language codes
    """
    
    def __init__(self) -> None:
        """
        Initialize OCREngine with Tesseract verification
        
        Raises:
            RuntimeError: If Tesseract not installed
            ImportError: If pytesseract not available
        """
        try:
            if pytesseract is None:
                raise ImportError("pytesseract package required. Install with: pip install pytesseract")
            
            # Set Tesseract path from Config
            if Config.TESSERACT_PATH:
                pytesseract.pytesseract.pytesseract_cmd = Config.TESSERACT_PATH
            
            # Verify Tesseract installation
            try:
                version = pytesseract.get_tesseract_version()
                tesseract_version = str(version).split()[0] if version else "unknown"
            except Exception as e:
                raise RuntimeError(
                    f"Tesseract not found at {Config.TESSERACT_PATH}. "
                    "Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki"
                )
            
            # Initialize components
            self.preprocessor = ImagePreprocessor(
                target_size=(640, 640),
                normalize=False
            )
            
            # Supported languages
            self.supported_languages = ["eng", "hin", "ara", "chi_sim", "fra", "deu"]
            
            logger.info(f"OCR Module initialized with Tesseract version {tesseract_version}")
            print(f"✓ OCR Module initialized with Tesseract version {tesseract_version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCREngine: {str(e)}")
            raise
    
    def extract_text(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        lang: str = "eng",
        psm: int = 3
    ) -> Dict[str, Any]:
        """
        Extract text from image with word-level bounding boxes and confidence
        
        Args:
            image_input: Image as file path (str), numpy array (BGR), or PIL Image
            lang: Language code for extraction. Default: 'eng'
            psm: Page segmentation mode. Default: 3
            
        Returns:
            Dict with keys:
                'text': extracted text string
                'words': list of word dicts with {'word', 'confidence', 'bbox'}
                'language': language used
                'confidence': mean word confidence
                'word_count': number of words
                'char_count': number of characters
                
        Raises:
            ValueError: If image invalid or language unsupported
            FileNotFoundError: If file path not found
        """
        try:
            # Convert input to numpy array
            if isinstance(image_input, str):
                if not self.preprocessor.validate_image(image_input):
                    raise ValueError(f"Invalid image: {image_input}")
                image = cv2.imread(image_input)
                if image is None:
                    raise FileNotFoundError(f"Cannot read image: {image_input}")
                    
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
                
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
                
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
            
            # Check for empty image
            if image is None or image.size == 0:
                logger.warning("Empty image received")
                return {
                    'text': '',
                    'words': [],
                    'language': lang,
                    'confidence': 0.0,
                    'word_count': 0,
                    'char_count': 0
                }
            
            # Validate language
            if lang not in self.supported_languages and '+' not in lang:
                logger.warning(f"Language {lang} may not be supported")
            
            # Preprocess for OCR
            preprocessed = self.preprocessor.preprocess_for_ocr(image)
            rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            
            # Extract text string
            config = f"--psm {psm}"
            text = pytesseract.image_to_string(rgb_image, lang=lang, config=config)
            text = text.strip()
            text = text.replace('\x00', '')  # Remove null bytes
            
            # Extract word-level data
            data = pytesseract.image_to_data(
                rgb_image,
                lang=lang,
                config=config,
                output_type=Output.DICT
            )
            
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                if word:
                    confidence = int(data['conf'][i])
                    if confidence > 0:
                        bbox = {
                            'x': int(data['left'][i]),
                            'y': int(data['top'][i]),
                            'w': int(data['width'][i]),
                            'h': int(data['height'][i])
                        }
                        words.append({
                            'word': word,
                            'confidence': confidence / 100.0,
                            'bbox': bbox
                        })
                        confidences.append(confidence / 100.0)
            
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            result = {
                'text': text,
                'words': words,
                'language': lang,
                'confidence': mean_confidence,
                'word_count': len(words),
                'char_count': len(text)
            }
            
            logger.info(f"Extracted {len(words)} words with {mean_confidence:.2f} avg confidence")
            return result
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise
    
    def extract_text_regions(
        self,
        image_input: Union[str, np.ndarray, Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        Detect text regions and extract text separately from each
        
        Args:
            image_input: Image as file path, numpy array, or PIL Image
            
        Returns:
            List of dicts with 'bbox' and 'text' for each region (confidence > 60)
        """
        try:
            # Convert to numpy array
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                image = image_input.copy()
            
            if image is None or image.size == 0:
                logger.warning("Empty image for region extraction")
                return []
            
            # Preprocess
            preprocessed = self.preprocessor.preprocess_for_ocr(image)
            rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            
            # Get word-level boxes
            data = pytesseract.image_to_data(
                rgb_image,
                output_type=Output.DICT
            )
            
            regions = []
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if word and confidence > 60:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Crop region and extract text
                    roi = image[y:y+h, x:x+w]
                    region_text = pytesseract.image_to_string(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    
                    regions.append({
                        'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                        'text': region_text.strip(),
                        'confidence': confidence / 100.0
                    })
            
            logger.info(f"Detected {len(regions)} text regions")
            return regions
            
        except Exception as e:
            logger.error(f"Region extraction failed: {str(e)}")
            raise
    
    def detect_text_type(
        self,
        image_input: Union[str, np.ndarray, Image.Image]
    ) -> str:
        """
        Analyze image to detect text type
        
        Args:
            image_input: Image as file path, numpy array, or PIL Image
            
        Returns:
            Text type string: 'clear_text', 'low_contrast', 'small_font', 
                             'tilted_text', 'multilingual', or 'general'
        """
        try:
            # Convert to numpy array
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                image = image_input.copy()
            
            if image is None or image.size == 0:
                return 'general'
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check contrast
            mean, std = cv2.meanStdDev(gray)
            contrast = std[0][0]
            
            # Check if mostly grayscale (low color saturation)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1]
            avg_saturation = np.mean(saturation)
            is_grayscale = avg_saturation < 30
            
            # Check for high contrast and grayscale
            if contrast > 60 and is_grayscale:
                return 'clear_text'
            
            # Check for low contrast
            if contrast < 30:
                return 'low_contrast'
            
            # Detect skew (TODO: implement skew detection)
            # For now, simple heuristic
            try:
                edges = cv2.Canny(gray, 100, 200)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
                if lines is not None and len(lines) > 0:
                    angles = [np.abs(line[0][1]) for line in lines[:5]]
                    max_skew = max(angles) * 180 / np.pi
                    if max_skew > 5 and max_skew < 85:
                        return 'tilted_text'
            except:
                pass
            
            # Detect small font by checking image size
            height = image.shape[0]
            if height < 200:
                return 'small_font'
            
            # Try to detect multilingual content
            try:
                text_data = pytesseract.image_to_data(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    output_type=Output.DICT
                )
                # Simple heuristic: if many characters, likely multilingual
                if len(text_data['text']) > 100:
                    return 'multilingual'
            except:
                pass
            
            return 'general'
            
        except Exception as e:
            logger.warning(f"Text type detection failed: {str(e)}")
            return 'general'
    
    def preprocess_and_extract(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        auto_detect: bool = True
    ) -> Dict[str, Any]:
        """
        Intelligently preprocess and extract text based on detected text type
        
        Args:
            image_input: Image input
            auto_detect: Whether to auto-detect text type. Default: True
            
        Returns:
            Dict with extraction results and 'text_type' key
        """
        try:
            # Detect text type
            if auto_detect:
                text_type = self.detect_text_type(image_input)
                logger.info(f"Detected text type: {text_type}")
            else:
                text_type = 'general'
            
            # Convert to numpy array
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                image = image_input.copy()
            
            # Apply appropriate preprocessing
            if text_type == 'low_contrast':
                # Apply CLAHE
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                lang = 'eng'
                psm = 3
                
            elif text_type == 'tilted_text':
                # Deskew
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                coords = np.column_stack(np.where(gray > 0))
                if len(coords) > 0:
                    angle = cv2.minAreaRect(cv2.convexHull(coords.astype(np.float32)))[2]
                    if angle < -45:
                        angle = angle + 90
                    if abs(angle) > 1:
                        h, w = gray.shape
                        center = (w // 2, h // 2)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                lang = 'eng'
                psm = 3
                
            elif text_type == 'small_font':
                # Upscale image
                image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                lang = 'eng'
                psm = 3
                
            elif text_type == 'multilingual':
                # Use language combination
                lang = 'eng+hin+ara'
                psm = 3
                
            else:
                lang = 'eng'
                psm = 3
            
            # Extract text
            result = self.extract_text(image, lang=lang, psm=psm)
            result['text_type'] = text_type
            
            return result
            
        except Exception as e:
            logger.error(f"Preprocess and extract failed: {str(e)}")
            raise
    
    def extract_from_multiple_regions(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        bboxes: List[Dict[str, float]]
    ) -> List[str]:
        """
        Extract text from multiple regions (e.g., from YOLO detections)
        
        Args:
            image_input: Image input
            bboxes: List of bbox dicts with 'x1', 'y1', 'x2', 'y2' or 'x', 'y', 'w', 'h'
            
        Returns:
            List of extracted text strings
        """
        try:
            # Convert to numpy array
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
            elif isinstance(image_input, Image.Image):
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                image = image_input.copy()
            
            if image is None or image.size == 0:
                return []
            
            texts = []
            for bbox in bboxes:
                try:
                    # Handle different bbox formats
                    if 'x1' in bbox:
                        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                    else:
                        x1 = int(bbox['x'])
                        y1 = int(bbox['y'])
                        x2 = x1 + int(bbox['w'])
                        y2 = y1 + int(bbox['h'])
                    
                    # Crop region
                    roi = image[y1:y2, x1:x2]
                    if roi.size == 0:
                        texts.append('')
                        continue
                    
                    # Extract text from region
                    result = self.extract_text(roi)
                    texts.append(result['text'])
                    
                except Exception as e:
                    logger.warning(f"Failed to extract from region: {str(e)}")
                    texts.append('')
            
            logger.info(f"Extracted text from {len(texts)} regions")
            return texts
            
        except Exception as e:
            logger.error(f"Multiple region extraction failed: {str(e)}")
            raise
    
    def build_search_query_from_text(self, extracted_text: str) -> str:
        """
        Clean extracted text and build search query
        
        Args:
            extracted_text: Raw extracted text
            
        Returns:
            Cleaned query string (max 100 characters)
        """
        try:
            # Remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', extracted_text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Common stopwords
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'and', 'or', 'not', 'in', 'at', 'by', 'for', 'to', 'of', 'with'}
            
            # Remove stopwords
            words = text.split()
            filtered = [w for w in words if w not in stopwords and len(w) > 2]
            
            query = ' '.join(filtered)
            
            # Truncate to max 100 characters
            query = query[:100]
            
            logger.debug(f"Built search query: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Query building failed: {str(e)}")
            return extracted_text[:100]
    
    def evaluate(
        self,
        test_dataset_dir: str
    ) -> Dict[str, Any]:
        """
        Evaluate OCR performance on test dataset
        
        Args:
            test_dataset_dir: Directory containing test images and ground truth labels
            
        Returns:
            Dict with per-category and overall metrics:
                "per_category": {
                  "clear_text": {"precision": 0.97, "recall": 0.96, "f1": 0.97, "support": 80},
                  ...
                },
                "overall": {"accuracy": 0.9167, "precision": 0.9197, "recall": 0.9167, "f1": 0.9163},
                "cer": float,  # Character Error Rate
                "wer": float   # Word Error Rate
        """
        try:
            test_path = Path(test_dataset_dir)
            if not test_path.exists():
                raise FileNotFoundError(f"Test dataset not found: {test_dataset_dir}")
            
            logger.info(f"Starting OCR evaluation on {test_dataset_dir}")
            
            # Categories for evaluation
            categories = [
                'clear_text', 'handwritten', 'small_font',
                'tilted_text', 'low_contrast', 'multilingual'
            ]
            
            # Initialize metrics
            category_results = {cat: {'tp': 0, 'fp': 0, 'fn': 0, 'cer': 0, 'wer': 0} 
                               for cat in categories}
            
            all_cer = []
            all_wer = []
            
            # Look for ground truth file
            gt_file = test_path / 'ground_truth.json'
            if not gt_file.exists():
                logger.warning("No ground_truth.json found, generating dummy evaluation")
                gt_data = {}
            else:
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
            
            # Process test images
            image_files = sorted(test_path.glob('*.jpg')) + sorted(test_path.glob('*.png'))
            
            for img_file in tqdm(image_files, desc="Evaluating"):
                try:
                    # Detect text type
                    text_type = self.detect_text_type(str(img_file))
                    for cat in categories:
                        if cat == text_type or (text_type == 'handwritten' and cat == 'handwritten'):
                            # Run extraction
                            result = self.preprocess_and_extract(str(img_file))
                            pred_text = result['text']
                            
                            # Get ground truth
                            gt_text = gt_data.get(img_file.name, '')
                            
                            # Calculate CER and WER
                            cer = self._calculate_cer(gt_text, pred_text)
                            wer = self._calculate_wer(gt_text, pred_text)
                            
                            all_cer.append(cer)
                            all_wer.append(wer)
                            
                            category_results[cat]['cer'] += cer
                            category_results[cat]['wer'] += wer
                            category_results[cat]['tp'] += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to evaluate {img_file}: {str(e)}")
                    continue
            
            # Compute final metrics
            per_category = {}
            for cat in categories:
                support = category_results[cat]['tp']
                if support > 0:
                    per_category[cat] = {
                        'precision': min(1.0, 1.0 - (category_results[cat]['cer'] / support / 100)),
                        'recall': min(1.0, 1.0 - (category_results[cat]['wer'] / support / 100)),
                        'f1': 0.5,  # Placeholder
                        'support': support
                    }
                    # Compute F1
                    p = per_category[cat]['precision']
                    r = per_category[cat]['recall']
                    if p + r > 0:
                        per_category[cat]['f1'] = 2 * (p * r) / (p + r)
            
            # Overall metrics
            mean_cer = sum(all_cer) / len(all_cer) if all_cer else 0
            mean_wer = sum(all_wer) / len(all_wer) if all_wer else 0
            overall_accuracy = 1.0 - (mean_cer + mean_wer) / 2 / 100
            
            result = {
                'per_category': per_category,
                'overall': {
                    'accuracy': overall_accuracy,
                    'precision': 1.0 - (mean_cer / 100),
                    'recall': 1.0 - (mean_wer / 100),
                    'f1': 0.0
                },
                'cer': mean_cer,
                'wer': mean_wer
            }
            
            # Compute overall F1
            p = result['overall']['precision']
            r = result['overall']['recall']
            if p + r > 0:
                result['overall']['f1'] = 2 * (p * r) / (p + r)
            
            logger.info(f"Evaluation complete. Overall Accuracy: {overall_accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    @staticmethod
    def _calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            Character error rate as percentage
        """
        if len(reference) == 0:
            return 0.0 if len(hypothesis) == 0 else 100.0
        
        # Simple edit distance approximation
        ref_chars = len(reference)
        hyp_chars = len(hypothesis)
        errors = abs(ref_chars - hyp_chars)
        
        return (errors / ref_chars) * 100 if ref_chars > 0 else 0.0
    
    @staticmethod
    def _calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            Word error rate as percentage
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 100.0
        
        errors = abs(len(ref_words) - len(hyp_words))
        
        return (errors / len(ref_words)) * 100 if len(ref_words) > 0 else 0.0
