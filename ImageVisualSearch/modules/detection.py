"""
Object Detection Module using YOLOv8
Implements the ObjectDetector class for real-time object detection and model training
"""

import logging
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None

# Import from project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils.preprocessing import ImagePreprocessor
from utils.helpers import setup_logging, draw_bounding_boxes, calculate_metrics

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Object Detection class using YOLOv8 for real-time detection
    
    Attributes:
        model: YOLO model instance
        confidence_threshold: Minimum confidence for detections
        device: Processing device (cuda or cpu)
        preprocessor: ImagePreprocessor instance
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize ObjectDetector with YOLO model
        
        Args:
            model_path: Path to YOLO model (uses Config.YOLO_MODEL if None)
            confidence_threshold: Detection confidence threshold (uses Config.CONFIDENCE_THRESHOLD if None)
            device: Device to use - 'cuda' or 'cpu' (uses Config.DEVICE if None)
            
        Raises:
            ImportError: If ultralytics package not installed
            FileNotFoundError: If model file not found
        """
        try:
            if YOLO is None:
                raise ImportError("ultralytics package required. Install with: pip install ultralytics")
            
            # Load configuration settings
            self.model_path = model_path or Config.YOLO_MODEL
            self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
            self.device = device or Config.DEVICE
            
            # Setup Tesseract path if available
            if Config.TESSERACT_PATH:
                import pytesseract
                pytesseract.pytesseract.pytesseract_cmd = Config.TESSERACT_PATH
            
            # Initialize YOLO model
            logger.info(f"Loading YOLO model: {self.model_path} on device {self.device}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Initialize preprocessor
            self.preprocessor = ImagePreprocessor(
                target_size=(Config.YOLO_IMG_SIZE, Config.YOLO_IMG_SIZE),
                normalize=True
            )
            
            # Initialize logger
            setup_logging(log_level=Config.LOG_LEVEL)
            
            model_info = f"Model: {self.model_path}, Device: {self.device}, Confidence: {self.confidence_threshold}"
            logger.info(f"Object Detection Module initialized - {model_info}")
            print(f"✓ Object Detection Module initialized - {model_info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ObjectDetector: {str(e)}")
            raise
    
    def detect(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Run object detection on a single image
        
        Args:
            image_input: Image input as file path (str), numpy array (BGR), or PIL Image
            
        Returns:
            List[Dict]: List of detection dictionaries with keys:
                       'class', 'confidence', 'bbox' (x1,y1,x2,y2), 'class_id', 'area'
                       
        Raises:
            ValueError: If image cannot be processed
            FileNotFoundError: If image path not found
        """
        try:
            # Convert input to numpy array
            if isinstance(image_input, str):
                # Validate image file
                if not self.preprocessor.validate_image(image_input):
                    raise ValueError(f"Invalid or corrupted image: {image_input}")
                
                image = cv2.imread(image_input)
                if image is None:
                    raise FileNotFoundError(f"Cannot read image: {image_input}")
                    
            elif isinstance(image_input, Image.Image):
                # Convert PIL Image to numpy array (BGR)
                image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
                
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
                
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Check for empty image
            if image is None or image.size == 0:
                logger.warning("Empty image received, returning empty detections")
                return []
            
            # Run inference
            logger.debug(f"Running detection on image shape: {image.shape}")
            results = self.model(
                image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                for box in result.boxes:
                    try:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() if torch else box.xyxy[0]
                        
                        # Calculate area
                        area = int((x2 - x1) * (y2 - y1))
                        
                        detection = {
                            'class': result.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': int(box.cls[0]),
                            'area': area
                        }
                        detections.append(detection)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse detection: {str(e)}")
                        continue
            
            # Sort by confidence descending
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise
    
    def detect_batch(self, image_list: List[Union[str, np.ndarray]]) -> List[List[Dict[str, Any]]]:
        """
        Run detection on multiple images with progress bar
        
        Args:
            image_list: List of image paths or numpy arrays
            
        Returns:
            List[List[Dict]]: List of detection results for each image
        """
        try:
            results = []
            
            for image in tqdm(image_list, desc="Processing images", unit="img"):
                try:
                    detections = self.detect(image)
                    results.append(detections)
                except Exception as e:
                    logger.warning(f"Failed to process image: {str(e)}")
                    results.append([])
            
            logger.info(f"Batch detection completed: {len(results)} images processed")
            return results
            
        except Exception as e:
            logger.error(f"Batch detection failed: {str(e)}")
            raise
    
    def get_dominant_objects(self, detections: List[Dict[str, Any]], top_n: int = 3) -> List[str]:
        """
        Get top N unique object classes by confidence
        Used for query building and scene understanding
        
        Args:
            detections: List of detection dictionaries
            top_n: Number of top classes to return. Default: 3
            
        Returns:
            List[str]: Top N class labels by confidence
        """
        try:
            if not detections:
                logger.warning("No detections provided")
                return []
            
            # Group by class and get max confidence for each
            class_confidence = {}
            for detection in detections:
                class_name = detection['class']
                confidence = detection['confidence']
                
                if class_name not in class_confidence:
                    class_confidence[class_name] = confidence
                else:
                    class_confidence[class_name] = max(class_confidence[class_name], confidence)
            
            # Sort by confidence and get top N
            sorted_classes = sorted(
                class_confidence.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            dominant = [cls[0] for cls in sorted_classes[:top_n]]
            logger.debug(f"Dominant objects: {dominant}")
            
            return dominant
            
        except Exception as e:
            logger.error(f"Failed to get dominant objects: {str(e)}")
            raise
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            thickness: Line thickness. Default: 2
            font_scale: Font scale. Default: 0.6
            
        Returns:
            np.ndarray: Annotated image
        """
        try:
            annotated = image.copy()
            
            if not detections:
                logger.warning("No detections to draw")
                return annotated
            
            for detection in detections:
                try:
                    bbox = detection['bbox']
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw green bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                    
                    # Prepare label
                    label = f"{class_name} {confidence:.2f}"
                    
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )
                    
                    # Draw black background
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width + 5, y1),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw white text
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        1
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to draw detection: {str(e)}")
                    continue
            
            logger.debug(f"Drew {len(detections)} bounding boxes")
            return annotated
            
        except Exception as e:
            logger.error(f"Failed to draw detections: {str(e)}")
            raise
    
    def evaluate(
        self,
        test_images_dir: str,
        annotations_dir: str
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset with YOLO format annotations
        
        Args:
            test_images_dir: Directory containing test images
            annotations_dir: Directory containing YOLO format .txt annotation files
            
        Returns:
            Dict: Evaluation metrics with per-class and overall performance:
                  {
                    "per_class": {
                      "person": {"precision": 0.93, "recall": 0.91, "f1": 0.92, "support": 120},
                      ...
                    },
                    "overall": {"accuracy": 0.9042, "precision": 0.9072, "recall": 0.9042, "f1": 0.9047}
                  }
        """
        try:
            test_path = Path(test_images_dir)
            annotations_path = Path(annotations_dir)
            
            if not test_path.exists():
                raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
            if not annotations_path.exists():
                raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
            
            logger.info(f"Starting evaluation on {test_images_dir}")
            
            y_true = []
            y_pred = []
            
            # Process each image
            image_files = sorted(test_path.glob('*.jpg')) + sorted(test_path.glob('*.png'))
            
            for image_file in tqdm(image_files, desc="Evaluating"):
                try:
                    # Get annotation file
                    annotation_file = annotations_path / f"{image_file.stem}.txt"
                    
                    if not annotation_file.exists():
                        logger.warning(f"Annotation not found for {image_file}")
                        continue
                    
                    # Run detection
                    detections = self.detect(str(image_file))
                    
                    # Parse ground truth (YOLO format: class_id x_center y_center width height)
                    with open(annotation_file, 'r') as f:
                        gt_objects = [line.strip().split()[0] for line in f if line.strip()]
                    
                    # Extract predicted classes
                    pred_classes = [str(det['class_id']) for det in detections]
                    
                    # Match predictions to ground truth
                    for gt_class in gt_objects:
                        y_true.append(int(gt_class))
                        if pred_classes:
                            y_pred.append(int(pred_classes[0]))
                            pred_classes.pop(0)
                        else:
                            y_pred.append(-1)  # No detection
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate {image_file}: {str(e)}")
                    continue
            
            # Calculate metrics
            if not y_true:
                logger.warning("No annotations found for evaluation")
                return {}
            
            metrics = calculate_metrics(
                np.array(y_true),
                np.array(y_pred),
                labels=list(range(len(Config.YOLO_CLASSES)))
            )
            
            # Format output matching Table 3 format
            result = {
                "per_class": {},
                "overall": {
                    "accuracy": metrics.get('accuracy', 0),
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "f1": metrics.get('f1', 0)
                }
            }
            
            # Add per-class metrics
            if 'classification_report' in metrics:
                for class_id, class_name in enumerate(Config.YOLO_CLASSES):
                    class_report = metrics['classification_report'].get(str(class_id), {})
                    result["per_class"][class_name] = {
                        "precision": class_report.get('precision', 0),
                        "recall": class_report.get('recall', 0),
                        "f1": class_report.get('f1-score', 0),
                        "support": class_report.get('support', 0)
                    }
            
            logger.info(f"Evaluation complete. Overall F1: {result['overall']['f1']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def save_model(self, save_path: str) -> None:
        """
        Save trained/fine-tuned model weights
        
        Args:
            save_path: Path to save model weights
            
        Returns:
            None
            
        Raises:
            Exception: If save operation fails
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.model.save(str(save_path))
            logger.info(f"Model saved to {save_path}")
            print(f"✓ Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def fine_tune(
        self,
        dataset_yaml_path: str,
        epochs: Optional[int] = None,
        batch: Optional[int] = None
    ) -> None:
        """
        Fine-tune YOLO model on custom dataset
        
        Args:
            dataset_yaml_path: Path to YOLO dataset.yaml file
            epochs: Number of training epochs (uses Config.NUM_EPOCHS if None)
            batch: Batch size (uses Config.BATCH_SIZE if None)
            
        Returns:
            None
            
        Raises:
            FileNotFoundError: If dataset.yaml not found
            Exception: If training fails
        """
        try:
            yaml_path = Path(dataset_yaml_path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"dataset.yaml not found: {dataset_yaml_path}")
            
            epochs = epochs or Config.NUM_EPOCHS
            batch = batch or Config.BATCH_SIZE
            
            logger.info(f"Starting fine-tuning: epochs={epochs}, batch={batch}")
            print(f"🔧 Starting fine-tuning: epochs={epochs}, batch={batch}")
            
            # Run training
            results = self.model.train(
                data=str(yaml_path),
                epochs=epochs,
                batch=batch,
                device=self.device,
                patience=10,
                save=True,
                verbose=True,
                project=str(Config.MODEL_DIR / "yolo"),
                name="training"
            )
            
            # Save best weights
            best_weights = Config.MODEL_DIR / "yolo" / "training" / "weights" / "best.pt"
            if best_weights.exists():
                self.save_model(str(Config.MODEL_DIR / "yolo" / "best.pt"))
            
            logger.info("Fine-tuning completed successfully")
            print("✓ Fine-tuning completed successfully")
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            raise
