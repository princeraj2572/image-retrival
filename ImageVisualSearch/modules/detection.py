"""
Object Detection Module using YOLOv8
Detects objects in images using pre-trained YOLOv8 models
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Object detection using YOLOv8
    
    Attributes:
        model_name (str): YOLOv8 model size (n, s, m, l, x)
        device (str): Device to use (cpu or cuda)
        confidence_threshold (float): Confidence threshold for detections
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu",
                 confidence_threshold: float = 0.5):
        """
        Initialize ObjectDetector
        
        Args:
            model_name: YOLOv8 model to load
            device: Device to use (cpu or cuda)
            confidence_threshold: Minimum confidence for detections
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.model.to(device)
            self.device = device
            self.confidence_threshold = confidence_threshold
            logger.info(f"ObjectDetector initialized with {model_name} on {device}")
        except ImportError:
            logger.error("ultralytics package required. Install with: pip install ultralytics")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections with boxes, confidences, and class labels
        """
        try:
            results = self.model(image, conf=self.confidence_threshold)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),  # x1, y1, x2, y2
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': result.names[int(box.cls[0])]
                    }
                    detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect objects in multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists
        """
        batch_detections = []
        for image in images:
            detections = self.detect(image)
            batch_detections.append(detections)
        return batch_detections
    
    def visualize(self, image: np.ndarray, detections: List[Dict],
                  output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detections on image
        
        Args:
            image: Input image
            detections: List of detections
            output_path: Optional path to save visualization
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Visualization saved to {output_path}")
        
        return vis_image
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            'model_name': self.model.model.yaml,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold
        }
