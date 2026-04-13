"""
Helper Functions Module
Comprehensive utility functions for logging, results management, metrics, and visualization
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import cv2

try:
    import torch
except ImportError:
    torch = None

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Define colors for different classes (BGR format for OpenCV)
CLASS_COLORS = {
    "person": (0, 255, 0),
    "car": (0, 0, 255),
    "building": (255, 0, 0),
    "animal": (255, 255, 0),
    "food": (0, 255, 255),
    "electronics": (255, 0, 255),
    "furniture": (128, 0, 128),
    "plant": (0, 128, 0),
}


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup comprehensive logging configuration with timestamp format
    Logs to both console and file (outputs/app.log)
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        ValueError: If invalid log level provided
    """
    try:
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")
        
        # Create logs directory
        from config import Config
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = Config.LOGS_DIR / "app.log"
        
        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(message)s",
            datefmt=date_format
        )
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized to {log_file} with level {log_level}")
        
        return root_logger
        
    except Exception as e:
        logger.error(f"Failed to setup logging: {str(e)}")
        raise


def get_device() -> any:
    """
    Get the appropriate device (CUDA if available, else CPU)
    Prints which device is being used
    
    Returns:
        torch.device: CUDA device if available, else CPU device
        
    Raises:
        ImportError: If PyTorch not installed
    """
    try:
        if torch is None:
            logger.error("PyTorch not installed. Install with: pip install torch")
            raise ImportError("PyTorch not installed")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {device_name}")
            print(f"✓ GPU Device: {device_name}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
            print("✓ Using CPU device")
        
        return device
        
    except Exception as e:
        logger.error(f"Failed to get device: {str(e)}")
        raise


def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save results dictionary as JSON to outputs/results/ directory
    
    Args:
        results: Dictionary containing results to save
        filename: Name of the output file (without extension)
        
    Returns:
        None
        
    Raises:
        IOError: If unable to write to file
        TypeError: If results contain non-serializable objects
    """
    try:
        from config import Config
        
        # Ensure results directory exists
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create output path
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        output_path = Config.RESULTS_DIR / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # Serialize and save
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Results saved to {output_path}")
        print(f"✓ Results saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load and return JSON results file from outputs/results/ directory
    
    Args:
        filename: Name of the results file to load (with or without .json extension)
        
    Returns:
        Dict: Loaded results dictionary
        
    Raises:
        FileNotFoundError: If results file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        from config import Config
        
        # Create filename with extension if needed
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        
        file_path = Config.RESULTS_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"Results file not found: {file_path}")
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        # Load JSON file
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {file_path}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to load results: {str(e)}")
        raise


def draw_bounding_boxes(image: np.ndarray, detections: List[Dict], 
                       thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
    """
    Draw colored bounding boxes with class labels and confidence scores on image
    Uses different color per class
    
    Args:
        image: Input image (BGR format from OpenCV)
        detections: List of detection dictionaries with keys:
                   'bbox' (x1, y1, x2, y2), 'class_name', 'confidence'
        thickness: Thickness of bounding box lines. Default: 2
        font_scale: Font scale for text. Default: 0.6
        
    Returns:
        np.ndarray: Annotated image with bounding boxes
        
    Raises:
        ValueError: If detections format is invalid
    """
    try:
        annotated = image.copy()
        
        if not detections:
            logger.warning("No detections to draw")
            return annotated
        
        for detection in detections:
            try:
                # Extract detection info
                bbox = detection.get('bbox', [0, 0, 0, 0])
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # Convert bbox coordinates to int
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get color for class
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label text
                label = f"{class_name}: {confidence:.2%}"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Draw text background
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width + 5, y1),
                    color,
                    -1
                )
                
                # Draw text
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
        logger.error(f"Failed to draw bounding boxes: {str(e)}")
        raise


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics using sklearn
    Returns precision, recall, F1, accuracy, and detailed report
    
    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        labels: List of label names. Default: None
        
    Returns:
        Dict with keys: 'accuracy', 'precision', 'recall', 'f1',
                       'classification_report', 'confusion_matrix'
        
    Raises:
        ValueError: If y_true and y_pred have different lengths
        ImportError: If sklearn not installed
    """
    try:
        from sklearn.metrics import confusion_matrix
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Generate detailed classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                      zero_division=0)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Compile results
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'n_samples': len(y_true),
            'n_classes': len(np.unique(y_true))
        }
        
        logger.info(f"Metrics calculated: Accuracy={accuracy:.4f}, "
                   f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return metrics
        
    except ImportError:
        logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
        raise
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {str(e)}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False


def create_directory(directory: str) -> bool:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created/verified: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory: {e}")
        return False


def get_file_size(file_path: str) -> Optional[float]:
    """
    Get file size in MB
    
    Args:
        file_path: Path to file
        
    Returns:
        float: File size in MB, or None if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return None


def get_directory_size(directory: str) -> float:
    """
    Get total size of directory in MB
    
    Args:
        directory: Directory path
        
    Returns:
        float: Total size in MB
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)


def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
    """
    List all files in directory
    
    Args:
        directory: Directory path
        extension: Optional file extension filter (e.g., '.jpg')
        
    Returns:
        List[str]: List of file paths
    """
    try:
        files = []
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                if extension is None or file_path.suffix.lower() == extension.lower():
                    files.append(str(file_path))
        return files
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return []


def format_time(seconds: float) -> str:
    """
    Format time duration to readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def validate_file_exists(file_path: str) -> bool:
    """
    Validate if file exists
    
    Args:
        file_path: Path to file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    exists = Path(file_path).exists()
    if not exists:
        logger.warning(f"File not found: {file_path}")
    return exists


def validate_directory_exists(directory: str) -> bool:
    """
    Validate if directory exists
    
    Args:
        directory: Directory path
        
    Returns:
        bool: True if directory exists, False otherwise
    """
    exists = Path(directory).is_dir()
    if not exists:
        logger.warning(f"Directory not found: {directory}")
    return exists
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    # Logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized to {log_file}")
    
    return root_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False


def create_directory(directory: str) -> bool:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created/verified: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory: {e}")
        return False


def get_file_size(file_path: str) -> Optional[float]:
    """
    Get file size in MB
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB, or None if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return None


def get_directory_size(directory: str) -> float:
    """
    Get total size of directory in MB
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in MB
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)


def list_files(directory: str, extension: Optional[str] = None) -> list:
    """
    List all files in directory
    
    Args:
        directory: Directory path
        extension: Optional file extension filter (e.g., '.jpg')
        
    Returns:
        List of file paths
    """
    try:
        files = []
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                if extension is None or file_path.suffix.lower() == extension.lower():
                    files.append(str(file_path))
        return files
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return []


def format_time(seconds: float) -> str:
    """
    Format time duration to readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def dict_to_string(data: Dict[str, Any], indent: int = 0) -> str:
    """
    Convert dictionary to formatted string
    
    Args:
        data: Dictionary to format
        indent: Indentation level
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{'  ' * indent}{key}:")
            lines.append(dict_to_string(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{'  ' * indent}{key}: [{len(value)} items]")
        else:
            lines.append(f"{'  ' * indent}{key}: {value}")
    return "\n".join(lines)


def validate_file_exists(file_path: str) -> bool:
    """
    Validate if file exists
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    exists = Path(file_path).exists()
    if not exists:
        logger.warning(f"File not found: {file_path}")
    return exists


def validate_directory_exists(directory: str) -> bool:
    """
    Validate if directory exists
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists, False otherwise
    """
    exists = Path(directory).is_dir()
    if not exists:
        logger.warning(f"Directory not found: {directory}")
    return exists
