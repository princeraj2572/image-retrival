"""
YOLO Model Training Script
Handles dataset preparation, splitting, and fine-tuning of YOLOv8 models
"""

import logging
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

# Import from project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils.helpers import setup_logging
from modules.detection import ObjectDetector

logger = logging.getLogger(__name__)


class YOLODatasetPreparer:
    """
    Prepare YOLO dataset format for training
    
    Attributes:
        dataset_dir: Root dataset directory
        train_split: Training data proportion. Default: 0.70
        val_split: Validation data proportion. Default: 0.15
        test_split: Test data proportion. Default: 0.15
    """
    
    def __init__(
        self,
        dataset_dir: str,
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15
    ) -> None:
        """
        Initialize dataset preparer
        
        Args:
            dataset_dir: Root directory containing images and labels
            train_split: Training proportion
            val_split: Validation proportion
            test_split: Test proportion
            
        Raises:
            ValueError: If splits don't sum to 1.0
            FileNotFoundError: If dataset directory not found
        """
        try:
            if train_split + val_split + test_split != 1.0:
                raise ValueError("Train, val, and test splits must sum to 1.0")
            
            if train_split < 0 or val_split < 0 or test_split < 0:
                raise ValueError("All splits must be non-negative")
            
            self.dataset_dir = Path(dataset_dir)
            if not self.dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
            
            self.train_split = train_split
            self.val_split = val_split
            self.test_split = test_split
            
            self.images_dir = self.dataset_dir / "images"
            self.labels_dir = self.dataset_dir / "labels"
            
            logger.info(f"YOLODatasetPreparer initialized: {dataset_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLODatasetPreparer: {str(e)}")
            raise
    
    def collect_images(self) -> List[Path]:
        """
        Collect all image files from dataset directory
        
        Returns:
            List[Path]: List of image file paths
        """
        try:
            image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            
            images = []
            for ext in image_formats:
                images.extend(self.images_dir.glob(f"*{ext}"))
                images.extend(self.images_dir.glob(f"*{ext.upper()}"))
            
            images = list(set(images))  # Remove duplicates
            images.sort()
            
            logger.info(f"Collected {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Failed to collect images: {str(e)}")
            raise
    
    def create_dataset_yaml(self, output_path: str) -> Dict:
        """
        Create YOLO dataset.yaml configuration file
        
        Args:
            output_path: Path to save dataset.yaml
            
        Returns:
            Dict: Dataset configuration
            
        Raises:
            FileNotFoundError: If image directories not found
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create split directories if they don't exist
            for split in ['train', 'val', 'test']:
                (self.images_dir.parent / split / "images").mkdir(parents=True, exist_ok=True)
                (self.images_dir.parent / split / "labels").mkdir(parents=True, exist_ok=True)
            
            # Create dataset.yaml configuration
            dataset_yaml = {
                'path': str(self.dataset_dir.parent),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(Config.YOLO_CLASSES),
                'names': {i: class_name for i, class_name in enumerate(Config.YOLO_CLASSES)}
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)
            
            logger.info(f"Created dataset.yaml at {output_path}")
            return dataset_yaml
            
        except Exception as e:
            logger.error(f"Failed to create dataset.yaml: {str(e)}")
            raise
    
    def split_dataset(self) -> Dict[str, List[Path]]:
        """
        Split dataset into train/val/test using sklearn
        
        Returns:
            Dict: Dictionary with 'train', 'val', 'test' keys containing file paths
            
        Raises:
            ImportError: If sklearn not available
        """
        try:
            if train_test_split is None:
                raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
            
            images = self.collect_images()
            
            if not images:
                raise ValueError("No images found in dataset")
            
            logger.info(f"Splitting {len(images)} images...")
            
            # First split: separate test set
            train_val, test = train_test_split(
                images,
                test_size=self.test_split,
                random_state=42
            )
            
            # Second split: separate train and val from remaining
            val_size_adjusted = self.val_split / (1 - self.test_split)
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=42
            )
            
            split_data = {
                'train': train,
                'val': val,
                'test': test
            }
            
            logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            
            return split_data
            
        except Exception as e:
            logger.error(f"Dataset splitting failed: {str(e)}")
            raise
    
    def organize_dataset(self, split_data: Dict[str, List[Path]]) -> None:
        """
        Organize images and labels into train/val/test directories
        
        Args:
            split_data: Dictionary with 'train', 'val', 'test' file paths
            
        Returns:
            None
        """
        try:
            import shutil
            
            for split_name, image_paths in split_data.items():
                output_images_dir = self.images_dir.parent / split_name / "images"
                output_labels_dir = self.images_dir.parent / split_name / "labels"
                
                output_images_dir.mkdir(parents=True, exist_ok=True)
                output_labels_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in tqdm(image_paths, desc=f"Organizing {split_name}"):
                    # Copy image
                    dest_img = output_images_dir / img_path.name
                    shutil.copy2(img_path, dest_img)
                    
                    # Copy corresponding label file if exists
                    label_file = self.labels_dir / f"{img_path.stem}.txt"
                    if label_file.exists():
                        dest_label = output_labels_dir / label_file.name
                        shutil.copy2(label_file, dest_label)
            
            logger.info("Dataset organized successfully")
            
        except Exception as e:
            logger.error(f"Failed to organize dataset: {str(e)}")
            raise


def prepare_yolo_dataset(
    dataset_dir: str,
    output_yaml: Optional[str] = None,
    train_split: float = None,
    val_split: float = None,
    test_split: float = None
) -> Tuple[Dict, str]:
    """
    Prepare dataset for YOLO training
    
    Args:
        dataset_dir: Root dataset directory
        output_yaml: Path to save dataset.yaml (uses Config.DATA_DIR if None)
        train_split: Training proportion (uses Config.TRAIN_SPLIT if None)
        val_split: Validation proportion (uses Config.VAL_SPLIT if None)
        test_split: Test proportion (uses Config.TEST_SPLIT if None)
        
    Returns:
        Tuple[Dict, str]: Dataset configuration and path to dataset.yaml
    """
    try:
        train_split = train_split or Config.TRAIN_SPLIT
        val_split = val_split or Config.VAL_SPLIT
        test_split = test_split or Config.TEST_SPLIT
        
        output_yaml = output_yaml or str(Config.DATA_DIR / "dataset.yaml")
        
        logger.info("Preparing YOLO dataset...")
        print("🔧 Preparing YOLO dataset...")
        
        # Initialize preparer
        preparer = YOLODatasetPreparer(
            dataset_dir=dataset_dir,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split
        )
        
        # Create dataset.yaml
        yaml_config = preparer.create_dataset_yaml(output_yaml)
        
        # Split dataset
        split_data = preparer.split_dataset()
        
        # Organize into directories
        preparer.organize_dataset(split_data)
        
        logger.info(f"Dataset preparation complete. Config saved to {output_yaml}")
        print(f"✓ Dataset prepared. Config: {output_yaml}")
        
        return yaml_config, output_yaml
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise


def train_yolo_model(
    dataset_yaml_path: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None
) -> Dict:
    """
    Train YOLO model on prepared dataset
    
    Args:
        dataset_yaml_path: Path to dataset.yaml
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_path: Path to YOLO model weights
        device: Device for training (cuda or cpu)
        
    Returns:
        Dict: Training results and metrics
    """
    try:
        epochs = epochs or Config.NUM_EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE
        model_path = model_path or Config.YOLO_MODEL
        device = device or Config.DEVICE
        
        logger.info(f"Starting YOLO training: {epochs} epochs, batch={batch_size}")
        print(f"🚀 Starting YOLO training: {epochs} epochs, batch={batch_size}")
        
        # Initialize detector and fine-tune
        detector = ObjectDetector(
            model_path=model_path,
            device=device
        )
        
        detector.fine_tune(
            dataset_yaml_path=dataset_yaml_path,
            epochs=epochs,
            batch=batch_size
        )
        
        logger.info("YOLO training completed")
        print("✓ YOLO training completed successfully")
        
        return {
            'status': 'success',
            'model_path': str(Config.MODEL_DIR / "yolo" / "best.pt"),
            'epochs': epochs,
            'batch_size': batch_size
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def save_training_results(
    results: Dict,
    output_dir: Optional[str] = None
) -> str:
    """
    Save training results and metrics
    
    Args:
        results: Training results dictionary
        output_dir: Directory to save results (uses Config.RESULTS_DIR if None)
        
    Returns:
        str: Path to saved results file
    """
    try:
        output_dir = Path(output_dir or Config.RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "training_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Training results saved to {results_file}")
        print(f"✓ Results saved: {results_file}")
        
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise


def main():
    """
    Main training pipeline
    """
    try:
        # Setup logging
        setup_logging(log_level=Config.LOG_LEVEL)
        
        logger.info("=" * 60)
        logger.info("YOLO Model Training Pipeline")
        logger.info("=" * 60)
        
        # Prepare dataset
        dataset_dir = Config.PROCESSED_DIR
        yaml_config, yaml_path = prepare_yolo_dataset(
            dataset_dir=str(dataset_dir),
            train_split=Config.TRAIN_SPLIT,
            val_split=Config.VAL_SPLIT,
            test_split=Config.TEST_SPLIT
        )
        
        # Train model
        training_results = train_yolo_model(
            dataset_yaml_path=yaml_path,
            epochs=Config.NUM_EPOCHS,
            batch_size=Config.BATCH_SIZE,
            model_path=Config.YOLO_MODEL,
            device=Config.DEVICE
        )
        
        # Evaluate model (optional)
        detector = ObjectDetector(
            model_path=str(Config.MODEL_DIR / "yolo" / "best.pt"),
            device=Config.DEVICE
        )
        
        eval_results = detector.evaluate(
            test_images_dir=str(Config.TEST_DIR / "images"),
            annotations_dir=str(Config.TEST_DIR / "labels")
        )
        
        # Combine results
        final_results = {
            'training': training_results,
            'evaluation': eval_results,
            'config': {
                'epochs': Config.NUM_EPOCHS,
                'batch_size': Config.BATCH_SIZE,
                'model': Config.YOLO_MODEL,
                'device': Config.DEVICE,
                'dataset_yaml': yaml_path
            }
        }
        
        # Save results
        save_training_results(
            results=final_results,
            output_dir=str(Config.RESULTS_DIR)
        )
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
