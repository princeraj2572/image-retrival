"""
Visual Similarity Matcher Module
Complete ResNet50 + FAISS implementation for image similarity matching
"""

import logging
import json
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

try:
    import torch
    import torchvision
    from torchvision import models, transforms
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    torchvision = None
    models = None
    transforms = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Import from project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """PyTorch Dataset for batch image processing"""
    
    def __init__(self, image_paths: List[str], transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.warning(f"Failed to load {self.image_paths[idx]}: {str(e)}")
            return None


class VisualSimilarityMatcher:
    """
    Visual Similarity Matcher using ResNet50 + FAISS
    
    Attributes:
        model: ResNet50 model without final FC layer
        index: FAISS IndexFlatIP for similarity search
        device: PyTorch device (cuda or cpu)
        transform: Image preprocessing pipeline
        reference_database: Dict with embeddings, paths, labels
    """
    
    def __init__(self, index_path: Optional[str] = None) -> None:
        """
        Initialize VisualSimilarityMatcher with ResNet50 and FAISS
        
        Args:
            index_path: Path to load existing FAISS index (optional)
            
        Raises:
            ImportError: If torch, torchvision, or faiss not installed
        """
        try:
            if torch is None:
                raise ImportError("torch and torchvision required. Install with: pip install torch torchvision")
            if faiss is None:
                raise ImportError("faiss required. Install with: pip install faiss-cpu or faiss-gpu")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load ResNet50 pretrained on ImageNet
            logger.info("Loading ResNet50 model...")
            resnet = models.resnet50(pretrained=True)
            
            # Remove final fully connected layer
            self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Define image transform pipeline
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Initialize FAISS IndexFlatIP (inner product for cosine similarity)
            self.index = faiss.IndexFlatIP(Config.EMBEDDING_DIM)
            
            # Initialize reference database
            self.reference_database = {
                'embeddings': [],
                'paths': [],
                'labels': []
            }
            
            # Load existing index if provided
            if index_path:
                self.load_index(
                    index_path=index_path,
                    metadata_path=str(Path(index_path).parent / 'db_metadata.json')
                )
            
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            logger.info(f"Visual Similarity Module initialized on {self.device} ({device_name})")
            print(f"✓ Visual Similarity Module initialized on {device_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VisualSimilarityMatcher: {str(e)}")
            raise
    
    def extract_embedding(
        self,
        image_input: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Extract 2048-dimensional embedding from image
        
        Args:
            image_input: Image as file path (str), numpy array (BGR), or PIL Image
            
        Returns:
            L2-normalized embedding of shape (2048,)
            
        Raises:
            ValueError: If image invalid
            FileNotFoundError: If file path not found
        """
        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                if not Path(image_input).exists():
                    raise FileNotFoundError(f"Image not found: {image_input}")
                image = Image.open(image_input).convert('RGB')
                
            elif isinstance(image_input, np.ndarray):
                # Convert BGR to RGB
                image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                
            elif isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
                
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
            
            # Apply transform
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(tensor)
            
            # Flatten to 2048-dimensional vector
            embedding = embedding.squeeze().cpu().numpy()
            
            # L2-normalize for cosine similarity
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            logger.debug(f"Extracted embedding shape: {embedding.shape}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            raise
    
    def extract_embeddings_batch(
        self,
        image_list: List[Union[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Extract embeddings for multiple images in batches
        
        Args:
            image_list: List of image paths or numpy arrays
            
        Returns:
            Numpy array of shape (N, 2048)
        """
        try:
            batch_size = 32
            embeddings = []
            
            # Create dataset
            dataset = ImageDataset(image_list, self.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
            
            for batch in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
                try:
                    # Filter out None values
                    batch = [img for img in batch if img is not None]
                    if not batch:
                        continue
                    
                    batch_tensor = torch.stack(batch).to(self.device)
                    
                    with torch.no_grad():
                        batch_embeddings = self.model(batch_tensor)
                    
                    # Flatten and normalize
                    batch_embeddings = batch_embeddings.squeeze().cpu().numpy()
                    if batch_embeddings.ndim == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)
                    
                    # L2-normalize
                    batch_embeddings = batch_embeddings / (np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + 1e-8)
                    embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch: {str(e)}")
                    continue
            
            if not embeddings:
                return np.array([]).reshape(0, Config.EMBEDDING_DIM).astype(np.float32)
            
            result = np.vstack(embeddings).astype(np.float32)
            logger.info(f"Extracted {len(result)} embeddings")
            return result
            
        except Exception as e:
            logger.error(f"Batch embedding extraction failed: {str(e)}")
            raise
    
    def build_reference_database(
        self,
        image_dir: str,
        labels_dict: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Build reference database by scanning directory and extracting embeddings
        
        Args:
            image_dir: Directory containing reference images
            labels_dict: Optional dict mapping image filename to label
            
        Raises:
            FileNotFoundError: If directory not found
        """
        try:
            image_dir = Path(image_dir)
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
            logger.info(f"Building reference database from {image_dir}")
            
            # Collect image paths
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(image_dir.glob(f'**/*{ext}'))
                image_paths.extend(image_dir.glob(f'**/*{ext.upper()}'))
            
            image_paths = sorted(list(set(image_paths)))
            
            if not image_paths:
                raise ValueError(f"No images found in {image_dir}")
            
            logger.info(f"Found {len(image_paths)} images")
            
            # Extract embeddings
            embeddings = self.extract_embeddings_batch([str(p) for p in image_paths])
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store metadata
            for i, img_path in enumerate(image_paths):
                self.reference_database['paths'].append(str(img_path))
                
                # Get label if provided
                label = labels_dict.get(img_path.name, '') if labels_dict else ''
                self.reference_database['labels'].append(label)
            
            self.reference_database['embeddings'] = embeddings
            
            # Save index and metadata
            index_path = Config.MODEL_DIR / "resnet" / "faiss_index.bin"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = index_path.parent / "db_metadata.json"
            metadata = {
                'paths': self.reference_database['paths'],
                'labels': self.reference_database['labels'],
                'num_images': len(image_paths)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Reference database built with {len(image_paths)} images")
            print(f"✓ Reference database built with {len(image_paths)} images")
            
        except Exception as e:
            logger.error(f"Failed to build reference database: {str(e)}")
            raise
    
    def find_similar(
        self,
        query_image: Union[str, np.ndarray, Image.Image],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top-k similar images from reference database
        
        Args:
            query_image: Query image input
            top_k: Number of top results. Default: 5
            
        Returns:
            List of dicts with 'rank', 'image_path', 'similarity_score', 
                 'similarity_level', 'label'
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("Reference database is empty")
                return []
            
            # Extract query embedding
            query_embedding = self.extract_embedding(query_image)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            results = []
            for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
                try:
                    # Convert inner product to cosine similarity (already normalized)
                    similarity_score = float(distance)  # Already in [0, 1] for normalized vectors
                    
                    # Classify similarity level
                    if similarity_score > Config.SIMILARITY.HIGH:
                        level = "High"
                    elif similarity_score > Config.SIMILARITY.MEDIUM:
                        level = "Medium"
                    elif similarity_score > Config.SIMILARITY.LOW:
                        level = "Low"
                    else:
                        level = "No Match"
                    
                    result = {
                        'rank': rank,
                        'image_path': self.reference_database['paths'][idx],
                        'similarity_score': similarity_score,
                        'similarity_level': level,
                        'label': self.reference_database['labels'][idx]
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process result {idx}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(results)} similar images")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise
    
    def compute_similarity(
        self,
        image1: Union[str, np.ndarray, Image.Image],
        image2: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Compute similarity between two images
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Dict with 'cosine_similarity', 'similarity_level', embedding shapes
        """
        try:
            # Extract embeddings
            embedding1 = self.extract_embedding(image1)
            embedding2 = self.extract_embedding(image2)
            
            # Compute cosine similarity (dot product of normalized vectors)
            cosine_similarity = np.dot(embedding1, embedding2)
            
            # Classify similarity level
            if cosine_similarity > Config.SIMILARITY.HIGH:
                level = "High"
            elif cosine_similarity > Config.SIMILARITY.MEDIUM:
                level = "Medium"
            elif cosine_similarity > Config.SIMILARITY.LOW:
                level = "Low"
            else:
                level = "No Match"
            
            result = {
                'cosine_similarity': float(cosine_similarity),
                'similarity_level': level,
                'embedding1_shape': embedding1.shape,
                'embedding2_shape': embedding2.shape
            }
            
            logger.info(f"Computed similarity: {cosine_similarity:.4f} ({level})")
            return result
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            raise
    
    def load_index(
        self,
        index_path: str,
        metadata_path: str
    ) -> None:
        """
        Load existing FAISS index and metadata
        
        Args:
            index_path: Path to saved FAISS index (.bin file)
            metadata_path: Path to saved metadata (.json file)
            
        Raises:
            FileNotFoundError: If files not found
        """
        try:
            index_path = Path(index_path)
            metadata_path = Path(metadata_path)
            
            if not index_path.exists():
                raise FileNotFoundError(f"Index not found: {index_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.reference_database['paths'] = metadata.get('paths', [])
            self.reference_database['labels'] = metadata.get('labels', [])
            
            num_entries = self.index.ntotal
            logger.info(f"Loaded index with {num_entries} entries")
            print(f"✓ Loaded index with {num_entries} entries")
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    
    def save_index(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ) -> None:
        """
        Save FAISS index and metadata
        
        Args:
            index_path: Path to save index (uses Config if None)
            metadata_path: Path to save metadata (uses Config if None)
        """
        try:
            index_path = Path(index_path or Config.MODEL_DIR / "resnet" / "faiss_index.bin")
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = Path(metadata_path or index_path.parent / "db_metadata.json")
            metadata = {
                'paths': self.reference_database['paths'],
                'labels': self.reference_database['labels'],
                'num_images': len(self.reference_database['paths'])
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
            print(f"✓ Saved index and metadata")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def evaluate(self, test_pairs_file: str) -> Dict[str, Any]:
        """
        Evaluate similarity matcher on test pairs dataset
        
        Args:
            test_pairs_file: CSV file with columns: image1_path, image2_path, true_similarity_level
            
        Returns:
            Dict with per-level and overall metrics matching Table 5:
                {
                  "per_level": {
                    "No Match": {"precision": 0.97, "recall": 0.81, "f1": 0.88, "support": 43},
                    "Low": {"precision": 0.92, "recall": 0.94, "f1": 0.93, "support": 50},
                    "Medium": {"precision": 0.90, "recall": 0.93, "f1": 0.91, "support": 56},
                    "High": {"precision": 0.87, "recall": 0.94, "f1": 0.91, "support": 51}
                  },
                  "overall": {"accuracy": 0.91, "precision": 0.913, "recall": 0.91, "f1": 0.9096}
                }
        """
        try:
            if pd is None:
                raise ImportError("pandas required. Install with: pip install pandas")
            
            test_path = Path(test_pairs_file)
            if not test_path.exists():
                raise FileNotFoundError(f"Test pairs file not found: {test_pairs_file}")
            
            logger.info(f"Starting evaluation on {test_pairs_file}")
            
            # Load test data
            df = pd.read_csv(test_pairs_file)
            
            # Similarity levels
            levels = ["No Match", "Low", "Medium", "High"]
            
            # Initialize metrics
            level_metrics = {level: {'tp': 0, 'fp': 0, 'fn': 0} for level in levels}
            total_correct = 0
            total_samples = len(df)
            
            # Evaluate each pair
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
                try:
                    image1_path = row['image1_path']
                    image2_path = row['image2_path']
                    true_level = row['true_similarity_level']
                    
                    # Compute similarity
                    result = self.compute_similarity(image1_path, image2_path)
                    pred_level = result['similarity_level']
                    
                    # Update metrics
                    if pred_level == true_level:
                        level_metrics[true_level]['tp'] += 1
                        total_correct += 1
                    else:
                        level_metrics[true_level]['fn'] += 1
                        level_metrics[pred_level]['fp'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate pair {idx}: {str(e)}")
                    continue
            
            # Compute per-level metrics
            per_level = {}
            for level in levels:
                tp = level_metrics[level]['tp']
                fp = level_metrics[level]['fp']
                fn = level_metrics[level]['fn']
                support = tp + fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / support if support > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                per_level[level] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support
                }
            
            # Compute overall metrics
            overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            # Weighted averages
            total_support = sum([per_level[level]['support'] for level in levels])
            overall_precision = sum([per_level[level]['precision'] * per_level[level]['support'] 
                                    for level in levels]) / total_support if total_support > 0 else 0.0
            overall_recall = sum([per_level[level]['recall'] * per_level[level]['support'] 
                                 for level in levels]) / total_support if total_support > 0 else 0.0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                        if (overall_precision + overall_recall) > 0 else 0.0
            
            result = {
                'per_level': per_level,
                'overall': {
                    'accuracy': overall_accuracy,
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1': overall_f1
                }
            }
            
            logger.info(f"Evaluation complete. Overall Accuracy: {overall_accuracy:.4f}, F1: {overall_f1:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
