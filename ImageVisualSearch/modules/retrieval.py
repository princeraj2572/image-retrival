"""
Image Retrieval Module
Performs efficient image retrieval from large databases using indexing
"""

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class ImageRetriever:
    """
    Image Retrieval System using FAISS indexing
    
    Attributes:
        device (str): Device to use (cpu or cuda)
        model: Feature extraction model
        index: FAISS index for similarity search
    """
    
    def __init__(self, device: str = "cpu", index_type: str = "flat"):
        """
        Initialize ImageRetriever
        
        Args:
            device: Device to use (cpu or cuda)
            index_type: Type of FAISS index (flat, ivf, hnsw)
        """
        try:
            import faiss
            from modules.similarity import SimilarityMatcher
            
            self.faiss = faiss
            self.device = device
            self.index_type = index_type
            self.similarity_matcher = SimilarityMatcher(device=device)
            self.index = None
            self.image_paths = []
            
            logger.info(f"ImageRetriever initialized with {index_type} index")
            
        except ImportError:
            logger.error("faiss-cpu or faiss-gpu required. Install with: pip install faiss-cpu")
            raise
    
    def build_index(self, image_paths: List[str]) -> None:
        """
        Build FAISS index from images
        
        Args:
            image_paths: List of paths to images
        """
        try:
            import cv2
            
            logger.info(f"Building index for {len(image_paths)} images...")
            
            features_list = []
            valid_paths = []
            
            for idx, image_path in enumerate(image_paths):
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        features = self.similarity_matcher.get_features(image)
                        features_list.append(features)
                        valid_paths.append(image_path)
                        
                        if (idx + 1) % 100 == 0:
                            logger.info(f"Processed {idx + 1}/{len(image_paths)} images")
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue
            
            if not features_list:
                logger.error("No valid images found")
                return
            
            # Convert to numpy array
            features_array = np.array(features_list).astype('float32')
            
            # Create index
            if self.index_type == "flat":
                self.index = self.faiss.IndexFlatL2(features_array.shape[1])
            elif self.index_type == "ivf":
                quantizer = self.faiss.IndexFlatL2(features_array.shape[1])
                self.index = self.faiss.IndexIVFFlat(quantizer, features_array.shape[1], 100)
            
            # Add vectors to index
            self.index.add(features_array)
            self.image_paths = valid_paths
            
            logger.info(f"Index built successfully with {len(valid_paths)} images")
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            raise
    
    def search(self, query_image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar images
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            
        Returns:
            List of results with paths and similarity scores
        """
        try:
            import cv2
            
            if self.index is None:
                logger.error("Index not built. Call build_index first.")
                return []
            
            # Load and extract features from query image
            query_image = cv2.imread(query_image_path)
            query_features = self.similarity_matcher.get_features(query_image)
            query_features = np.array([query_features]).astype('float32')
            
            # Search in index
            distances, indices = self.index.search(query_features, top_k)
            
            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.image_paths):
                    # Convert L2 distance to similarity score
                    similarity = 1 / (1 + distance)
                    
                    result = {
                        'path': self.image_paths[idx],
                        'similarity': similarity,
                        'distance': distance
                    }
                    results.append(result)
            
            logger.debug(f"Found {len(results)} similar images")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def save_index(self, index_path: str) -> None:
        """
        Save index to disk
        
        Args:
            index_path: Path to save index
        """
        try:
            if self.index is None:
                logger.error("Index not built")
                return
            
            # Save FAISS index
            self.faiss.write_index(self.index, str(index_path) + ".index")
            
            # Save image paths
            metadata = {'image_paths': self.image_paths}
            with open(str(index_path) + ".pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, index_path: str) -> None:
        """
        Load index from disk
        
        Args:
            index_path: Path to load index from
        """
        try:
            # Load FAISS index
            self.index = self.faiss.read_index(str(index_path) + ".index")
            
            # Load metadata
            with open(str(index_path) + ".pkl", 'rb') as f:
                metadata = pickle.load(f)
                self.image_paths = metadata['image_paths']
            
            logger.info(f"Index loaded from {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        return {
            'index_type': self.index_type,
            'num_images': len(self.image_paths),
            'index_size': self.index.ntotal if self.index else 0,
            'device': self.device
        }
