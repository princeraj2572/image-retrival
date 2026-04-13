"""
Information Retrieval Module
Complete pipeline combining all modules: YOLO detection, OCR, visual similarity, and web search
"""

import logging
import json
import time
from datetime import datetime
from typing import Union, List, Dict, Optional, Any
from pathlib import Path
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Import from project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from modules.detection import ObjectDetector
from modules.ocr_engine import OCREngine
from modules.similarity import VisualSimilarityMatcher

logger = logging.getLogger(__name__)


class InformationRetriever:
    """
    Complete Information Retrieval Pipeline
    Combines YOLO detection, Tesseract OCR, ResNet50 similarity, and web search
    
    Attributes:
        detector: ObjectDetector instance
        ocr_engine: OCREngine instance
        similarity_matcher: VisualSimilarityMatcher instance
        search_api_key: Google Custom Search API key
        search_engine_id: Google Custom Search engine ID
    """
    
    def __init__(self) -> None:
        """
        Initialize InformationRetriever with all 3 modules and search APIs
        
        Raises:
            ImportError: If required modules not available
        """
        try:
            logger.info("Initializing Information Retrieval Pipeline...")
            
            # Initialize all 3 modules
            self.detector = ObjectDetector()
            self.ocr_engine = OCREngine()
            self.similarity_matcher = VisualSimilarityMatcher()
            
            # Setup search APIs
            self.search_api_key = Config.SEARCH_API.get('API_KEY') if hasattr(Config, 'SEARCH_API') else None
            self.search_engine_id = Config.SEARCH_API.get('ENGINE_ID') if hasattr(Config, 'SEARCH_API') else None
            
            # Fallback if API not configured
            if not self.search_api_key or not self.search_engine_id:
                logger.warning("Google Custom Search API not configured, will use DuckDuckGo fallback")
                self.use_google = False
            else:
                self.use_google = True
            
            # Initialize logger
            logger.info("Information Retrieval Pipeline ready")
            print("✓ Information Retrieval Pipeline ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize InformationRetriever: {str(e)}")
            raise
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run all 3 modules in parallel on a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with combined analysis results from all 3 modules
        """
        try:
            start_time = time.time()
            
            logger.info(f"Analyzing image: {image_path}")
            
            # Dictionary to store results
            results = {
                'detections': None,
                'ocr_result': None,
                'similar_images': None,
                'error': None
            }
            
            # Run all 3 modules in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Thread 1: Object Detection
                detection_future = executor.submit(self._run_detection, image_path)
                
                # Thread 2: OCR Extraction
                ocr_future = executor.submit(self._run_ocr, image_path)
                
                # Thread 3: Visual Similarity
                similarity_future = executor.submit(self._run_similarity, image_path)
                
                # Collect results
                try:
                    results['detections'] = detection_future.result(timeout=60)
                except Exception as e:
                    logger.warning(f"Detection failed: {str(e)}")
                    results['detections'] = []
                
                try:
                    results['ocr_result'] = ocr_future.result(timeout=60)
                except Exception as e:
                    logger.warning(f"OCR failed: {str(e)}")
                    results['ocr_result'] = {}
                
                try:
                    results['similar_images'] = similarity_future.result(timeout=60)
                except Exception as e:
                    logger.warning(f"Similarity search failed: {str(e)}")
                    results['similar_images'] = []
            
            # Extract dominant objects
            dominant_objects = []
            if results['detections']:
                dominant_objects = self.detector.get_dominant_objects(
                    results['detections'],
                    top_n=3
                )
            
            # Extract text
            extracted_text = results['ocr_result'].get('text', '') if results['ocr_result'] else ''
            
            processing_time = time.time() - start_time
            
            return {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'detections': results['detections'],
                'ocr_result': results['ocr_result'],
                'similar_images': results['similar_images'],
                'dominant_objects': dominant_objects,
                'extracted_text': extracted_text,
                'processing_time_seconds': processing_time
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            raise
    
    def _run_detection(self, image_path: str) -> List[Dict]:
        """Run object detection"""
        try:
            return self.detector.detect(image_path)
        except Exception as e:
            logger.warning(f"Detection error: {str(e)}")
            return []
    
    def _run_ocr(self, image_path: str) -> Dict[str, Any]:
        """Run OCR extraction"""
        try:
            return self.ocr_engine.preprocess_and_extract(image_path, auto_detect=True)
        except Exception as e:
            logger.warning(f"OCR error: {str(e)}")
            return {}
    
    def _run_similarity(self, image_path: str) -> List[Dict]:
        """Run visual similarity search"""
        try:
            return self.similarity_matcher.find_similar(image_path, top_k=3)
        except Exception as e:
            logger.warning(f"Similarity error: {str(e)}")
            return []
    
    def build_query(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Build multi-faceted search queries from analysis results
        
        Args:
            analysis_result: Result from analyze_image()
            
        Returns:
            Dict with primary_query, object_query, text_query, combined_query
        """
        try:
            primary_query = ""
            object_query = ""
            text_query = ""
            query_source = "unknown"
            
            # Extract components
            dominant_objects = analysis_result.get('dominant_objects', [])
            extracted_text = analysis_result.get('extracted_text', '').strip()
            ocr_result = analysis_result.get('ocr_result', {})
            
            # Build primary query from top object + text
            if dominant_objects:
                top_object = dominant_objects[0]
                primary_query = f"{top_object}"
                
                if extracted_text:
                    # Add text, truncate to 60 chars total
                    text_part = extracted_text[:40]
                    primary_query = f"{top_object} {text_part}"
                    query_source = "object+ocr"
                else:
                    query_source = "object_only"
            elif extracted_text:
                primary_query = extracted_text[:60]
                query_source = "ocr_only"
            else:
                # Use similarity if no detections/text
                similar = analysis_result.get('similar_images', [])
                if similar:
                    primary_query = f"images similar to {similar[0].get('label', 'reference')}"
                    query_source = "similarity_only"
            
            # Build object-specific query
            if dominant_objects:
                top_object = dominant_objects[0]
                object_query = f"what is a {top_object}"
            
            # Build text query
            if ocr_result and 'text' in ocr_result:
                try:
                    text_query = self.ocr_engine.build_search_query_from_text(ocr_result['text'])
                except:
                    text_query = extracted_text[:60]
            
            # Build combined query
            queries = [q for q in [primary_query, object_query, text_query] if q]
            combined_query = " ".join(queries)
            
            if not combined_query:
                combined_query = "image search"
            
            return {
                'primary_query': primary_query,
                'object_query': object_query,
                'text_query': text_query,
                'combined_query': combined_query[:200],  # Limit length
                'query_source': query_source
            }
            
        except Exception as e:
            logger.warning(f"Query building failed: {str(e)}")
            return {
                'primary_query': 'image search',
                'object_query': '',
                'text_query': '',
                'combined_query': 'image search',
                'query_source': 'error'
            }
    
    def search_google(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using Google Custom Search API
        
        Args:
            query: Search query string
            num_results: Number of results. Default: 5
            
        Returns:
            List of result dicts with title, link, snippet, source
        """
        try:
            if not self.search_api_key or not self.search_engine_id:
                logger.warning("Google API not configured")
                return []
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.search_api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10)
            }
            
            logger.debug(f"Searching Google for: {query}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle API errors
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown error')
                logger.error(f"Google API error: {error_msg}")
                return []
            
            results = []
            for item in data.get('items', []):
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google'
                }
                results.append(result)
            
            logger.info(f"Google search returned {len(results)} results")
            return results
            
        except requests.exceptions.Timeout:
            logger.error("Google API timeout")
            return []
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            return []
    
    def search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback web search using DuckDuckGo HTML endpoint
        
        Args:
            query: Search query string
            num_results: Number of results. Default: 5
            
        Returns:
            List of result dicts with title, link, snippet, source
        """
        try:
            if BeautifulSoup is None:
                logger.warning("BeautifulSoup not installed, skipping DuckDuckGo search")
                return []
            
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            
            # Use headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            logger.debug(f"Searching DuckDuckGo for: {query}")
            
            # Retry logic: 3 retries with 2 second delay
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt < 2:
                        logger.debug(f"Timeout, retrying... ({attempt + 1}/3)")
                        time.sleep(2)
                    else:
                        raise
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            result_divs = soup.find_all('div', class_='result')
            
            for result_div in result_divs[:num_results]:
                try:
                    title_elem = result_div.find('a', class_='result__a')
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        link = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True)
                        
                        # Clean up link if needed
                        if link and link.startswith('http'):
                            result = {
                                'title': title,
                                'link': link,
                                'snippet': snippet,
                                'source': 'duckduckgo'
                            }
                            results.append(result)
                except Exception as e:
                    logger.debug(f"Failed to parse result: {str(e)}")
                    continue
            
            logger.info(f"DuckDuckGo search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    def retrieve(self, image_path: str) -> Dict[str, Any]:
        """
        Main information retrieval pipeline
        
        Args:
            image_path: Path to image file
            
        Returns:
            Complete result dict with analysis, queries, search results, summary
        """
        try:
            start_time = time.time()
            
            logger.info(f"Starting information retrieval for: {image_path}")
            
            # Step 1: Analyze image
            analysis = self.analyze_image(image_path)
            
            # Step 2: Build search queries
            queries = self.build_query(analysis)
            
            # Step 3: Perform web search
            search_results = []
            
            try:
                # Try Google first
                if self.use_google:
                    search_results = self.search_google(
                        queries['combined_query'],
                        num_results=5
                    )
                
                # Fallback to DuckDuckGo if Google fails or not configured
                if not search_results:
                    logger.info("Falling back to DuckDuckGo search...")
                    search_results = self.search_duckduckgo(
                        queries['combined_query'],
                        num_results=5
                    )
                
            except Exception as e:
                logger.warning(f"Search failed: {str(e)}, continuing with local analysis only")
            
            # Step 4: Add relevance ranks
            for idx, result in enumerate(search_results, 1):
                result['relevance_rank'] = idx
            
            # Step 5: Generate summary
            summary = self.generate_summary(analysis, search_results)
            
            # Compile final results
            processing_time = time.time() - start_time
            
            final_result = {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'detections': analysis.get('detections', []),
                    'ocr': analysis.get('ocr_result', {}),
                    'similar_images': analysis.get('similar_images', [])
                },
                'queries': queries,
                'search_results': search_results,
                'summary': summary,
                'processing_time_seconds': processing_time
            }
            
            logger.info(f"Information retrieval completed in {processing_time:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Information retrieval failed: {str(e)}")
            raise
    
    def retrieve_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple images with progress bar
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of result dicts (one per image)
        """
        try:
            results = []
            output_dir = Config.RESULTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {len(image_paths)} images...")
            
            for idx, image_path in enumerate(tqdm(image_paths, desc="Retrieving information"), 1):
                try:
                    result = self.retrieve(image_path)
                    results.append(result)
                    
                    # Save result to JSON
                    filename = Path(image_path).stem
                    result_file = output_dir / f"retrieval_{filename}_{int(time.time())}.json"
                    
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=4)
                    
                    logger.debug(f"Saved result to {result_file}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {str(e)}")
                    continue
            
            logger.info(f"Batch processing completed. Processed {len(results)}/{len(image_paths)} images")
            return results
            
        except Exception as e:
            logger.error(f"Batch retrieval failed: {str(e)}")
            raise
    
    def generate_summary(
        self,
        analysis: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable 2-3 sentence summary
        
        Args:
            analysis: Analysis result from analyze_image()
            search_results: Search results from web search
            
        Returns:
            Summary string
        """
        try:
            summary_parts = []
            
            # Get detected objects
            objects = analysis.get('dominant_objects', [])
            if objects:
                obj_text = ', '.join(objects[:2])
                summary_parts.append(f"This image contains {obj_text}")
            
            # Get OCR text
            ocr_text = analysis.get('extracted_text', '').strip()
            if ocr_text and len(ocr_text) > 5:
                # Truncate long text
                if len(ocr_text) > 60:
                    ocr_text = ocr_text[:60] + "..."
                summary_parts.append(f"The text '{ocr_text}' was detected")
            
            # Get top search result
            if search_results:
                top_snippet = search_results[0].get('snippet', '').strip()
                if top_snippet:
                    if len(top_snippet) > 100:
                        top_snippet = top_snippet[:100] + "..."
                    summary_parts.append(f"Based on analysis, this appears related to: {top_snippet}")
            
            # Build final summary
            if summary_parts:
                summary = ". ".join(summary_parts) + "."
            else:
                summary = "Analysis could not extract sufficient information from the image."
            
            return summary
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {str(e)}")
            return "Unable to generate summary."
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
