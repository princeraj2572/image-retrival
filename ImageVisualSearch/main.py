"""
Main entry point for ImageVisualSearch
Orchestrates the image retrieval pipeline
"""

import argparse
import logging
from pathlib import Path

from config import setup_logging, LOG_FILE, LOG_LEVEL
from modules import ObjectDetector, OCREngine, SimilarityMatcher, ImageRetriever
from utils import ImagePreprocessor, setup_logging

# Setup logging
logger = setup_logging(LOG_FILE, LOG_LEVEL)


def main():
    """Main pipeline orchestration"""
    
    parser = argparse.ArgumentParser(
        description="ImageVisualSearch - Image-based visual search system"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "ocr", "retrieval", "similarity"],
        default="detect",
        help="Task to perform on the image"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to reference image database"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for processing"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting ImageVisualSearch pipeline")
    logger.info(f"Task: {args.task}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Initialize modules
        preprocessor = ImagePreprocessor()
        detector = ObjectDetector(device=args.device)
        ocr = OCREngine()
        matcher = SimilarityMatcher(device=args.device)
        retriever = ImageRetriever(device=args.device)
        
        if args.image:
            image_path = Path(args.image)
            
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return
            
            # Preprocess image
            logger.info(f"Processing image: {image_path}")
            processed_img = preprocessor.preprocess(str(image_path))
            
            # Execute requested task
            if args.task == "detect":
                logger.info("Running object detection...")
                detections = detector.detect(processed_img)
                logger.info(f"Detected {len(detections)} objects")
                
            elif args.task == "ocr":
                logger.info("Running OCR...")
                text = ocr.extract_text(processed_img)
                logger.info(f"Extracted text: {text}")
                
            elif args.task == "retrieval":
                if not args.db_path:
                    logger.error("Database path required for retrieval task")
                    return
                logger.info(f"Searching in database: {args.db_path}")
                results = retriever.search(str(image_path), args.db_path, top_k=5)
                logger.info(f"Found {len(results)} similar images")
                
            elif args.task == "similarity":
                logger.info("Computing image features...")
                features = matcher.get_features(processed_img)
                logger.info(f"Feature vector size: {features.shape}")
            
            logger.info("Pipeline completed successfully")
            
        else:
            logger.info("No image provided. Running system check...")
            logger.info("Object Detector: OK")
            logger.info("OCR Engine: OK")
            logger.info("Similarity Matcher: OK")
            logger.info("Image Retriever: OK")
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
