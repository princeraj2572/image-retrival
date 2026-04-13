"""
Gradio Web UI for ImageVisualSearch
Provides an interactive web interface for image retrieval and analysis
"""

import logging
import os
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def create_gradio_app():
    """
    Create Gradio web application
    
    Returns:
        Gradio Interface object
    """
    try:
        import gradio as gr
        from PIL import Image
        from modules import ObjectDetector, OCREngine, SimilarityMatcher
        from utils import ImagePreprocessor
        
        # Initialize modules
        preprocessor = ImagePreprocessor()
        detector = ObjectDetector(device="cpu")
        ocr = OCREngine()
        matcher = SimilarityMatcher(device="cpu")
        
        logger.info("UI modules initialized")
        
        # Define processing functions
        def detect_objects(image):
            """Object detection interface"""
            if image is None:
                return None, "No image provided"
            
            try:
                # Convert PIL image to CV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect
                detections = detector.detect(cv_image)
                
                # Visualize
                vis_image = detector.visualize(cv_image, detections)
                
                # Convert back to RGB
                vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                
                # Create result message
                result_text = f"Detected {len(detections)} objects:\n"
                for i, det in enumerate(detections):
                    result_text += f"{i+1}. {det['class_name']}: {det['confidence']:.2%}\n"
                
                return vis_image_rgb, result_text
                
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                return None, f"Error: {str(e)}"
        
        def extract_text(image):
            """OCR interface"""
            if image is None:
                return "No image provided"
            
            try:
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                text = ocr.extract_text(cv_image)
                return text if text else "No text detected"
                
            except Exception as e:
                logger.error(f"OCR failed: {e}")
                return f"Error: {str(e)}"
        
        def compare_images(image1, image2):
            """Image comparison interface"""
            if image1 is None or image2 is None:
                return 0.0, None, "Provide both images"
            
            try:
                cv_image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
                cv_image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
                
                similarity = matcher.is_similar(cv_image1, cv_image2)
                
                # Visualize
                vis_image = matcher.visualize_similarity(cv_image1, cv_image2)
                vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                
                features1 = matcher.get_features(cv_image1)
                features2 = matcher.get_features(cv_image2)
                sim_score = matcher.compute_similarity(features1, features2)
                
                result_text = f"Similarity Score: {sim_score:.2%}\nSimilar: {'Yes' if similarity else 'No'}"
                
                return sim_score, vis_image_rgb, result_text
                
            except Exception as e:
                logger.error(f"Comparison failed: {e}")
                return 0.0, None, f"Error: {str(e)}"
        
        # Create Gradio interface
        with gr.Blocks(title="ImageVisualSearch", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# ImageVisualSearch - Computer Vision System")
            gr.Markdown("Advanced image analysis using YOLOv8, Tesseract OCR, and ResNet")
            
            with gr.Tabs():
                # Detection Tab
                with gr.Tab("Object Detection"):
                    with gr.Row():
                        with gr.Column():
                            det_input = gr.Image(label="Upload Image", type="pil")
                            det_button = gr.Button("Detect Objects", variant="primary")
                        with gr.Column():
                            det_output_image = gr.Image(label="Detection Results")
                            det_output_text = gr.Textbox(label="Results", lines=5)
                    
                    det_button.click(
                        detect_objects,
                        inputs=[det_input],
                        outputs=[det_output_image, det_output_text]
                    )
                
                # OCR Tab
                with gr.Tab("Text Recognition (OCR)"):
                    with gr.Row():
                        with gr.Column():
                            ocr_input = gr.Image(label="Upload Image", type="pil")
                            ocr_button = gr.Button("Extract Text", variant="primary")
                        with gr.Column():
                            ocr_output = gr.Textbox(label="Extracted Text", lines=10)
                    
                    ocr_button.click(
                        extract_text,
                        inputs=[ocr_input],
                        outputs=[ocr_output]
                    )
                
                # Similarity Tab
                with gr.Tab("Image Similarity"):
                    with gr.Row():
                        with gr.Column():
                            sim_input1 = gr.Image(label="Image 1", type="pil")
                        with gr.Column():
                            sim_input2 = gr.Image(label="Image 2", type="pil")
                    
                    with gr.Row():
                        sim_button = gr.Button("Compare Images", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            sim_score = gr.Number(label="Similarity Score", interactive=False)
                        with gr.Column():
                            sim_output_image = gr.Image(label="Comparison Visualization")
                        with gr.Column():
                            sim_output_text = gr.Textbox(label="Result", lines=3)
                    
                    sim_button.click(
                        compare_images,
                        inputs=[sim_input1, sim_input2],
                        outputs=[sim_score, sim_output_image, sim_output_text]
                    )
                
                # About Tab
                with gr.Tab("About"):
                    gr.Markdown("""
                    ## ImageVisualSearch System
                    
                    ### Features:
                    - **Object Detection**: Real-time detection using YOLOv8
                    - **Text Recognition**: OCR using Tesseract
                    - **Image Similarity**: Deep learning-based image comparison
                    
                    ### Technologies:
                    - YOLOv8 for object detection
                    - ResNet50 for feature extraction
                    - Tesseract for OCR
                    - Gradio for web interface
                    
                    ### System Information:
                    - Python 3.10+
                    - PyTorch 2.1.0
                    - TorchVision 0.16.0
                    - OpenCV 4.8.1
                    
                    ### Project Repository:
                    [GitHub: image-visual-search](https://github.com/yourusername/image-visual-search)
                    """)
        
        return demo
        
    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        raise


def launch_app(share: bool = False, port: int = 7860, debug: bool = False):
    """
    Launch the Gradio web application
    
    Args:
        share: Whether to share the app publicly
        port: Port to run the app on
        debug: Whether to run in debug mode
    """
    from utils import setup_logging
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    logger.info(f"Launching ImageVisualSearch UI...")
    logger.info(f"Port: {port}")
    logger.info(f"Share: {share}")
    
    try:
        app = create_gradio_app()
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=share,
            debug=debug
        )
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch ImageVisualSearch UI")
    parser.add_argument("--share", action="store_true", help="Share app publicly")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    launch_app(share=args.share, port=args.port, debug=args.debug)
