"""
Helper Functions Module
Utility functions for logging, configuration, and common tasks
"""

import logging
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "app.log", log_level: str = "INFO") -> logging.Logger:
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
