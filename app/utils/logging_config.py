"""
Logging configuration for the recommender system.

Provides structured logging with file and console handlers for
development and production environments.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_file: Name of log file (default: None, logs to console only)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_dir: Directory for log files (default: 'logs')
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    """
    # Create log directory if needed
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        full_log_path = log_path / log_file
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always active)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        file_handler = RotatingFileHandler(
            full_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {full_log_path}")
    
    # Silence noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Optional logging level override
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger


# Default configuration
def configure_inference_logging(debug: bool = False):
    """
    Configure logging for inference operations.
    
    Args:
        debug: Enable debug logging (default: False)
    """
    level = "DEBUG" if debug else "INFO"
    setup_logging(
        log_file="inference.log",
        level=level,
        log_dir="logs"
    )


def configure_training_logging(debug: bool = False):
    """
    Configure logging for training operations.
    
    Args:
        debug: Enable debug logging (default: False)
    """
    level = "DEBUG" if debug else "INFO"
    setup_logging(
        log_file="training.log",
        level=level,
        log_dir="logs"
    )


def configure_api_logging(debug: bool = False):
    """
    Configure logging for API operations.
    
    Args:
        debug: Enable debug logging (default: False)
    """
    level = "DEBUG" if debug else "INFO"
    setup_logging(
        log_file="api.log",
        level=level,
        log_dir="logs"
    )


if __name__ == "__main__":
    # Test logging configuration
    configure_inference_logging(debug=True)
    logger = get_logger(__name__)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    print("\nLogging test complete. Check logs/inference.log")
