"""
Shared utilities package.

This package contains logging configuration, validators, and other
shared utilities used across the application.
"""

from app.utils.logging_config import setup_logging, get_logger

__all__ = ['setup_logging', 'get_logger']
