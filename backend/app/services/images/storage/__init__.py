"""
Image storage and CDN management system.

This module provides scalable image storage with AWS S3 and CloudFront CDN integration.
Features include multi-size image processing, smart cropping, WebP optimization, and
cache management.
"""

from .config import StorageConfig
from .models import StoredImage, ImageStorageError
from .s3_manager import ImageStorageManager

__all__ = [
    'StorageConfig',
    'StoredImage', 
    'ImageStorageError',
    'ImageStorageManager'
] 