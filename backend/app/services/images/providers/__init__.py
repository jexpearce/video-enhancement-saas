"""
Image Providers Package - Phase 2

Multi-source image API integration with:
- Unified provider interface
- Legal compliance validation
- Rate limiting and error handling
- Advanced query optimization
"""

from .base import ImageProvider, ImageResult, ImageLicense
from .wikimedia import WikimediaProvider
from .unsplash import UnsplashProvider

__all__ = [
    'ImageProvider',
    'ImageResult', 
    'ImageLicense',
    'WikimediaProvider',
    'UnsplashProvider'
] 