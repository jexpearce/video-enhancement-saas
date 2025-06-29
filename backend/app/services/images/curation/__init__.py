"""
Image Curation System - Phase 2 Days 23-24

ML-powered image curation with CLIP-based relevance scoring,
face detection for person entities, and advanced quality assessment.
"""

from .curator import ImageCurator, CuratedImage
from .cache_manager import ImageCacheManager
from .validator import LegalComplianceValidator, ValidationResult
from .quality_assessor import ImageQualityAssessor

__all__ = [
    'ImageCurator',
    'CuratedImage', 
    'ImageCacheManager',
    'LegalComplianceValidator',
    'ValidationResult',
    'ImageQualityAssessor'
] 