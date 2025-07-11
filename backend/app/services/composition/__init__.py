"""
Video Composition Services

This package provides video composition and rendering capabilities,
integrating with the existing animation, style, and curation systems.
"""

from .video_composer import VideoComposer, CompositionConfig
from .models import CompositionResult, CompositionError

__all__ = [
    'VideoComposer',
    'CompositionConfig',
    'CompositionResult',
    'CompositionError'
]
