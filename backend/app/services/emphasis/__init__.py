"""
Emphasis Detection Module

This module provides multi-modal emphasis detection capabilities combining
acoustic, prosodic, and linguistic analysis for accurate identification
of emphasized words in speech.
"""

from .detector import MultiModalEmphasisDetector, EmphasisResult
from .acoustic_analyzer import AcousticAnalyzer
from .prosodic_analyzer import ProsodicAnalyzer
from .linguistic_analyzer import LinguisticAnalyzer

__all__ = [
    'MultiModalEmphasisDetector',
    'EmphasisResult',
    'AcousticAnalyzer',
    'ProsodicAnalyzer',
    'LinguisticAnalyzer'
]
