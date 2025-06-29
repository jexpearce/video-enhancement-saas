"""
Style System Architecture (Days 29-30).

This module provides trendy, eye-catching visual styles optimized for
TikTok and Instagram Reel creators with automatic selection and manual override.
"""

from .models import (
    VisualStyle,
    StyleTemplate,
    AnimationEffect,
    TypographyStyle,
    ColorScheme,
    StyleConfig
)
from .style_engine import StyleEngine
from .template_manager import TemplateManager
from .auto_selector import AutoStyleSelector

__all__ = [
    'VisualStyle',
    'StyleTemplate', 
    'AnimationEffect',
    'TypographyStyle',
    'ColorScheme',
    'StyleConfig',
    'StyleEngine',
    'TemplateManager',
    'AutoStyleSelector'
] 