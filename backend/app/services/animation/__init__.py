"""
Animation and Transition System (Days 31-32)

Sophisticated animation engine for TikTok/Instagram style video enhancements.
Creates smooth, engaging animations synchronized with emphasis points and music beats.
"""

try:
    from .animation_engine import AnimationEngine
    from .timeline import AnimationTimeline, AnimationEvent
    from .easing import EasingFunction, EasingType
    from .effects import (
        ImageAnimation,
        TextAnimation, 
        TransitionEffect,
        KenBurnsEffect,
        ParticleEffect
    )
    from .synchronizer import BeatSynchronizer, EmphasisSynchronizer
except ImportError as e:
    # Handle import errors gracefully during development
    AnimationEngine = None
    AnimationTimeline = None
    AnimationEvent = None
    EasingFunction = None
    EasingType = None
    ImageAnimation = None
    TextAnimation = None
    TransitionEffect = None
    KenBurnsEffect = None
    ParticleEffect = None
    BeatSynchronizer = None
    EmphasisSynchronizer = None

__all__ = [
    'AnimationEngine',
    'AnimationTimeline',
    'AnimationEvent', 
    'EasingFunction',
    'EasingType',
    'ImageAnimation',
    'TextAnimation',
    'TransitionEffect', 
    'KenBurnsEffect',
    'ParticleEffect',
    'BeatSynchronizer',
    'EmphasisSynchronizer'
] 