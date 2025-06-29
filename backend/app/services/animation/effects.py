"""
Animation Effects Library

Provides specialized animation effects for different types of content enhancements:
- Image animations (entrance, exit, transitions)
- Text animations and overlays
- Special effects (Ken Burns, particles, etc.)
- Transition effects between scenes
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import math


class AnimationType(Enum):
    """Types of animations available."""
    
    # Basic animations
    FADE = "fade"
    SLIDE = "slide" 
    ZOOM = "zoom"
    BOUNCE = "bounce"
    
    # Advanced animations
    ZOOM_BLAST = "zoom_blast"
    NEON_FLASH = "neon_flash"
    GLITCH_POP = "glitch_pop"
    PARTICLE_BURST = "particle_burst"
    WAVE_REVEAL = "wave_reveal"
    
    # Text specific
    TYPEWRITER = "typewriter"
    TEXT_BOUNCE = "text_bounce"
    WORD_BY_WORD = "word_by_word"


class EffectIntensity(Enum):
    """Effect intensity levels."""
    
    SUBTLE = "subtle"
    MEDIUM = "medium"
    HIGH = "high"
    VIRAL = "viral"  # Maximum intensity for viral content


@dataclass
class AnimationProperties:
    """Base properties for all animations."""
    
    duration: float = 1.0
    delay: float = 0.0
    easing: str = "ease_out_quart"
    intensity: EffectIntensity = EffectIntensity.MEDIUM
    
    # Transform properties
    scale_from: float = 1.0
    scale_to: float = 1.0
    opacity_from: float = 1.0
    opacity_to: float = 1.0
    
    # Position properties
    translate_x_from: float = 0.0
    translate_x_to: float = 0.0
    translate_y_from: float = 0.0
    translate_y_to: float = 0.0
    
    # Rotation properties
    rotate_from: float = 0.0
    rotate_to: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'duration': self.duration,
            'delay': self.delay,
            'easing': self.easing,
            'intensity': self.intensity.value,
            'from': {
                'scale': self.scale_from,
                'opacity': self.opacity_from,
                'translateX': self.translate_x_from,
                'translateY': self.translate_y_from,
                'rotate': self.rotate_from
            },
            'to': {
                'scale': self.scale_to,
                'opacity': self.opacity_to,
                'translateX': self.translate_x_to,
                'translateY': self.translate_y_to,
                'rotate': self.rotate_to
            }
        }


class ImageAnimation:
    """
    Handles image entrance, exit, and transition animations.
    Optimized for social media content with eye-catching effects.
    """
    
    def __init__(self):
        """Initialize image animation handler."""
        self.animation_presets = self._create_animation_presets()
    
    def create_entrance_animation(
        self, 
        animation_type: AnimationType,
        intensity: EffectIntensity = EffectIntensity.MEDIUM,
        duration: float = 0.8
    ) -> AnimationProperties:
        """Create entrance animation for images."""
        
        props = AnimationProperties(duration=duration, intensity=intensity)
        
        if animation_type == AnimationType.FADE:
            props.opacity_from = 0.0
            props.opacity_to = 1.0
        elif animation_type == AnimationType.SLIDE:
            direction = random.choice(['left', 'right', 'top', 'bottom'])
            distance = 100 if intensity != EffectIntensity.VIRAL else 150
            
            props.opacity_from = 0.0
            props.opacity_to = 1.0
            
            if direction == 'left':
                props.translate_x_from = -distance
            elif direction == 'right':
                props.translate_x_from = distance
            elif direction == 'top':
                props.translate_y_from = -distance
            elif direction == 'bottom':
                props.translate_y_from = distance
            
            props.easing = "ease_out_back" if intensity == EffectIntensity.VIRAL else "ease_out_quart"
        elif animation_type == AnimationType.ZOOM:
            props.opacity_from = 0.0
            props.opacity_to = 1.0
            props.scale_from = 0.5
            props.scale_to = 1.0
        elif animation_type == AnimationType.BOUNCE:
            props.opacity_from = 0.0
            props.opacity_to = 1.0
            props.scale_from = 0.3
            props.scale_to = 1.0
            props.translate_y_from = -50
            props.translate_y_to = 0
            props.easing = "ease_out_bounce"
        elif animation_type == AnimationType.ZOOM_BLAST:
            props.opacity_from = 0.0
            props.opacity_to = 1.0
            props.scale_from = 0.1
            props.scale_to = 1.0
            props.easing = "ease_out_back"
            props.duration *= 1.2  # Longer for dramatic effect
        elif animation_type == AnimationType.GLITCH_POP:
            props.opacity_from = 0.0
            props.opacity_to = 1.0
            props.scale_from = 0.8
            props.scale_to = 1.0
            # Add slight rotation for glitch effect
            props.rotate_from = random.uniform(-5, 5)
            props.rotate_to = 0
            props.easing = "ease_out_elastic"
        elif animation_type == AnimationType.PARTICLE_BURST:
            props.opacity_from = 0.0
            props.opacity_to = 1.0
            props.scale_from = 0.1
            props.scale_to = 1.0
            props.easing = "ease_out_quart"
            # This would trigger particle effects in the renderer
        else:
            props.opacity_from = 0.0
            props.opacity_to = 1.0
            props.easing = "ease_out_quart"
        
        return props
    
    def create_exit_animation(
        self,
        animation_type: AnimationType,
        intensity: EffectIntensity = EffectIntensity.MEDIUM,
        duration: float = 0.5
    ) -> AnimationProperties:
        """Create exit animation for images."""
        
        # Exit animations are usually the reverse of entrance
        entrance = self.create_entrance_animation(animation_type, intensity, duration)
        
        # Swap from and to values
        return AnimationProperties(
            duration=duration,
            easing="ease_in_quart",  # Different easing for exit
            intensity=intensity,
            scale_from=entrance.scale_to,
            scale_to=entrance.scale_from,
            opacity_from=entrance.opacity_to,
            opacity_to=entrance.opacity_from,
            translate_x_from=entrance.translate_x_to,
            translate_x_to=entrance.translate_x_from,
            translate_y_from=entrance.translate_y_to,
            translate_y_to=entrance.translate_y_from,
            rotate_from=entrance.rotate_to,
            rotate_to=entrance.rotate_from
        )
    
    def _create_animation_presets(self) -> Dict[str, AnimationProperties]:
        """Create preset animations for quick access."""
        return {
            'viral_entrance': self.create_entrance_animation(
                AnimationType.ZOOM_BLAST, EffectIntensity.VIRAL
            ),
            'smooth_entrance': self.create_entrance_animation(
                AnimationType.FADE, EffectIntensity.MEDIUM
            ),
            'bouncy_entrance': self.create_entrance_animation(
                AnimationType.BOUNCE, EffectIntensity.HIGH
            )
        }


class TextAnimation:
    """
    Handles text and caption animations synchronized with video content.
    """
    
    def __init__(self):
        """Initialize text animation handler."""
        pass
    
    def create_caption_animation(
        self,
        text: str,
        animation_type: AnimationType = AnimationType.FADE,
        emphasis_score: float = 0.5,
        duration: float = 1.5
    ) -> Dict[str, Any]:
        """Create caption animation based on emphasis."""
        
        # Adjust animation based on emphasis score
        intensity = self._calculate_text_intensity(emphasis_score)
        
        if animation_type == AnimationType.TYPEWRITER:
            return self._create_typewriter_effect(text, duration, intensity)
        elif animation_type == AnimationType.TEXT_BOUNCE:
            return self._create_text_bounce(text, duration, intensity)
        elif animation_type == AnimationType.WORD_BY_WORD:
            return self._create_word_by_word(text, duration, intensity)
        else:
            return self._create_text_fade(text, duration, intensity)
    
    def _calculate_text_intensity(self, emphasis_score: float) -> EffectIntensity:
        """Calculate text animation intensity from emphasis score."""
        if emphasis_score > 0.8:
            return EffectIntensity.VIRAL
        elif emphasis_score > 0.6:
            return EffectIntensity.HIGH
        elif emphasis_score > 0.4:
            return EffectIntensity.MEDIUM
        else:
            return EffectIntensity.SUBTLE
    
    def _create_typewriter_effect(
        self, 
        text: str, 
        duration: float, 
        intensity: EffectIntensity
    ) -> Dict[str, Any]:
        """Create typewriter effect for text."""
        
        char_delay = duration / len(text) if text else 0.1
        
        return {
            'type': 'typewriter',
            'text': text,
            'duration': duration,
            'char_delay': char_delay,
            'intensity': intensity.value,
            'properties': {
                'cursor': True if intensity != EffectIntensity.SUBTLE else False,
                'sound': True if intensity == EffectIntensity.VIRAL else False
            }
        }
    
    def _create_text_bounce(
        self, 
        text: str, 
        duration: float, 
        intensity: EffectIntensity
    ) -> Dict[str, Any]:
        """Create bouncing text effect."""
        
        bounce_height = {
            EffectIntensity.SUBTLE: 5,
            EffectIntensity.MEDIUM: 10,
            EffectIntensity.HIGH: 20,
            EffectIntensity.VIRAL: 30
        }[intensity]
        
        return {
            'type': 'text_bounce',
            'text': text,
            'duration': duration,
            'properties': {
                'bounce_height': bounce_height,
                'easing': 'ease_out_bounce',
                'stagger_delay': 0.1
            }
        }
    
    def _create_word_by_word(
        self, 
        text: str, 
        duration: float, 
        intensity: EffectIntensity
    ) -> Dict[str, Any]:
        """Create word-by-word reveal effect."""
        
        words = text.split()
        word_delay = duration / len(words) if words else 0.5
        
        return {
            'type': 'word_by_word',
            'text': text,
            'duration': duration,
            'properties': {
                'word_delay': word_delay,
                'animation': 'fade_up',
                'intensity': intensity.value
            }
        }
    
    def _create_text_fade(
        self, 
        text: str, 
        duration: float, 
        intensity: EffectIntensity
    ) -> Dict[str, Any]:
        """Create simple text fade effect."""
        
        return {
            'type': 'text_fade',
            'text': text,
            'duration': duration,
            'properties': {
                'easing': 'ease_out_quart',
                'from': {'opacity': 0, 'translateY': 10},
                'to': {'opacity': 1, 'translateY': 0}
            }
        }


class KenBurnsEffect:
    """
    Implements the Ken Burns effect for dynamic image movement.
    Creates subtle pan and zoom effects to add life to static images.
    """
    
    def __init__(self):
        """Initialize Ken Burns effect handler."""
        self.movement_patterns = self._create_movement_patterns()
    
    def create_ken_burns_animation(
        self,
        duration: float,
        intensity: float = 0.3,
        pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create Ken Burns pan and zoom animation."""
        
        if pattern and pattern in self.movement_patterns:
            movement = self.movement_patterns[pattern]
        else:
            movement = self._generate_random_movement(intensity)
        
        return {
            'type': 'ken_burns',
            'duration': duration,
            'properties': {
                'zoom_start': movement['zoom_start'],
                'zoom_end': movement['zoom_end'],
                'pan_x_start': movement['pan_x_start'],
                'pan_x_end': movement['pan_x_end'],
                'pan_y_start': movement['pan_y_start'],
                'pan_y_end': movement['pan_y_end'],
                'easing': 'linear'
            }
        }
    
    def _generate_random_movement(self, intensity: float) -> Dict[str, float]:
        """Generate random but controlled movement."""
        
        # Zoom range based on intensity
        zoom_range = intensity * 0.5  # Max 50% zoom
        zoom_start = 1.0
        zoom_end = 1.0 + random.uniform(0, zoom_range)
        
        # Pan range based on intensity
        pan_range = intensity * 0.2  # Max 20% pan
        pan_x_start = random.uniform(-pan_range/2, pan_range/2)
        pan_x_end = random.uniform(-pan_range/2, pan_range/2)
        pan_y_start = random.uniform(-pan_range/4, pan_range/4)  # Less vertical movement
        pan_y_end = random.uniform(-pan_range/4, pan_range/4)
        
        return {
            'zoom_start': zoom_start,
            'zoom_end': zoom_end,
            'pan_x_start': pan_x_start,
            'pan_x_end': pan_x_end,
            'pan_y_start': pan_y_start,
            'pan_y_end': pan_y_end
        }
    
    def _create_movement_patterns(self) -> Dict[str, Dict[str, float]]:
        """Create predefined movement patterns."""
        return {
            'zoom_in_left': {
                'zoom_start': 1.0, 'zoom_end': 1.3,
                'pan_x_start': 0.1, 'pan_x_end': -0.1,
                'pan_y_start': 0.0, 'pan_y_end': 0.0
            },
            'zoom_out_right': {
                'zoom_start': 1.2, 'zoom_end': 1.0,
                'pan_x_start': -0.1, 'pan_x_end': 0.1,
                'pan_y_start': 0.0, 'pan_y_end': 0.0
            },
            'subtle_drift': {
                'zoom_start': 1.0, 'zoom_end': 1.1,
                'pan_x_start': 0.0, 'pan_x_end': 0.05,
                'pan_y_start': 0.0, 'pan_y_end': 0.02
            }
        }


class TransitionEffect:
    """
    Handles transitions between different visual elements or scenes.
    """
    
    def __init__(self):
        """Initialize transition effect handler."""
        pass
    
    def create_crossfade(self, duration: float = 1.0) -> Dict[str, Any]:
        """Create crossfade transition between images."""
        return {
            'type': 'crossfade',
            'duration': duration,
            'properties': {
                'easing': 'ease_in_out_quart'
            }
        }
    
    def create_slide_transition(
        self, 
        direction: str = 'left',
        duration: float = 0.8
    ) -> Dict[str, Any]:
        """Create slide transition between images."""
        
        direction_map = {
            'left': {'from': {'translateX': '100%'}, 'to': {'translateX': '0%'}},
            'right': {'from': {'translateX': '-100%'}, 'to': {'translateX': '0%'}},
            'up': {'from': {'translateY': '100%'}, 'to': {'translateY': '0%'}},
            'down': {'from': {'translateY': '-100%'}, 'to': {'translateY': '0%'}}
        }
        
        transform = direction_map.get(direction, direction_map['left'])
        
        return {
            'type': 'slide_transition',
            'duration': duration,
            'properties': {
                'easing': 'ease_out_quart',
                **transform
            }
        }
    
    def create_wipe_transition(
        self, 
        direction: str = 'horizontal',
        duration: float = 0.6
    ) -> Dict[str, Any]:
        """Create wipe transition effect."""
        
        return {
            'type': 'wipe_transition',
            'duration': duration,
            'properties': {
                'direction': direction,
                'easing': 'ease_in_out_quart'
            }
        }


class ParticleEffect:
    """
    Creates particle effects for enhanced visual impact.
    """
    
    def __init__(self):
        """Initialize particle effect handler."""
        pass
    
    def create_sparkle_effect(
        self, 
        particle_count: int = 20,
        duration: float = 2.0,
        intensity: EffectIntensity = EffectIntensity.MEDIUM
    ) -> Dict[str, Any]:
        """Create sparkle particle effect."""
        
        intensity_multipliers = {
            EffectIntensity.SUBTLE: 0.5,
            EffectIntensity.MEDIUM: 1.0,
            EffectIntensity.HIGH: 1.5,
            EffectIntensity.VIRAL: 2.0
        }
        
        multiplier = intensity_multipliers[intensity]
        
        return {
            'type': 'sparkle_particles',
            'duration': duration,
            'properties': {
                'particle_count': int(particle_count * multiplier),
                'size_range': [2, 8],
                'colors': ['#FFD700', '#FFA500', '#FF69B4', '#00FFFF'],
                'animation': 'twinkle',
                'spawn_area': 'around_image'
            }
        }
    
    def create_explosion_effect(
        self, 
        duration: float = 1.0,
        intensity: EffectIntensity = EffectIntensity.HIGH
    ) -> Dict[str, Any]:
        """Create explosion particle effect."""
        
        return {
            'type': 'explosion_particles',
            'duration': duration,
            'properties': {
                'particle_count': 50 if intensity == EffectIntensity.VIRAL else 30,
                'velocity_range': [50, 200],
                'colors': ['#FF4500', '#FF6347', '#FFD700'],
                'animation': 'burst',
                'gravity': True
            }
        }
    
    def create_confetti_effect(
        self, 
        duration: float = 3.0,
        colors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create confetti particle effect."""
        
        if not colors:
            colors = ['#FF69B4', '#00FFFF', '#FFD700', '#32CD32', '#FF4500']
        
        return {
            'type': 'confetti_particles',
            'duration': duration,
            'properties': {
                'particle_count': 100,
                'shapes': ['circle', 'square', 'triangle'],
                'colors': colors,
                'animation': 'falling',
                'gravity': True,
                'wind': True
            }
        } 