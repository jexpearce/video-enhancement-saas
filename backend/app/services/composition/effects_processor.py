"""
EffectsProcessor - Advanced video effects and transitions.

Handles complex visual effects that complement the animation timeline:
- Particle effects
- Color grading
- Motion blur
- Glitch effects  
- Beat-synchronized effects
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import math

logger = logging.getLogger(__name__)

@dataclass
class EffectConfig:
    """Configuration for a video effect."""
    
    effect_type: str  # particle, glitch, blur, glow, etc.
    intensity: float = 1.0  # 0.0 to 1.0
    duration: float = 1.0
    start_time: float = 0.0
    
    # Effect-specific parameters
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass 
class ParticleEffect:
    """Configuration for particle effects."""
    
    particle_count: int = 50
    particle_size: int = 5
    velocity: Tuple[int, int] = (10, 10)
    gravity: float = 0.5
    lifetime: float = 2.0
    color: str = "#FFFFFF"
    blend_mode: str = "add"  # add, multiply, overlay
    emission_shape: str = "point"  # point, line, circle
    
class EffectsProcessor:
    """
    Handles advanced video effects and transitions.
    
    Integrates with the animation timeline to add sophisticated 
    visual effects that enhance emphasis points and story moments.
    """
    
    def __init__(self):
        """Initialize the effects processor."""
        
        self.supported_effects = {
            'particle_burst', 'glitch_pop', 'motion_blur', 
            'glow_pulse', 'color_shift', 'zoom_burst',
            'shake_effect', 'chromatic_aberration', 'vignette',
            'film_grain', 'light_rays', 'energy_wave'
        }
        
        # Effect presets for different styles
        self.effect_presets = {
            'tiktok_viral': {
                'particle_burst': ParticleEffect(
                    particle_count=100,
                    particle_size=8,
                    velocity=(20, 15),
                    color="#FF6B6B",
                    blend_mode="add"
                ),
                'glitch_pop': {
                    'intensity': 0.8,
                    'frequency': 15,
                    'color_shift': True
                }
            },
            
            'instagram_aesthetic': {
                'glow_pulse': {
                    'intensity': 0.6,
                    'color': "#FFF2E6",
                    'spread': 10
                },
                'vignette': {
                    'intensity': 0.3,
                    'softness': 0.7
                }
            },
            
            'youtube_epic': {
                'light_rays': {
                    'ray_count': 8,
                    'intensity': 0.9,
                    'color': "#FFD700"
                },
                'energy_wave': {
                    'amplitude': 20,
                    'frequency': 2,
                    'speed': 1.5
                }
            }
        }
        
        logger.info("EffectsProcessor initialized")
    
    def generate_effects_for_timeline(
        self,
        emphasis_points: List[Dict],
        animation_timeline: Dict,
        style: Dict,
        video_duration: float
    ) -> List[EffectConfig]:
        """
        Generate effects based on emphasis points and animation timeline.
        
        Args:
            emphasis_points: Emphasis detection results
            animation_timeline: Animation timeline events
            style: Selected style template
            video_duration: Total video duration
            
        Returns:
            List of effect configurations
        """
        
        effects = []
        
        try:
            style_name = style.get('template_name', 'default')
            platform = style.get('platform', 'tiktok')
            
            # Process high-emphasis points for special effects
            for point in emphasis_points:
                emphasis_score = point.get('emphasis_score', 0)
                confidence = point.get('confidence', 0)
                
                # Only add effects for high-confidence, high-emphasis points
                if emphasis_score > 0.7 and confidence > 0.8:
                    effect = self._create_emphasis_effect(point, style_name, platform)
                    if effect:
                        effects.append(effect)
            
            # Add beat-synchronized effects if audio beats available
            timeline_events = animation_timeline.get('events', [])
            beat_effects = self._create_beat_effects(timeline_events, style_name)
            effects.extend(beat_effects)
            
            # Add ambient/background effects
            ambient_effects = self._create_ambient_effects(style_name, video_duration)
            effects.extend(ambient_effects)
            
            logger.info(f"Generated {len(effects)} effects for timeline")
            return effects
            
        except Exception as e:
            logger.error(f"Error generating effects: {e}")
            return []
    
    def _create_emphasis_effect(
        self,
        emphasis_point: Dict,
        style_name: str,
        platform: str
    ) -> Optional[EffectConfig]:
        """Create effect for high-emphasis word/moment."""
        
        start_time = emphasis_point.get('start_time', 0)
        emphasis_score = emphasis_point.get('emphasis_score', 0)
        word = emphasis_point.get('word', '')
        
        # Choose effect based on word characteristics and emphasis level
        if emphasis_score > 0.9:
            # Super high emphasis - dramatic effect
            effect_type = 'zoom_burst' if 'action' in word.lower() else 'particle_burst'
            intensity = 1.0
            duration = 0.8
            
        elif emphasis_score > 0.8:
            # High emphasis - noticeable effect
            effect_type = 'glitch_pop' if platform == 'tiktok' else 'glow_pulse'
            intensity = 0.8
            duration = 0.6
            
        else:
            # Medium emphasis - subtle effect
            effect_type = 'glow_pulse'
            intensity = 0.5
            duration = 0.4
        
        # Get preset parameters
        preset_key = f"{platform}_viral" if platform == 'tiktok' else f"{platform}_aesthetic"
        preset_params = self.effect_presets.get(preset_key, {}).get(effect_type, {})
        
        return EffectConfig(
            effect_type=effect_type,
            intensity=intensity,
            duration=duration,
            start_time=start_time,
            parameters={
                **preset_params,
                'emphasis_word': word,
                'emphasis_score': emphasis_score
            }
        )
    
    def _create_beat_effects(
        self,
        timeline_events: List[Dict],
        style_name: str
    ) -> List[EffectConfig]:
        """Create effects synchronized with audio beats."""
        
        effects = []
        
        # Look for beat synchronization events in timeline
        for event in timeline_events:
            if event.get('type') == 'beat_sync':
                beat_time = event.get('start_time', 0)
                beat_strength = event.get('properties', {}).get('strength', 0.5)
                
                # Create subtle pulse effect on beats
                if beat_strength > 0.3:
                    effect = EffectConfig(
                        effect_type='glow_pulse',
                        intensity=beat_strength * 0.4,
                        duration=0.2,
                        start_time=beat_time,
                        parameters={
                            'color': '#FFFFFF',
                            'spread': 5,
                            'is_beat_sync': True
                        }
                    )
                    effects.append(effect)
        
        return effects
    
    def _create_ambient_effects(
        self,
        style_name: str,
        video_duration: float
    ) -> List[EffectConfig]:
        """Create subtle ambient effects throughout the video."""
        
        effects = []
        
        # Add subtle film grain for texture
        if 'vintage' in style_name.lower() or 'cinematic' in style_name.lower():
            grain_effect = EffectConfig(
                effect_type='film_grain',
                intensity=0.2,
                duration=video_duration,
                start_time=0,
                parameters={
                    'grain_size': 1,
                    'noise_level': 0.15
                }
            )
            effects.append(grain_effect)
        
        # Add subtle vignette for focus
        if 'dramatic' in style_name.lower():
            vignette_effect = EffectConfig(
                effect_type='vignette',
                intensity=0.3,
                duration=video_duration,
                start_time=0,
                parameters={
                    'softness': 0.8,
                    'color': '#000000'
                }
            )
            effects.append(vignette_effect)
        
        return effects
    
    def generate_ffmpeg_effects_filter(
        self,
        effects: List[EffectConfig],
        video_resolution: Tuple[int, int] = (1080, 1920)
    ) -> str:
        """
        Generate FFmpeg filter string for all effects.
        
        Args:
            effects: List of effect configurations
            video_resolution: Video width and height
            
        Returns:
            FFmpeg filter string for complex effects
        """
        
        if not effects:
            return ""
        
        filter_parts = []
        video_width, video_height = video_resolution
        
        for i, effect in enumerate(effects):
            filter_str = self._generate_single_effect_filter(
                effect, video_width, video_height, i
            )
            if filter_str:
                filter_parts.append(filter_str)
        
        return ";".join(filter_parts) if filter_parts else ""
    
    def _generate_single_effect_filter(
        self,
        effect: EffectConfig,
        video_width: int,
        video_height: int,
        effect_index: int
    ) -> str:
        """Generate FFmpeg filter for a single effect."""
        
        effect_type = effect.effect_type
        intensity = effect.intensity
        start_time = effect.start_time
        duration = effect.duration
        params = effect.parameters
        
        # Enable filter only during effect duration
        enable_expr = f"between(t,{start_time},{start_time + duration})"
        
        if effect_type == 'glow_pulse':
            # Glow effect using box blur and blend
            blur_amount = int(intensity * 10)
            opacity = intensity * 0.6
            return f"[0:v]split[main][glow];[glow]boxblur={blur_amount}:1,format=yuv420p[glowblur];[main][glowblur]blend=all_mode=addition:all_opacity={opacity}:enable='{enable_expr}'[v{effect_index}]"
        
        elif effect_type == 'glitch_pop':
            # Digital glitch effect with datamosh
            frequency = params.get('frequency', 10)
            return f"[0:v]noise=alls={int(intensity * 20)}:allf=t+u:enable='{enable_expr}',datascope=size={video_width}x{video_height}:enable='{enable_expr}'[v{effect_index}]"
        
        elif effect_type == 'zoom_burst':
            # Zoom burst effect
            zoom_factor = 1 + (intensity * 0.3)
            return f"[0:v]zoompan=z='if(between(t,{start_time},{start_time + duration}),{zoom_factor},1)':x='iw/2':y='ih/2':d=1:enable='{enable_expr}'[v{effect_index}]"
        
        elif effect_type == 'motion_blur':
            # Motion blur effect
            blur_amount = int(intensity * 5)
            return f"[0:v]minterpolate=fps=60:mb_size=16:search_param=400,boxblur={blur_amount}:1:enable='{enable_expr}'[v{effect_index}]"
        
        elif effect_type == 'chromatic_aberration':
            # RGB channel separation
            offset = int(intensity * 5)
            return f"[0:v]split=3[r][g][b];[r]lutrgb=g=0:b=0[red];[g]lutrgb=r=0:b=0[green];[b]lutrgb=r=0:g=0[blue];[red]pad={video_width + offset}:{video_height}[redpad];[green]overlay={offset//2}:0[rg];[rg][blue]overlay={offset}:0:enable='{enable_expr}'[v{effect_index}]"
        
        elif effect_type == 'vignette':
            # Vignette effect
            softness = params.get('softness', 0.7)
            return f"[0:v]vignette=angle=PI/3:x0={video_width//2}:y0={video_height//2}:mode=backward:eval=frame:dither=1:aspect=1:enable='{enable_expr}'[v{effect_index}]"
        
        elif effect_type == 'film_grain':
            # Film grain texture
            noise_level = params.get('noise_level', 0.1)
            return f"[0:v]noise=alls={int(noise_level * 100)}:allf=t:enable='{enable_expr}'[v{effect_index}]"
        
        else:
            # Unknown effect, skip
            logger.warning(f"Unknown effect type: {effect_type}")
            return ""
    
    def get_supported_effects(self) -> List[str]:
        """Get list of supported effect types."""
        return list(self.supported_effects)
    
    def get_effect_presets(self, platform: str = None) -> Dict[str, Any]:
        """Get effect presets, optionally filtered by platform."""
        
        if platform:
            return {
                k: v for k, v in self.effect_presets.items() 
                if platform.lower() in k.lower()
            }
        
        return self.effect_presets
    
    def validate_effect_config(self, effect: EffectConfig) -> bool:
        """Validate an effect configuration."""
        
        try:
            # Check required fields
            if not effect.effect_type or effect.effect_type not in self.supported_effects:
                return False
            
            # Check intensity range
            if not 0.0 <= effect.intensity <= 1.0:
                return False
            
            # Check duration is positive
            if effect.duration <= 0:
                return False
            
            # Check start time is non-negative
            if effect.start_time < 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Effect validation error: {e}")
            return False 