"""
Sophisticated Animation Engine for TikTok/Instagram Videos

Creates engaging, synchronized animations that enhance talking-head videos with:
- Image entrance/exit animations synchronized with emphasis points
- Ken Burns effects for dynamic movement
- Text overlay animations with beat matching
- Particle effects and transitions
- Intelligent timing to avoid visual overload
"""

import logging
import math
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio

import numpy as np

from .timeline import AnimationTimeline, AnimationEvent
from .easing import EasingFunction, EasingType
from .effects import ImageAnimation, TransitionEffect, KenBurnsEffect
from .synchronizer import EmphasisSynchronizer, BeatSynchronizer
from ..images.styles.models import StyleTemplate
from ...database.models import EmphasisPoint
from ..images.ranking.models import RankedImage

logger = logging.getLogger(__name__)

@dataclass
class AnimationConfig:
    """Configuration for animation engine."""
    
    # Timing settings
    min_gap_between_animations: float = 1.5  # Minimum seconds between animations
    max_concurrent_animations: int = 2       # Max overlapping animations
    emphasis_animation_duration: float = 0.8 # Standard duration for emphasis animations
    
    # Visual settings
    image_display_duration_range: Tuple[float, float] = (2.0, 4.0)  # Min, max display time
    ken_burns_intensity: float = 0.3         # How much to zoom/pan (0-1)
    particle_density: float = 0.5           # Particle effect density (0-1)
    
    # Synchronization settings
    beat_sync_tolerance: float = 0.2         # How close to beat to snap (seconds)
    emphasis_sync_priority: bool = True      # Prioritize emphasis over beat sync
    
    # Performance settings
    max_timeline_events: int = 50           # Limit events to prevent overload
    enable_gpu_acceleration: bool = True    # Use GPU for effects when available

@dataclass(eq=False)
class GroupedEmphasis:
    """Group of nearby emphasis points treated as one animation unit."""
    
    start_time: float
    end_time: float
    emphasis_points: List[Dict]  # Will be EmphasisPoint when imported
    primary_entity: str                      # Most important entity in group
    energy_level: float                      # Combined energy of group
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def center_time(self) -> float:
        return (self.start_time + self.end_time) / 2
        
    def __hash__(self):
        """Make GroupedEmphasis hashable for use as dictionary keys."""
        return hash((self.start_time, self.end_time, self.primary_entity, self.energy_level))
        
    def __eq__(self, other):
        """Define equality for hashing."""
        if not isinstance(other, GroupedEmphasis):
            return False
        return (self.start_time == other.start_time and 
                self.end_time == other.end_time and 
                self.primary_entity == other.primary_entity and 
                self.energy_level == other.energy_level)

class AnimationEngine:
    """
    Sophisticated animation engine for video enhancement.
    
    Features:
    - Intelligent grouping of emphasis points to avoid visual overload
    - Multiple animation types (fade, slide, zoom, bounce, particles)
    - Ken Burns effects for dynamic image movement
    - Beat synchronization with audio
    - Easing functions for smooth motion
    - Performance optimization with GPU acceleration
    """
    
    def __init__(self, config: Optional[AnimationConfig] = None):
        """Initialize animation engine."""
        self.config = config or AnimationConfig()
        
        # Animation state
        self.timeline_cache = {}
        self.performance_stats = {
            'total_timelines_created': 0,
            'avg_processing_time_ms': 0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("AnimationEngine initialized with sophisticated effects")
    
    async def create_image_animation_timeline(
        self,
        emphasis_points: List[Dict],
        ranked_images: List[Dict],
        style: Dict,
        video_duration: float,
        audio_beats: Optional[List[float]] = None
    ) -> Dict:
        """
        Create sophisticated animation timeline for video enhancement.
        
        Args:
            emphasis_points: Detected emphasis points with timing
            ranked_images: Images ranked by relevance
            style: Selected style template
            video_duration: Total video duration in seconds
            audio_beats: Optional beat timestamps for synchronization
            
        Returns:
            AnimationTimeline with all animation events
        """
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Creating animation timeline for {len(emphasis_points)} emphasis points")
            
            # Check cache first
            cache_key = self._generate_cache_key(emphasis_points, style, video_duration)
            if cache_key in self.timeline_cache:
                logger.debug("Using cached timeline")
                return self.timeline_cache[cache_key]
            
            # 1. Group nearby emphasis points to avoid visual overload
            grouped_points = self._group_emphasis_points(emphasis_points)
            logger.debug(f"Grouped {len(emphasis_points)} points into {len(grouped_points)} groups")
            
            # 2. Assign images to groups based on relevance and timing
            try:
                image_assignments = self._assign_images_to_groups(grouped_points, ranked_images)
                logger.debug(f"✅ Image assignment successful: {len(image_assignments)} groups")
            except Exception as assignment_error:
                logger.error(f"❌ Image assignment failed: {assignment_error}")
                raise assignment_error
            
            # 3. Create animation timeline
            timeline = {
                'duration': video_duration,
                'events': []
            }
            
            # 4. Create animation events for each group
            for group, images in image_assignments.items():
                if not images:
                    continue
                    
                # Determine display duration based on group characteristics
                display_duration = self._calculate_display_duration(group, style)
                
                # Create animations based on number of images
                if len(images) == 1:
                    events = await self._create_single_image_animation(
                        images[0], group, display_duration, style
                    )
                else:
                    events = await self._create_multi_image_animation(
                        images, group, display_duration, style
                    )
                
                timeline['events'].extend(events)
            
            # 5. Add text/caption animations synchronized with images
            if style.get('has_text_overlays', False):
                caption_events = await self._create_caption_animations(
                    emphasis_points, timeline, style
                )
                timeline['events'].extend(caption_events)
            
            # 6. Add beat synchronization if audio beats provided
            if audio_beats and style.get('has_pulse_to_beat', False):
                beat_events = self._create_beat_sync_events(
                    timeline, audio_beats, style
                )
                timeline['events'].extend(beat_events)
            
            # 7. Optimize timeline to prevent conflicts and improve performance
            timeline = self._optimize_timeline(timeline)
            
            # 8. Cache result
            self.timeline_cache[cache_key] = timeline
            
            # 9. Update performance stats
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(processing_time)
            
            logger.info(f"Created timeline with {len(timeline['events'])} events in {processing_time:.1f}ms")
            
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to create animation timeline: {e}")
            # Return basic timeline as fallback
            return self._create_fallback_timeline(emphasis_points, ranked_images, video_duration)
    
    def _group_emphasis_points(self, emphasis_points: List[Dict]) -> List[GroupedEmphasis]:
        """Group nearby emphasis points to avoid visual overload."""
        
        if not emphasis_points:
            return []
        
        # Sort by start time
        sorted_points = sorted(emphasis_points, key=lambda p: p.get('start_time', 0))
        
        groups = []
        current_group = [sorted_points[0]]
        current_start = sorted_points[0].get('start_time', 0)
        current_end = sorted_points[0].get('end_time', current_start + 1)
        
        for point in sorted_points[1:]:
            # Check if this point is close enough to merge with current group
            gap = point.get('start_time', 0) - current_end
            
            if gap <= self.config.min_gap_between_animations:
                # Merge into current group
                current_group.append(point)
                current_end = max(current_end, point.get('end_time', point.get('start_time', 0) + 1))
            else:
                # Create new group
                groups.append(self._create_grouped_emphasis(current_group, current_start, current_end))
                
                # Start new group
                current_group = [point]
                current_start = point.get('start_time', 0)
                current_end = point.get('end_time', current_start + 1)
        
        # Add final group
        if current_group:
            groups.append(self._create_grouped_emphasis(current_group, current_start, current_end))
        
        return groups
    
    def _create_grouped_emphasis(
        self, 
        points: List[Dict], 
        start_time: float, 
        end_time: float
    ) -> GroupedEmphasis:
        """Create a grouped emphasis from individual points."""
        
        # Find primary entity (most emphasized)
        primary_entity = max(points, key=lambda p: p.get('emphasis_score', 0)).get('word_text', 'entity')
        
        # Calculate combined energy level
        total_energy = sum(p.get('emphasis_score', 0.5) for p in points)
        avg_energy = total_energy / len(points) if points else 0.5
        
        return GroupedEmphasis(
            start_time=start_time,
            end_time=end_time,
            emphasis_points=points,
            primary_entity=primary_entity,
            energy_level=avg_energy
        )
    
    def _assign_images_to_groups(
        self, 
        grouped_points: List[GroupedEmphasis], 
        ranked_images: List[Dict]
    ) -> Dict[GroupedEmphasis, List[Dict]]:
        """Assign best images to each emphasis group."""
        
        assignments = {}
        used_images = set()
        
        logger.debug(f"🔧 Assigning {len(ranked_images)} images to {len(grouped_points)} groups")
        
        for i, group in enumerate(grouped_points):
            logger.debug(f"🔧 Group {i}: {group.primary_entity} ({group.start_time:.1f}s-{group.end_time:.1f}s)")
            
            # Find images matching the primary entity
            matching_images = [
                img for img in ranked_images
                if img.get('entity_id') == group.primary_entity and 
                img.get('image_id') not in used_images
            ]
            
            # If no exact matches, use top-ranked available images
            if not matching_images:
                matching_images = [
                    img for img in ranked_images
                    if img.get('image_id') not in used_images
                ]
            
            # Select images based on group energy and duration
            num_images = self._determine_image_count(group)
            selected_images = matching_images[:num_images]
            
            logger.debug(f"🔧 Selected {len(selected_images)} images for group {i}")
            
            # Mark images as used
            for img in selected_images:
                used_images.add(img.get('image_id'))
            
            # FIXED: This should now work with hashable GroupedEmphasis
            assignments[group] = selected_images
        
        logger.debug(f"🔧 Image assignment completed: {len(assignments)} groups with images")
        return assignments
    
    def _determine_image_count(self, group: GroupedEmphasis) -> int:
        """Determine how many images to show for a group."""
        
        # Base count on group duration and energy
        base_count = 1
        
        if group.duration > 3.0:  # Long emphasis
            base_count = 2
        
        if group.energy_level > 0.8:  # High energy
            base_count += 1
        
        if len(group.emphasis_points) > 2:  # Multiple points
            base_count += 1
        
        return min(base_count, 3)  # Max 3 images per group
    
    def _calculate_display_duration(self, group: GroupedEmphasis, style: Dict) -> float:
        """Calculate optimal display duration for image group."""
        
        min_duration, max_duration = self.config.image_display_duration_range
        
        # Base duration on group duration
        base_duration = max(min_duration, group.duration + 0.5)
        
        # Adjust based on style
        animation_intensity = style.get('animation_intensity', 'medium')
        if animation_intensity == "high":
            base_duration *= 0.8  # Faster for high intensity
        elif animation_intensity == "low":
            base_duration *= 1.2  # Slower for low intensity
        
        # Adjust based on energy level
        energy_factor = 1.0 + (group.energy_level - 0.5) * 0.4
        base_duration *= energy_factor
        
        return min(max_duration, base_duration)
    
    async def _create_single_image_animation(
        self,
        image: Dict,
        group: GroupedEmphasis,
        duration: float,
        style: Dict
    ) -> List[Dict]:
        """Create animation events for a single image."""
        
        events = []
        transition_duration = 0.5  # Fixed transition time
        image_id = image.get('image_id', 'default')
        
        # 1. Entry animation
        entry_event = {
            'type': 'image_entry',
            'target_id': image_id,
            'start_time': group.start_time - 0.2,  # Start slightly before emphasis
            'duration': transition_duration,
            'properties': self._get_entry_animation_properties(style, image)
        }
        events.append(entry_event)
        
        # 2. Ken Burns effect (if enabled and appropriate)
        if style.get('has_ken_burns', False) and duration > 2.0:
            kb_event = self._create_ken_burns_effect(
                image, group.start_time, duration, style
            )
            events.append(kb_event)
        
        # 3. Pulse effect synchronized with emphasis
        if style.get('has_pulse_to_beat', False):
            for point in group.emphasis_points:
                pulse_event = {
                    'type': 'pulse_effect',
                    'target_id': image_id,
                    'start_time': point.get('start_time', 0),
                    'duration': 0.3,
                    'properties': {
                        'intensity': point.get('emphasis_score', 0.5),
                        'easing': 'ease_out_back'
                    }
                }
                events.append(pulse_event)
        
        # 4. Exit animation
        exit_event = {
            'type': 'image_exit',
            'target_id': image_id,
            'start_time': group.start_time + duration - transition_duration,
            'duration': transition_duration,
            'properties': self._get_exit_animation_properties(style, image)
        }
        events.append(exit_event)
        
        return events
    
    async def _create_multi_image_animation(
        self,
        images: List[Dict],
        group: GroupedEmphasis,
        duration: float,
        style: Dict
    ) -> List[Dict]:
        """Create staggered animations for multiple images."""
        
        events = []
        
        # Calculate stagger timing
        overlap_duration = duration * 0.6  # 60% overlap
        stagger_delay = (duration - overlap_duration) / (len(images) - 1) if len(images) > 1 else 0
        
        for i, image in enumerate(images):
            # Stagger start times
            image_start = group.start_time - 0.2 + (i * stagger_delay)
            image_duration = overlap_duration
            
            # Create individual animation events
            image_group = GroupedEmphasis(
                start_time=image_start + 0.2,  # Adjust for the -0.2 offset
                end_time=image_start + image_duration,
                emphasis_points=group.emphasis_points[:1] if group.emphasis_points else [],
                primary_entity=group.primary_entity,
                energy_level=group.energy_level
            )
            
            image_events = await self._create_single_image_animation(
                image, image_group, image_duration, style
            )
            
            # Adjust positions for multi-image layout
            for event in image_events:
                if 'position' in event['properties']:
                    event['properties']['position'] = self._get_multi_image_position(i, len(images), style)
                if 'scale' in event['properties'].get('to', {}):
                    event['properties']['to']['scale'] *= 0.8  # Smaller for multiple images
            
            events.extend(image_events)
        
        return events
    
    def _get_entry_animation_properties(self, style: Dict, image: Dict) -> Dict[str, Any]:
        """Get properties for image entry animation."""
        
        animation_type = style.get('animation_type', 'fade')
        
        base_properties = {
            'position': style.get('position', 'top_right'),
            'size': 'medium',
            'easing': 'ease_out_quart'
        }
        
        if animation_type == "fade":
            base_properties.update({
                'from': {'opacity': 0, 'scale': 1},
                'to': {'opacity': 1, 'scale': 1}
            })
        elif animation_type == "slide":
            base_properties.update({
                'from': {'opacity': 0, 'translateX': 100},
                'to': {'opacity': 1, 'translateX': 0}
            })
        elif animation_type == "zoom":
            base_properties.update({
                'from': {'opacity': 0, 'scale': 0.5},
                'to': {'opacity': 1, 'scale': 1}
            })
        elif animation_type == "bounce":
            base_properties.update({
                'from': {'opacity': 0, 'translateY': -50, 'scale': 0.8},
                'to': {'opacity': 1, 'translateY': 0, 'scale': 1},
                'easing': 'ease_out_bounce'
            })
        
        return base_properties
    
    def _get_exit_animation_properties(self, style: Dict, image: Dict) -> Dict[str, Any]:
        """Get properties for image exit animation."""
        
        # Usually reverse of entry animation
        entry_props = self._get_entry_animation_properties(style, image)
        
        return {
            'position': entry_props['position'],
            'size': entry_props['size'],
            'easing': 'ease_in_quart',
            'from': entry_props['to'],
            'to': entry_props['from']
        }
    
    def _create_ken_burns_effect(
        self,
        image: Dict,
        start_time: float,
        duration: float,
        style: Dict
    ) -> Dict:
        """Create Ken Burns pan and zoom effect."""
        
        # Random but controlled movement
        zoom_start = 1.0
        zoom_end = 1.0 + (self.config.ken_burns_intensity * 0.5)
        
        # Pan direction based on image composition
        pan_x = np.random.uniform(-self.config.ken_burns_intensity, self.config.ken_burns_intensity)
        pan_y = np.random.uniform(-self.config.ken_burns_intensity * 0.5, self.config.ken_burns_intensity * 0.5)
        
        return {
            'type': 'ken_burns',
            'target_id': image.get('image_id', 'default'),
            'start_time': start_time,
            'duration': duration,
            'properties': {
                'zoom_start': zoom_start,
                'zoom_end': zoom_end,
                'pan_x': pan_x,
                'pan_y': pan_y,
                'easing': 'linear'
            }
        }
    
    def _get_multi_image_position(self, index: int, total: int, style: Dict) -> str:
        """Get position for multi-image layouts."""
        
        if total == 2:
            return "top_right" if index == 0 else "bottom_right"
        elif total == 3:
            positions = ["top_right", "center_right", "bottom_right"]
            return positions[index % 3]
        else:
            # Grid layout for 4+
            return f"grid_position_{index + 1}"
    
    async def _create_caption_animations(
        self,
        emphasis_points: List[Dict],
        timeline: Dict,
        style: Dict
    ) -> List[Dict]:
        """Create text caption animations synchronized with images."""
        
        caption_events = []
        
        for point in emphasis_points:
            # Check if there's an image animation at this time
            point_start = point.get('start_time', 0)
            overlapping_images = [
                e for e in timeline['events']
                if e['type'] == 'image_entry' and abs(e['start_time'] - point_start) < 1.0
            ]
            
            if overlapping_images and style.get('has_text_overlays', False):
                caption_event = {
                    'type': 'text_caption',
                    'target_id': f"caption_{point.get('word_text', 'text')}",
                    'start_time': point_start + 0.1,  # Slight delay after image
                    'duration': 1.5,
                    'properties': {
                        'text': point.get('word_text', '').upper(),
                        'style': style.get('typography_style', 'bold'),
                        'animation': 'fade_up',
                        'position': 'bottom_center',
                        'emphasis_score': point.get('emphasis_score', 0.5)
                    }
                }
                caption_events.append(caption_event)
        
        return caption_events
    
    def _create_beat_sync_events(
        self,
        timeline: Dict,
        audio_beats: List[float],
        style: Dict
    ) -> List[Dict]:
        """Create beat synchronization events."""
        
        beat_events = []
        
        for beat_time in audio_beats:
            # Find closest image animation
            closest_event = None
            min_distance = float('inf')
            
            for event in timeline['events']:
                if event['type'] in ['image_entry', 'image_exit']:
                    distance = abs(event['start_time'] - beat_time)
                    if distance < min_distance and distance < self.config.beat_sync_tolerance:
                        min_distance = distance
                        closest_event = event
            
            if closest_event:
                beat_event = {
                    'type': 'beat_sync',
                    'target_id': closest_event['target_id'],
                    'start_time': beat_time,
                    'duration': 0.1,
                    'properties': {
                        'intensity': 0.3,
                        'effect': 'scale_pulse'
                    }
                }
                beat_events.append(beat_event)
        
        return beat_events
    
    def _optimize_timeline(self, timeline: Dict) -> Dict:
        """Optimize timeline to prevent conflicts and improve performance."""
        
        events = timeline['events']
        
        # 1. Remove overlapping events that might cause visual conflicts
        events = self._resolve_conflicts(events)
        
        # 2. Limit total number of events to prevent performance issues
        if len(events) > self.config.max_timeline_events:
            # Keep highest priority events (image events first)
            priority_order = {'image_entry': 3, 'image_exit': 3, 'ken_burns': 2, 'pulse_effect': 1, 'text_caption': 1}
            events = sorted(
                events, 
                key=lambda e: priority_order.get(e['type'], 0), 
                reverse=True
            )[:self.config.max_timeline_events]
        
        # 3. Sort by start time
        events.sort(key=lambda e: e['start_time'])
        
        timeline['events'] = events
        return timeline
    
    def _resolve_conflicts(self, events: List[Dict]) -> List[Dict]:
        """Remove conflicting events."""
        
        # For now, just return all events
        # In production, this would detect and resolve conflicts
        return events
    
    def _generate_cache_key(
        self, 
        emphasis_points: List[Dict], 
        style: Dict, 
        video_duration: float
    ) -> str:
        """Generate cache key for timeline."""
        
        # Create a hash based on key parameters
        key_data = f"{len(emphasis_points)}_{style.get('template_id', 'default')}_{video_duration}"
        
        # Add emphasis timing signature
        if emphasis_points:
            timing_signature = "_".join([f"{p.get('start_time', 0):.1f}" for p in emphasis_points[:5]])
            key_data += f"_{timing_signature}"
        
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _create_fallback_timeline(
        self,
        emphasis_points: List[Dict],
        ranked_images: List[Dict],
        video_duration: float
    ) -> Dict:
        """Create basic timeline as fallback when main creation fails."""
        
        timeline = {
            'duration': video_duration,
            'events': []
        }
        
        logger.warning("Creating fallback timeline due to animation engine error")
        
        # Simple fade animations for top images
        for i, point in enumerate(emphasis_points[:5]):  # Limit to 5
            if i < len(ranked_images):
                image = ranked_images[i]
                point_start = point.get('start_time', i * 2)
                
                # FIXED: Use actual image_id instead of generic fallback
                image_id = image.get('image_id') or image.get('id') or f'fallback_image_{i}'
                
                logger.info(f"🔧 Fallback: Using image_id '{image_id}' for emphasis point at {point_start}s")
                
                # Simple fade in/out
                fade_in = {
                    'type': 'image_entry',
                    'target_id': image_id,
                    'start_time': point_start,
                    'duration': 0.5,
                    'properties': {'animation': 'fade', 'position': 'top_right'}
                }
                
                fade_out = {
                    'type': 'image_exit',
                    'target_id': image_id,
                    'start_time': point_start + 2.5,
                    'duration': 0.5,
                    'properties': {'animation': 'fade', 'position': 'top_right'}
                }
                
                timeline['events'].extend([fade_in, fade_out])
        
        logger.info(f"🔧 Fallback timeline created with {len(timeline['events'])} events")
        return timeline
    
    def _update_performance_stats(self, processing_time_ms: float):
        """Update performance statistics."""
        
        self.performance_stats['total_timelines_created'] += 1
        
        # Update average processing time
        current_avg = self.performance_stats['avg_processing_time_ms']
        count = self.performance_stats['total_timelines_created']
        
        new_avg = ((current_avg * (count - 1)) + processing_time_ms) / count
        self.performance_stats['avg_processing_time_ms'] = new_avg
        
        # Update cache hit rate
        total_requests = count
        cache_requests = len(self.timeline_cache)
        self.performance_stats['cache_hit_rate'] = cache_requests / total_requests if total_requests > 0 else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def clear_cache(self):
        """Clear timeline cache."""
        self.timeline_cache.clear()
        logger.info("Animation timeline cache cleared")

    def generate_css_keyframes(self, timeline: Dict) -> str:
        """Generate optimized CSS keyframes for web rendering."""
        
        css_output = []
        generated_keyframes = set()
        
        for event in timeline['events']:
            if event['type'] in ['image_entry', 'image_exit', 'ken_burns', 'pulse_effect']:
                keyframe_name = self._generate_keyframe_name(event)
                
                if keyframe_name not in generated_keyframes:
                    css_keyframes = self._create_css_keyframes(event)
                    css_output.append(css_keyframes)
                    generated_keyframes.add(keyframe_name)
        
        return "\n\n".join(css_output)
    
    def _generate_keyframe_name(self, event: Dict) -> str:
        """Generate unique keyframe name for CSS."""
        
        animation_type = event['properties'].get('animation', event['type'])
        easing = event['properties'].get('easing', 'ease')
        
        return f"{animation_type}_{easing}".replace("-", "_").replace(".", "_")
    
    def _create_css_keyframes(self, event: Dict) -> str:
        """Create CSS keyframes for an animation event."""
        
        keyframe_name = self._generate_keyframe_name(event)
        from_props = event['properties'].get('from', {})
        to_props = event['properties'].get('to', {})
        
        css = f"@keyframes {keyframe_name} {{\n"
        css += f"  0% {{ {self._props_to_css(from_props)} }}\n"
        css += f"  100% {{ {self._props_to_css(to_props)} }}\n"
        css += "}"
        
        return css
    
    def _props_to_css(self, props: Dict[str, Any]) -> str:
        """Convert animation properties to CSS."""
        
        css_props = []
        
        for key, value in props.items():
            if key == 'opacity':
                css_props.append(f"opacity: {value}")
            elif key == 'scale':
                css_props.append(f"transform: scale({value})")
            elif key == 'translateX':
                css_props.append(f"transform: translateX({value}px)")
            elif key == 'translateY':
                css_props.append(f"transform: translateY({value}px)")
        
        return "; ".join(css_props) 