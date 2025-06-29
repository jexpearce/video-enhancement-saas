"""
Animation Synchronization

Handles synchronization of animations with:
- Emphasis points in speech
- Audio beats and rhythm
- Video timing and pacing
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class SyncPoint:
    """Represents a synchronization point for animations."""
    time: float
    intensity: float
    type: str  # 'emphasis', 'beat', 'silence'
    confidence: float = 1.0


class EmphasisSynchronizer:
    """
    Synchronizes animations with emphasis points in speech.
    """
    
    def __init__(self, config):
        """Initialize emphasis synchronizer."""
        self.config = config
        self.sync_tolerance = 0.3  # Seconds
    
    def create_emphasis_sync_events(
        self,
        emphasis_points: List[Dict],
        existing_timeline: Dict,
        style: Dict
    ) -> List[Dict]:
        """Create animation events synchronized with emphasis points."""
        
        sync_events = []
        
        for point in emphasis_points:
            start_time = point.get('start_time', 0)
            emphasis_score = point.get('emphasis_score', 0.5)
            word_text = point.get('word_text', '')
            
            # Find closest animation event to sync with
            closest_event = self._find_closest_animation(
                start_time, existing_timeline['events']
            )
            
            if closest_event:
                # Create synchronized pulse effect
                sync_event = {
                    'type': 'emphasis_sync',
                    'target_id': closest_event['target_id'],
                    'start_time': start_time,
                    'duration': 0.3,
                    'properties': {
                        'intensity': emphasis_score,
                        'effect': 'pulse',
                        'sync_word': word_text,
                        'easing': 'ease_out_back'
                    }
                }
                sync_events.append(sync_event)
        
        return sync_events
    
    def _find_closest_animation(
        self, 
        target_time: float, 
        events: List[Dict]
    ) -> Optional[Dict]:
        """Find the closest animation event to synchronize with."""
        
        closest_event = None
        min_distance = float('inf')
        
        for event in events:
            if event['type'] in ['image_entry', 'image_exit']:
                distance = abs(event['start_time'] - target_time)
                
                if distance < min_distance and distance < self.sync_tolerance:
                    min_distance = distance
                    closest_event = event
        
        return closest_event
    
    def adjust_timing_for_emphasis(
        self,
        events: List[Dict],
        emphasis_points: List[Dict]
    ) -> List[Dict]:
        """Adjust animation timing to better align with emphasis points."""
        
        adjusted_events = []
        
        for event in events:
            adjusted_event = event.copy()
            
            # Find nearby emphasis points
            nearby_emphasis = [
                p for p in emphasis_points
                if abs(p.get('start_time', 0) - event['start_time']) < self.sync_tolerance
            ]
            
            if nearby_emphasis:
                # Adjust timing to closest emphasis
                closest_emphasis = min(
                    nearby_emphasis,
                    key=lambda p: abs(p.get('start_time', 0) - event['start_time'])
                )
                
                # Slightly adjust start time
                time_diff = closest_emphasis.get('start_time', 0) - event['start_time']
                if abs(time_diff) < 0.2:  # Only small adjustments
                    adjusted_event['start_time'] += time_diff * 0.5
            
            adjusted_events.append(adjusted_event)
        
        return adjusted_events


class BeatSynchronizer:
    """
    Synchronizes animations with audio beats and rhythm.
    """
    
    def __init__(self, config):
        """Initialize beat synchronizer."""
        self.config = config
        self.beat_tolerance = config.beat_sync_tolerance
    
    def create_beat_sync_events(
        self,
        timeline: Dict,
        audio_beats: List[float],
        style: Dict
    ) -> List[Dict]:
        """Create beat synchronization events."""
        
        if not audio_beats:
            return []
        
        beat_events = []
        
        for beat_time in audio_beats:
            # Find animation events near this beat
            nearby_events = self._find_events_near_beat(
                beat_time, timeline['events']
            )
            
            for event in nearby_events:
                beat_event = {
                    'type': 'beat_sync',
                    'target_id': event['target_id'],
                    'start_time': beat_time,
                    'duration': 0.1,
                    'properties': {
                        'intensity': 0.3,
                        'effect': 'scale_pulse',
                        'beat_strength': self._calculate_beat_strength(beat_time, audio_beats)
                    }
                }
                beat_events.append(beat_event)
        
        return beat_events
    
    def _find_events_near_beat(
        self, 
        beat_time: float, 
        events: List[Dict]
    ) -> List[Dict]:
        """Find animation events near a beat time."""
        
        nearby_events = []
        
        for event in events:
            # Check if event is active during beat time
            event_start = event['start_time']
            event_end = event_start + event['duration']
            
            if (event_start <= beat_time <= event_end or 
                abs(event_start - beat_time) < self.beat_tolerance):
                
                if event['type'] in ['image_entry', 'image_exit', 'ken_burns']:
                    nearby_events.append(event)
        
        return nearby_events
    
    def _calculate_beat_strength(
        self, 
        beat_time: float, 
        all_beats: List[float]
    ) -> float:
        """Calculate the strength/intensity of a beat."""
        
        # Find intervals between beats
        beat_index = None
        for i, bt in enumerate(all_beats):
            if abs(bt - beat_time) < 0.01:  # Found the beat
                beat_index = i
                break
        
        if beat_index is None:
            return 0.5  # Default strength
        
        # Calculate strength based on position and intervals
        if beat_index == 0 or beat_index == len(all_beats) - 1:
            return 0.8  # Strong beats at start/end
        
        # Check if this is on a strong beat (every 4th beat typically)
        if beat_index % 4 == 0:
            return 1.0  # Strongest beat
        elif beat_index % 2 == 0:
            return 0.7  # Medium beat
        else:
            return 0.4  # Weak beat
    
    def snap_to_beats(
        self,
        events: List[Dict],
        audio_beats: List[float]
    ) -> List[Dict]:
        """Snap animation events to nearby beats."""
        
        if not audio_beats:
            return events
        
        snapped_events = []
        
        for event in events:
            snapped_event = event.copy()
            
            # Find closest beat
            closest_beat = min(
                audio_beats,
                key=lambda beat: abs(beat - event['start_time'])
            )
            
            # Snap if close enough
            time_diff = abs(closest_beat - event['start_time'])
            if time_diff < self.beat_tolerance:
                snapped_event['start_time'] = closest_beat
                snapped_event['properties'] = snapped_event.get('properties', {})
                snapped_event['properties']['snapped_to_beat'] = True
            
            snapped_events.append(snapped_event)
        
        return snapped_events


class TempoAnalyzer:
    """
    Analyzes audio tempo and rhythm for better synchronization.
    """
    
    def __init__(self):
        """Initialize tempo analyzer."""
        pass
    
    def analyze_tempo(self, audio_beats: List[float]) -> Dict[str, Any]:
        """Analyze tempo from beat timestamps."""
        
        if len(audio_beats) < 2:
            return {'bpm': 120, 'confidence': 0.0, 'tempo_variation': 0.0}
        
        # Calculate intervals between beats
        intervals = []
        for i in range(1, len(audio_beats)):
            interval = audio_beats[i] - audio_beats[i-1]
            intervals.append(interval)
        
        # Calculate BPM
        avg_interval = sum(intervals) / len(intervals)
        bpm = 60.0 / avg_interval if avg_interval > 0 else 120
        
        # Calculate tempo variation (stability)
        if len(intervals) > 1:
            variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
            tempo_variation = math.sqrt(variance) / avg_interval
        else:
            tempo_variation = 0.0
        
        # Calculate confidence based on consistency
        confidence = max(0.0, 1.0 - tempo_variation)
        
        return {
            'bpm': round(bpm, 1),
            'confidence': confidence,
            'tempo_variation': tempo_variation,
            'avg_beat_interval': avg_interval,
            'total_beats': len(audio_beats)
        }
    
    def detect_tempo_changes(
        self, 
        audio_beats: List[float],
        window_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Detect tempo changes throughout the audio."""
        
        tempo_changes = []
        
        if len(audio_beats) < window_size:
            return tempo_changes
        
        for i in range(0, len(audio_beats) - window_size, window_size // 2):
            window_beats = audio_beats[i:i + window_size]
            tempo_info = self.analyze_tempo(window_beats)
            
            tempo_changes.append({
                'start_time': window_beats[0],
                'end_time': window_beats[-1],
                'bpm': tempo_info['bpm'],
                'confidence': tempo_info['confidence']
            })
        
        return tempo_changes


class SilenceDetector:
    """
    Detects silence periods for strategic animation timing.
    """
    
    def __init__(self, silence_threshold: float = 0.5):
        """Initialize silence detector."""
        self.silence_threshold = silence_threshold
    
    def detect_silence_periods(
        self,
        emphasis_points: List[Dict],
        video_duration: float
    ) -> List[Tuple[float, float]]:
        """Detect periods of silence between emphasis points."""
        
        if not emphasis_points:
            return [(0.0, video_duration)]
        
        silence_periods = []
        
        # Sort emphasis points by time
        sorted_points = sorted(emphasis_points, key=lambda p: p.get('start_time', 0))
        
        # Check for silence at the beginning
        first_emphasis = sorted_points[0].get('start_time', 0)
        if first_emphasis > self.silence_threshold:
            silence_periods.append((0.0, first_emphasis))
        
        # Check for silence between emphasis points
        for i in range(1, len(sorted_points)):
            prev_end = sorted_points[i-1].get('end_time', sorted_points[i-1].get('start_time', 0))
            curr_start = sorted_points[i].get('start_time', 0)
            
            gap_duration = curr_start - prev_end
            if gap_duration > self.silence_threshold:
                silence_periods.append((prev_end, curr_start))
        
        # Check for silence at the end
        last_emphasis = sorted_points[-1].get('end_time', sorted_points[-1].get('start_time', 0))
        if video_duration - last_emphasis > self.silence_threshold:
            silence_periods.append((last_emphasis, video_duration))
        
        return silence_periods
    
    def suggest_animation_timing(
        self,
        silence_periods: List[Tuple[float, float]],
        emphasis_points: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Suggest optimal timing for animations based on silence."""
        
        suggestions = []
        
        for start_time, end_time in silence_periods:
            duration = end_time - start_time
            
            if duration > 2.0:  # Long silence - good for transitions
                suggestions.append({
                    'type': 'transition_opportunity',
                    'start_time': start_time + 0.2,
                    'duration': duration - 0.4,
                    'confidence': 0.9,
                    'recommended_effect': 'crossfade'
                })
            elif duration > 1.0:  # Medium silence - good for subtle effects
                suggestions.append({
                    'type': 'subtle_animation',
                    'start_time': start_time + 0.1,
                    'duration': duration - 0.2,
                    'confidence': 0.7,
                    'recommended_effect': 'ken_burns'
                })
        
        return suggestions 