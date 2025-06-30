"""
Content Matching Service for Video Enhancement SaaS
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoSegment:
    """Video segment with timing info."""
    start_time: float
    end_time: float
    emphasized_entities: List[str]
    confidence: float
    text_content: str = ""

@dataclass  
class ImageMatch:
    """Matched image for video segment."""
    image_result: object
    video_segment: VideoSegment
    match_score: float
    start_time: float
    duration: float
    position: str = "top-right"
    opacity: float = 0.9

class ContentMatcher:
    """Content matching service for image-video synchronization."""
    
    def __init__(self):
        """Initialize the content matcher."""
        self.min_duration = 2.0
        self.max_duration = 8.0
        self.default_duration = 3.5
        
    async def match_images_to_video(self, 
                                   video_segments: List[VideoSegment],
                                   image_results: List[object],
                                   video_format: str = "portrait") -> List[ImageMatch]:
        """Match images to video segments."""
        
        matches = []
        
        for segment in video_segments:
            for image_result in image_results:
                if self._is_relevant(image_result, segment):
                    start_time, duration = self._calculate_timing(segment)
                    
                    match = ImageMatch(
                        image_result=image_result,
                        video_segment=segment, 
                        match_score=segment.confidence,
                        start_time=start_time,
                        duration=duration
                    )
                    
                    matches.append(match)
        
        # Sort by score and return top matches with safe type conversion
        def safe_match_score_key(m: ImageMatch) -> float:
            try:
                return float(m.match_score) if m.match_score is not None else 0.0
            except (ValueError, TypeError):
                return 0.0
        
        matches.sort(key=safe_match_score_key, reverse=True)
        return matches[:5]  # Return top 5 matches
    
    def _is_relevant(self, image_result: object, segment: VideoSegment) -> bool:
        """Check if image is relevant to segment."""
        image_entity = getattr(image_result, 'entity_name', '')
        
        for entity in segment.emphasized_entities:
            if entity.lower() in image_entity.lower():
                return True
        
        return False
    
    def _calculate_timing(self, segment: VideoSegment) -> Tuple[float, float]:
        """Calculate optimal timing for image display."""
        segment_duration = segment.end_time - segment.start_time
        
        # Default duration based on segment
        if segment_duration <= 3.0:
            duration = min(segment_duration * 0.8, self.max_duration)
        else:
            duration = self.default_duration
        
        duration = max(duration, self.min_duration)
        
        # Center in segment
        start_offset = (segment_duration - duration) / 2
        start_time = segment.start_time + start_offset
        
        return start_time, duration 