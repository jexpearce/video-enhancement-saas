"""
Data models and exceptions for video composition services.

Integrates with existing animation timeline, style system, and storage infrastructure.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

@dataclass
class CompositionConfig:
    """Configuration for video composition operations."""
    
    # Output video settings
    output_resolution: Tuple[int, int] = (1080, 1920)  # 9:16 for TikTok/Instagram
    output_fps: int = 30
    output_bitrate: str = "5M"
    output_codec: str = "libx264"
    output_format: str = "mp4"
    
    # Audio settings
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 44100
    
    # Overlay and animation settings
    overlay_opacity: float = 0.95
    overlay_duration_default: float = 3.0
    transition_duration: float = 0.5
    max_overlays_simultaneous: int = 3
    
    # Performance settings
    preset: str = "fast"  # ultrafast, superfast, veryfast, faster, fast, medium, slow
    threads: int = 0  # 0 = auto
    gpu_acceleration: bool = True
    
    # Quality settings
    crf: int = 23  # Constant Rate Factor (lower = higher quality)
    two_pass_encoding: bool = False
    
    # Platform-specific settings
    platform_optimizations: Dict[str, Dict] = field(default_factory=lambda: {
        'tiktok': {
            'resolution': (1080, 1920),
            'max_bitrate': '6M',
            'max_duration': 180,
            'optimize_for_mobile': True
        },
        'instagram_reels': {
            'resolution': (1080, 1920), 
            'max_bitrate': '5M',
            'max_duration': 90,
            'optimize_for_mobile': True
        },
        'youtube_shorts': {
            'resolution': (1080, 1920),
            'max_bitrate': '8M', 
            'max_duration': 60,
            'optimize_for_mobile': False
        }
    })

@dataclass
class CompositionAsset:
    """Represents an asset (image, video, audio) used in composition."""
    
    asset_id: str
    asset_type: str  # 'image', 'video', 'audio'
    local_path: str
    original_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Animation properties
    start_time: float = 0.0
    duration: float = 0.0
    position: str = "top-right"  # top-left, top-center, etc.
    size: str = "medium"  # small, medium, large
    animation_type: str = "fade"  # fade, slide, zoom, etc.
    
    # Effects
    effects: List[str] = field(default_factory=list)
    opacity: float = 1.0
    
    def __post_init__(self):
        """Validate asset exists and is accessible."""
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(f"Asset file not found: {self.local_path}")

@dataclass
class CompositionTimeline:
    """Timeline of composition events extracted from animation timeline."""
    
    duration: float
    events: List[Dict[str, Any]]
    overlays: List[CompositionAsset] = field(default_factory=list)
    captions: List[Dict[str, Any]] = field(default_factory=list)
    effects: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_events_at_time(self, timestamp: float, tolerance: float = 0.1) -> List[Dict]:
        """Get all events happening at a specific time."""
        return [
            event for event in self.events
            if event['start_time'] <= timestamp <= event['start_time'] + event.get('duration', 0)
        ]
    
    def get_active_overlays_at_time(self, timestamp: float) -> List[CompositionAsset]:
        """Get all overlays that should be visible at a specific time."""
        return [
            overlay for overlay in self.overlays
            if overlay.start_time <= timestamp <= overlay.start_time + overlay.duration
        ]

@dataclass
class CompositionResult:
    """Result of video composition operation."""
    
    success: bool
    output_path: str
    processing_time: float
    composition_config: CompositionConfig
    
    # Statistics
    total_overlays_applied: int = 0
    total_effects_applied: int = 0
    total_timeline_events: int = 0
    output_file_size_bytes: int = 0
    
    # Output metadata
    output_duration: float = 0.0
    output_resolution: Tuple[int, int] = (0, 0)
    output_fps: float = 0.0
    output_bitrate: int = 0
    
    # Processing details
    ffmpeg_command: str = ""
    processing_stages: List[Dict[str, Any]] = field(default_factory=list)
    assets_used: List[CompositionAsset] = field(default_factory=list)
    
    # Error information (if success=False)
    error_message: Optional[str] = None
    error_stage: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for API responses."""
        return {
            'success': self.success,
            'output_path': self.output_path,
            'processing_time': self.processing_time,
            'statistics': {
                'total_overlays': self.total_overlays_applied,
                'total_effects': self.total_effects_applied,
                'timeline_events': self.total_timeline_events,
                'file_size_mb': self.output_file_size_bytes / (1024 * 1024) if self.output_file_size_bytes else 0
            },
            'output_metadata': {
                'duration': self.output_duration,
                'resolution': f"{self.output_resolution[0]}x{self.output_resolution[1]}",
                'fps': self.output_fps,
                'bitrate': self.output_bitrate
            },
            'quality_metrics': self.quality_metrics,
            'error': {
                'message': self.error_message,
                'stage': self.error_stage,
                'details': self.error_details
            } if not self.success else None
        }

@dataclass
class FilterGraph:
    """Represents an FFmpeg filter graph for composition."""
    
    inputs: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    def add_input(self, input_spec: str) -> str:
        """Add input and return its reference label."""
        label = f"[{len(self.inputs)}:v]"
        self.inputs.append(input_spec)
        return label
    
    def add_filter(self, filter_spec: str) -> None:
        """Add a filter to the graph."""
        self.filters.append(filter_spec)
    
    def build(self) -> str:
        """Build the complete filter graph string."""
        return ";".join(self.filters) if self.filters else ""

# Exceptions
class CompositionError(Exception):
    """Base exception for composition errors."""
    
    def __init__(self, message: str, stage: str = "unknown", details: Optional[Dict] = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}

class FFmpegError(CompositionError):
    """FFmpeg-specific errors."""
    
    def __init__(self, message: str, command: str = "", stderr: str = ""):
        super().__init__(message, "ffmpeg_execution")
        self.command = command
        self.stderr = stderr

class AssetError(CompositionError):
    """Asset-related errors (missing files, invalid formats, etc.)."""
    pass

class TimelineError(CompositionError):
    """Timeline processing errors."""
    pass

class QualityError(CompositionError):
    """Quality assessment and optimization errors."""
    pass 