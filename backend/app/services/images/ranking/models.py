"""
Data models for image ranking and selection system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from ..curation.curator import CuratedImage, WordContext
from ....database.models import EnrichedEntity


@dataclass
class RankingFeatures:
    """Comprehensive features extracted for image ranking."""
    
    # Relevance features
    clip_similarity_score: float        # CLIP model similarity to text
    semantic_match_score: float         # Semantic similarity to entity
    context_match_score: float          # Match to surrounding context
    entity_type_match: bool             # Does image match entity type
    
    # Quality features  
    visual_quality_score: float         # Overall image quality
    resolution_score: float             # Resolution quality (0-1)
    aspect_ratio_score: float           # How well aspect ratio fits video
    sharpness_score: float              # Image sharpness/focus
    
    # Engagement features
    popularity_score: float             # Likes, downloads, views
    recency_score: float                # How recent the image is
    
    # Visual features
    color_vibrancy: float               # Color saturation and appeal
    composition_score: float            # Rule of thirds, balance, etc.
    text_space_available: bool          # Space for text overlays
    face_detection_score: float         # Quality of detected faces (for people)
    
    # Diversity features
    visual_signature: str               # Perceptual hash for similarity
    color_palette: List[str]            # Dominant colors
    visual_complexity: float            # How visually busy the image is
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array([
            self.clip_similarity_score,
            self.semantic_match_score,
            self.context_match_score,
            1.0 if self.entity_type_match else 0.0,
            self.visual_quality_score,
            self.resolution_score,
            self.aspect_ratio_score,
            self.sharpness_score,
            self.popularity_score,
            self.recency_score,
            self.color_vibrancy,
            self.composition_score,
            1.0 if self.text_space_available else 0.0,
            self.face_detection_score,
            self.visual_complexity
        ])


@dataclass
class RankedImage:
    """An image with its ranking score and metadata."""
    
    curated_image: CuratedImage         # Original curated image
    ranking_features: RankingFeatures   # Extracted features
    
    # Scoring results
    base_score: float                   # ML model base score
    contextual_adjustments: Dict[str, float]  # Applied adjustments
    diversity_penalty: float            # Penalty for visual similarity
    final_score: float                  # Final ranking score
    
    # Selection metadata
    selected_for_emphasis: Optional[int] = None  # Which emphasis point (index)
    display_duration: Optional[float] = None     # How long to show
    confidence_score: Optional[float] = None     # Confidence in selection
    
    @property
    def image_id(self) -> str:
        """Get the underlying image ID."""
        return self.curated_image.source.id
    
    @property
    def image_url(self) -> str:
        """Get the image URL."""
        return self.curated_image.source.url
    
    def get_score_breakdown(self) -> Dict[str, float]:
        """Get detailed score breakdown for debugging."""
        return {
            'base_score': self.base_score,
            'diversity_penalty': self.diversity_penalty,
            'final_score': self.final_score,
            **self.contextual_adjustments
        }


@dataclass
class VideoMetadata:
    """Metadata about the video being processed."""
    
    duration: float                     # Video duration in seconds
    aspect_ratio: str                   # "16:9", "9:16", "1:1", etc.
    fps: int                           # Frames per second
    resolution: tuple[int, int]        # Width, height
    genre: Optional[str] = None        # "news", "entertainment", "educational"
    target_platform: Optional[str] = None  # "tiktok", "instagram", "youtube"
    language: Optional[str] = None     # "en", "es", "fr", etc.


@dataclass
class RankingResult:
    """Result of image ranking operation."""
    
    entity: EnrichedEntity             # Entity being ranked for
    context: WordContext               # Word context
    ranked_images: List[RankedImage]   # Images sorted by score
    
    # Ranking metadata
    total_candidates: int              # Total images considered
    processing_time_ms: float          # Time to rank
    model_version: str                 # Which ranking model used
    feature_importance: Dict[str, float]  # Feature importance scores
    
    # Quality metrics
    score_distribution: Dict[str, float]  # Min, max, mean, std of scores
    diversity_metrics: Dict[str, float]   # Diversity measurements
    
    def get_top_images(self, count: int = 5) -> List[RankedImage]:
        """Get top N ranked images."""
        return self.ranked_images[:count]
    
    def get_score_statistics(self) -> Dict[str, float]:
        """Get statistical summary of scores."""
        scores = [img.final_score for img in self.ranked_images]
        return {
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'std_score': float(np.std(scores)) if scores else 0.0,
            'median_score': float(np.median(scores)) if scores else 0.0
        }


@dataclass
class RankingConfig:
    """Configuration for ranking algorithm."""
    
    # Feature weights (should sum to 1.0)
    feature_weights: Optional[Dict[str, float]] = None
    
    # Diversity settings
    diversity_penalty_strength: float = 0.3
    max_similar_images: int = 2
    max_similar_colors: int = 2         # Max images with similar color palette
    similarity_threshold: float = 0.8
    
    # Quality thresholds
    min_quality_score: float = 0.3
    min_resolution_score: float = 0.2
    
    # Context settings
    context_window_boost: float = 0.2   # Boost for contextually relevant images
    entity_type_boost: float = 0.15     # Boost for matching entity type
    
    # Model settings
    model_name: str = "default_ranking_v1"
    use_ensemble: bool = False
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        """Set default feature weights if not provided."""
        if self.feature_weights is None:
            self.feature_weights = {
                'relevance': 0.35,
                'quality': 0.25, 
                'engagement': 0.15,
                'recency': 0.10,
                'diversity': 0.15
            }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.feature_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Feature weights must sum to 1.0, got {total_weight}")


@dataclass
class SelectionStrategy:
    """Strategy for selecting images from ranked results."""
    
    strategy_name: str                  # "top_n", "diverse_selection", "temporal_spread"
    max_images_per_entity: int = 3      # Max images per entity
    max_total_images: int = 10          # Max total images
    min_score_threshold: float = 0.4    # Minimum score to consider
    
    # Temporal distribution
    prefer_temporal_spread: bool = True  # Spread images across video
    min_time_gap: float = 2.0           # Minimum seconds between images
    
    # Diversity requirements
    enforce_visual_diversity: bool = True
    max_similar_colors: int = 2         # Max images with similar color palette
    require_different_compositions: bool = True 