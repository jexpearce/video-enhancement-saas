"""
Image Selection using ML Ranking.
"""

import logging
from typing import List, Dict, Optional

from .models import RankedImage, SelectionStrategy, VideoMetadata
from .ranker import ImageRankingEngine


logger = logging.getLogger(__name__)


class ImageSelector:
    """High-level image selection using ML ranking."""
    
    def __init__(self, ranking_engine: ImageRankingEngine = None):
        """Initialize with ranking engine."""
        self.ranking_engine = ranking_engine or ImageRankingEngine()
    
    def select_top_images(
        self, 
        ranked_images: List[RankedImage], 
        max_count: int = 10,
        min_score: float = 0.4
    ) -> List[RankedImage]:
        """Select top N images with diversity."""
        
        # Filter by score threshold
        candidates = [
            img for img in ranked_images 
            if img.final_score >= min_score
        ]
        
        # Sort by score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply diversity selection
        selected = []
        seen_signatures = set()
        
        for img in candidates:
            if len(selected) >= max_count:
                break
            
            # Check visual diversity
            signature = img.ranking_features.visual_signature
            
            # Skip if too similar visually
            if signature in seen_signatures:
                continue
            
            # Add to selection
            selected.append(img)
            seen_signatures.add(signature)
        
        logger.info(f"Selected {len(selected)} diverse images from {len(candidates)} candidates")
        return selected 