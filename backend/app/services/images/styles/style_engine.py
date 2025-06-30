"""
Style Engine - Main Orchestrator for Visual Style System.

Coordinates automatic style selection, manual overrides, and style application
for TikTok and Instagram content creators.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .models import (
    StyleTemplate,
    VisualStyle,
    Platform,
    ContentType,
    StyleConfig
)
from .auto_selector import AutoStyleSelector
from .template_manager import TemplateManager

logger = logging.getLogger(__name__)


class StyleApplicationResult:
    """Result of style application process."""
    
    def __init__(
        self,
        visual_style: VisualStyle,
        selection_method: str,
        confidence_score: float,
        processing_time_ms: float,
        alternatives: List[StyleTemplate] = None
    ):
        self.visual_style = visual_style
        self.selection_method = selection_method  # "auto", "manual", "recommendation"
        self.confidence_score = confidence_score
        self.processing_time_ms = processing_time_ms
        self.alternatives = alternatives or []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "applied_template": {
                "id": self.visual_style.template.template_id,
                "name": self.visual_style.template.name,
                "description": self.visual_style.template.description
            },
            "selection_method": self.selection_method,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
            "customizations_applied": {
                "custom_colors": self.visual_style.custom_colors is not None,
                "custom_position": self.visual_style.custom_position is not None,
                "intensity_level": self.visual_style.intensity_level,
                "speed_multiplier": self.visual_style.speed_multiplier
            },
            "alternatives": [
                {
                    "id": alt.template_id,
                    "name": alt.name,
                    "engagement_score": alt.get_engagement_score()
                }
                for alt in self.alternatives
            ]
        }


class StyleEngine:
    """
    Main style engine for TikTok/Instagram creators.
    
    Features:
    - Automatic style selection based on content analysis
    - Manual override with creator preferences
    - Template customization and creation
    - Performance tracking and learning
    """
    
    def __init__(self, config: StyleConfig = None):
        """Initialize style engine."""
        self.config = config or StyleConfig()
        self.auto_selector = AutoStyleSelector(self.config)
        self.template_manager = TemplateManager(self.config)
        
        logger.info("StyleEngine initialized for TikTok/Instagram creators")
    
    async def apply_style_to_video(
        self,
        video_id: str,
        audio_transcript: str,
        video_title: str = "",
        creator_id: str = None,
        creator_age: int = None,
        creator_gender: str = None,
        target_platform: Platform = Platform.TIKTOK,
        manual_template_id: str = None,
        customizations: Dict = None
    ) -> StyleApplicationResult:
        """
        Apply optimal style to a video.
        
        Args:
            video_id: Unique video identifier
            audio_transcript: Video audio transcript for analysis
            video_title: Video title for context
            creator_id: Creator identifier for personalization
            creator_age: Creator age for demographic targeting
            creator_gender: Creator gender for demographic targeting
            target_platform: Target social media platform
            manual_template_id: Manual template override
            customizations: Custom style modifications
            
        Returns:
            StyleApplicationResult with applied style and metadata
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Applying style to video {video_id} for platform {target_platform.value}")
            
            # Determine selection method
            if manual_template_id:
                # Manual override path
                result = await self._apply_manual_style(
                    video_id=video_id,
                    creator_id=creator_id,
                    template_id=manual_template_id,
                    customizations=customizations
                )
                selection_method = "manual"
                
            else:
                # Automatic selection path
                result = await self._apply_automatic_style(
                    video_id=video_id,
                    audio_transcript=audio_transcript,
                    video_title=video_title,
                    creator_id=creator_id,
                    creator_age=creator_age,
                    creator_gender=creator_gender,
                    target_platform=target_platform,
                    customizations=customizations
                )
                selection_method = "auto"
            
            # Calculate processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get alternative recommendations
            alternatives = await self._get_alternatives(
                audio_transcript=audio_transcript,
                target_platform=target_platform,
                applied_template_id=result.visual_style.template.template_id
            )
            
            final_result = StyleApplicationResult(
                visual_style=result.visual_style,
                selection_method=selection_method,
                confidence_score=result.confidence_score,
                processing_time_ms=processing_time,
                alternatives=alternatives
            )
            
            logger.info(
                f"Applied style '{result.visual_style.template.name}' to video {video_id} "
                f"in {processing_time:.1f}ms"
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Failed to apply style to video {video_id}: {str(e)}")
            raise
    
    async def get_creator_recommendations(
        self,
        creator_id: str,
        platform: Platform,
        content_type: ContentType = None,
        count: int = 5
    ) -> List[Dict]:
        """Get personalized template recommendations for creator."""
        
        # Auto-detect content type if not provided
        if not content_type:
            content_type = ContentType.LIFESTYLE  # Default
        
        recommendations = self.template_manager.get_recommended_templates(
            creator_id=creator_id,
            platform=platform,
            content_type=content_type,
            count=count
        )
        
        # Enhance with trending information
        enhanced_recommendations = []
        for rec in recommendations:
            template = rec["template"]
            
            enhanced_rec = {
                "template": {
                    "id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "engagement_score": template.get_engagement_score(),
                    "popularity_score": template.popularity_score,
                    "preview_colors": {
                        "primary": template.color_scheme.primary,
                        "secondary": template.color_scheme.secondary,
                        "background": template.color_scheme.background
                    },
                    "features": {
                        "has_particles": template.entrance_animation.has_particles,
                        "has_glow": template.entrance_animation.has_glow,
                        "pulse_to_beat": template.has_pulse_to_beat,
                        "emoji_reactions": template.has_emoji_reactions
                    }
                },
                "score": rec["score"],
                "reasoning": rec["reasoning"],
                "trending": template.popularity_score > 0.8
            }
            
            enhanced_recommendations.append(enhanced_rec)
        
        return enhanced_recommendations
    
    async def create_custom_template(
        self,
        creator_id: str,
        base_template_id: str,
        customizations: Dict,
        name: str = None
    ) -> StyleTemplate:
        """Create a custom template for a creator."""
        
        custom_template = self.template_manager.create_custom_template(
            creator_id=creator_id,
            base_template_id=base_template_id,
            customizations=customizations,
            name=name
        )
        
        logger.info(f"Created custom template for creator {creator_id}")
        
        return custom_template
    
    async def update_creator_preferences(
        self,
        creator_id: str,
        preferences: Dict[str, Any]
    ):
        """Update creator style preferences."""
        
        self.template_manager.update_creator_preferences(creator_id, preferences)
        
        logger.info(f"Updated preferences for creator {creator_id}")
    
    async def record_video_performance(
        self,
        video_id: str,
        creator_id: str,
        template_id: str,
        engagement_metrics: Dict[str, float]
    ):
        """Record video performance for style learning."""
        
        self.template_manager.record_template_usage(
            creator_id=creator_id,
            template_id=template_id,
            engagement_metrics=engagement_metrics
        )
        
        # Also update auto-selector
        await self.auto_selector.analyze_template_performance(
            template_id=template_id,
            engagement_data=engagement_metrics
        )
        
        logger.info(f"Recorded performance for video {video_id}")
    
    def get_available_templates(
        self,
        platform: Platform = None,
        content_type: ContentType = None
    ) -> List[Dict]:
        """Get available templates with metadata."""
        
        templates = self.template_manager.get_available_templates(platform, content_type)
        
        template_data = []
        for template in templates:
            template_data.append({
                "id": template.template_id,
                "name": template.name,
                "description": template.description,
                "engagement_score": template.get_engagement_score(),
                "popularity_score": template.popularity_score,
                "target_platforms": [p.value for p in template.target_platforms],
                "content_types": [c.value for c in template.content_types],
                "preview": {
                    "colors": {
                        "primary": template.color_scheme.primary,
                        "secondary": template.color_scheme.secondary,
                        "gradient_start": template.color_scheme.gradient_start,
                        "gradient_end": template.color_scheme.gradient_end
                    },
                    "position": template.position_style.value,
                    "animation": template.entrance_animation.animation_type.value,
                    "has_particles": template.entrance_animation.has_particles,
                    "emoji_set": template.emoji_set
                },
                "features": {
                    "ken_burns": template.has_ken_burns,
                    "parallax": template.has_parallax,
                    "pulse_to_beat": template.has_pulse_to_beat,
                    "emoji_reactions": template.has_emoji_reactions,
                    "background_blur": template.has_background_blur
                }
            })
        
        return template_data
    
    def get_style_analytics(self) -> Dict:
        """Get analytics for the style system."""
        
        analytics = {
            "total_templates": len(self.template_manager.templates),
            "total_creators": len(self.template_manager.creator_preferences),
            "most_popular_templates": [],
            "trending_features": {},
            "platform_preferences": {}
        }
        
        # Get most popular templates with safe type conversion
        all_templates = list(self.template_manager.templates.values())
        
        def safe_popularity_score_key(t) -> float:
            try:
                return float(t.popularity_score) if t.popularity_score is not None else 0.0
            except (ValueError, TypeError):
                return 0.0
        
        all_templates.sort(key=safe_popularity_score_key, reverse=True)
        
        analytics["most_popular_templates"] = [
            {
                "id": t.template_id,
                "name": t.name,
                "popularity": t.popularity_score,
                "engagement": t.get_engagement_score()
            }
            for t in all_templates[:5]
        ]
        
        # Calculate trending features
        feature_counts = {
            "particle_effects": 0,
            "neon_colors": 0,
            "pulse_to_beat": 0,
            "emoji_reactions": 0
        }
        
        for template in all_templates:
            if template.entrance_animation.has_particles:
                feature_counts["particle_effects"] += 1
            if "neon" in template.color_scheme.name.lower():
                feature_counts["neon_colors"] += 1
            if template.has_pulse_to_beat:
                feature_counts["pulse_to_beat"] += 1
            if template.has_emoji_reactions:
                feature_counts["emoji_reactions"] += 1
        
        analytics["trending_features"] = feature_counts
        
        return analytics
    
    async def _apply_manual_style(
        self,
        video_id: str,
        creator_id: str,
        template_id: str,
        customizations: Dict = None
    ) -> 'StyleApplicationResult':
        """Apply manually selected style."""
        
        visual_style = self.template_manager.apply_manual_override(
            creator_id=creator_id,
            video_id=video_id,
            template_id=template_id,
            customizations=customizations
        )
        
        return StyleApplicationResult(
            visual_style=visual_style,
            selection_method="manual",
            confidence_score=1.0,  # Manual selection has full confidence
            processing_time_ms=0,  # Will be calculated by caller
            alternatives=[]
        )
    
    async def _apply_automatic_style(
        self,
        video_id: str,
        audio_transcript: str,
        video_title: str,
        creator_id: str,
        creator_age: int,
        creator_gender: str,
        target_platform: Platform,
        customizations: Dict = None
    ) -> 'StyleApplicationResult':
        """Apply automatically selected style."""
        
        # Get creator preferences for auto-selection
        creator_preferences = {}
        if creator_id:
            prefs = self.template_manager._get_creator_preferences(creator_id)
            creator_preferences = prefs.preferences
        
        # Select optimal template
        selected_template, confidence = await self.auto_selector.select_optimal_style(
            audio_transcript=audio_transcript,
            video_title=video_title,
            target_platform=target_platform,
            creator_age=creator_age,
            creator_gender=creator_gender,
            creator_preferences=creator_preferences
        )
        
        # Create visual style
        visual_style = VisualStyle(
            template=selected_template,
            video_id=video_id,
            applied_at=datetime.now().isoformat()
        )
        
        # Apply customizations if provided
        if customizations:
            if "colors" in customizations:
                visual_style.custom_colors = self.template_manager._create_custom_color_scheme(
                    customizations["colors"]
                )
            
            if "position" in customizations:
                visual_style.custom_position = PositionStyle(customizations["position"])
            
            if "intensity" in customizations:
                visual_style.intensity_level = customizations["intensity"]
            
            if "speed" in customizations:
                visual_style.speed_multiplier = customizations["speed"]
        
        return StyleApplicationResult(
            visual_style=visual_style,
            selection_method="auto",
            confidence_score=confidence,
            processing_time_ms=0,  # Will be calculated by caller
            alternatives=[]
        )
    
    async def _get_alternatives(
        self,
        audio_transcript: str,
        target_platform: Platform,
        applied_template_id: str,
        count: int = 3
    ) -> List[StyleTemplate]:
        """Get alternative template recommendations."""
        
        # Analyze content to get recommendations
        content_type = self.auto_selector.content_analyzer.analyze_content_type(
            audio_transcript, ""
        )
        
        recommendations = self.auto_selector.get_style_recommendations(
            content_type=content_type,
            target_platform=target_platform,
            count=count + 1  # Get extra to filter out applied template
        )
        
        # Filter out the applied template and return alternatives
        alternatives = [
            rec[0] for rec in recommendations
            if rec[0].template_id != applied_template_id
        ]
        
        return alternatives[:count] 