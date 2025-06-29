"""
Template Manager - Creator Template Selection and Customization.

Handles template management with manual override capabilities,
creator preferences, and customization options.
"""

import logging
import json
from typing import List, Dict, Optional, Any
from datetime import datetime

from .models import (
    StyleTemplate,
    VisualStyle,
    ColorScheme,
    PositionStyle,
    AnimationEffect,
    Platform,
    ContentType,
    VIRAL_TEMPLATES,
    StyleConfig
)

logger = logging.getLogger(__name__)


class CreatorPreferences:
    """Manages individual creator preferences and customizations."""
    
    def __init__(self, creator_id: str):
        self.creator_id = creator_id
        self.preferences = {
            "favorite_templates": [],
            "preferred_styles": [],
            "animation_intensity": "medium",  # "low", "medium", "high"
            "color_preferences": [],
            "position_preferences": [],
            "custom_templates": [],
            "engagement_history": {}
        }
    
    def add_favorite_template(self, template_id: str):
        """Add a template to favorites."""
        if template_id not in self.preferences["favorite_templates"]:
            self.preferences["favorite_templates"].append(template_id)
            logger.info(f"Added template {template_id} to favorites for creator {self.creator_id}")
    
    def remove_favorite_template(self, template_id: str):
        """Remove a template from favorites."""
        if template_id in self.preferences["favorite_templates"]:
            self.preferences["favorite_templates"].remove(template_id)
    
    def update_style_preferences(self, preferences: Dict[str, Any]):
        """Update style preferences from user input."""
        valid_keys = [
            "preferred_styles", "animation_intensity", "color_preferences",
            "position_preferences"
        ]
        
        for key, value in preferences.items():
            if key in valid_keys:
                self.preferences[key] = value
        
        logger.info(f"Updated preferences for creator {self.creator_id}")
    
    def record_engagement(self, template_id: str, metrics: Dict[str, float]):
        """Record engagement metrics for template usage."""
        if template_id not in self.preferences["engagement_history"]:
            self.preferences["engagement_history"][template_id] = []
        
        self.preferences["engagement_history"][template_id].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
    
    def get_template_performance(self, template_id: str) -> Dict[str, float]:
        """Get performance metrics for a template."""
        history = self.preferences["engagement_history"].get(template_id, [])
        
        if not history:
            return {}
        
        # Calculate averages
        total_metrics = {}
        for record in history:
            for metric, value in record["metrics"].items():
                if metric not in total_metrics:
                    total_metrics[metric] = []
                total_metrics[metric].append(value)
        
        avg_metrics = {}
        for metric, values in total_metrics.items():
            avg_metrics[f"avg_{metric}"] = sum(values) / len(values)
        
        return avg_metrics
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "creator_id": self.creator_id,
            "preferences": self.preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CreatorPreferences':
        """Create from dictionary."""
        instance = cls(data["creator_id"])
        instance.preferences = data["preferences"]
        return instance


class TemplateManager:
    """Manages style templates with creator customization and override."""
    
    def __init__(self, config: StyleConfig = None):
        """Initialize template manager."""
        self.config = config or StyleConfig()
        self.templates = dict(VIRAL_TEMPLATES)  # Copy viral templates
        self.creator_preferences = {}  # Map creator_id -> CreatorPreferences
        
        logger.info(f"TemplateManager initialized with {len(self.templates)} templates")
    
    def get_available_templates(
        self,
        platform: Platform = None,
        content_type: ContentType = None
    ) -> List[StyleTemplate]:
        """Get list of available templates with optional filtering."""
        
        templates = list(self.templates.values())
        
        # Filter by platform
        if platform:
            templates = [
                t for t in templates
                if platform in t.target_platforms or Platform.UNIVERSAL in t.target_platforms
            ]
        
        # Filter by content type
        if content_type:
            templates = [
                t for t in templates
                if content_type in t.content_types or len(t.content_types) == 0
            ]
        
        # Sort by popularity
        templates.sort(key=lambda t: t.popularity_score, reverse=True)
        
        return templates
    
    def get_template_by_id(self, template_id: str) -> Optional[StyleTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def get_creator_favorites(self, creator_id: str) -> List[StyleTemplate]:
        """Get creator's favorite templates."""
        prefs = self._get_creator_preferences(creator_id)
        favorites = []
        
        for template_id in prefs.preferences["favorite_templates"]:
            template = self.get_template_by_id(template_id)
            if template:
                favorites.append(template)
        
        return favorites
    
    def get_recommended_templates(
        self,
        creator_id: str,
        platform: Platform,
        content_type: ContentType,
        count: int = 5
    ) -> List[Dict]:
        """Get personalized template recommendations."""
        
        prefs = self._get_creator_preferences(creator_id)
        available_templates = self.get_available_templates(platform, content_type)
        
        scored_templates = []
        
        for template in available_templates:
            score = self._score_template_for_creator(template, prefs)
            
            # Generate recommendation reason
            reasons = []
            if template.template_id in prefs.preferences["favorite_templates"]:
                reasons.append("One of your favorites")
            
            if template.popularity_score > 0.85:
                reasons.append("Trending now")
            
            if any(style in template.name.lower() for style in prefs.preferences["preferred_styles"]):
                reasons.append("Matches your style preferences")
            
            performance = prefs.get_template_performance(template.template_id)
            if performance.get("avg_engagement_rate", 0) > 0.7:
                reasons.append("High engagement in your past videos")
            
            if not reasons:
                reasons.append("Popular with similar creators")
            
            scored_templates.append({
                "template": template,
                "score": score,
                "reasoning": " • ".join(reasons)
            })
        
        # Sort by score and return top N
        scored_templates.sort(key=lambda x: x["score"], reverse=True)
        return scored_templates[:count]
    
    def create_custom_template(
        self,
        creator_id: str,
        base_template_id: str,
        customizations: Dict,
        name: str = None
    ) -> StyleTemplate:
        """Create a custom template based on an existing one."""
        
        base_template = self.get_template_by_id(base_template_id)
        if not base_template:
            raise ValueError(f"Base template {base_template_id} not found")
        
        # Generate custom template ID
        custom_id = f"{base_template_id}_custom_{creator_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Apply customizations
        custom_template = self._apply_template_customizations(base_template, customizations)
        custom_template.template_id = custom_id
        custom_template.name = name or f"{base_template.name} (Custom)"
        custom_template.creator = creator_id
        
        # Store custom template
        self.templates[custom_id] = custom_template
        
        # Add to creator's custom templates
        prefs = self._get_creator_preferences(creator_id)
        prefs.preferences["custom_templates"].append(custom_id)
        
        logger.info(f"Created custom template {custom_id} for creator {creator_id}")
        
        return custom_template
    
    def apply_manual_override(
        self,
        creator_id: str,
        video_id: str,
        template_id: str,
        customizations: Dict = None
    ) -> VisualStyle:
        """Apply manual template override for a video."""
        
        if not self.config.allow_manual_override:
            raise ValueError("Manual override is disabled")
        
        template = self.get_template_by_id(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Create visual style with customizations
        visual_style = VisualStyle(
            template=template,
            video_id=video_id,
            applied_at=datetime.now().isoformat()
        )
        
        # Apply customizations if provided
        if customizations:
            if "colors" in customizations:
                visual_style.custom_colors = self._create_custom_color_scheme(customizations["colors"])
            
            if "position" in customizations:
                visual_style.custom_position = PositionStyle(customizations["position"])
            
            if "intensity" in customizations:
                visual_style.intensity_level = customizations["intensity"]
            
            if "speed" in customizations:
                visual_style.speed_multiplier = customizations["speed"]
        
        # Record the override choice for learning
        prefs = self._get_creator_preferences(creator_id)
        if template_id not in prefs.preferences["favorite_templates"]:
            # This might indicate a new preference
            pass
        
        logger.info(f"Applied manual override for creator {creator_id}, video {video_id}")
        
        return visual_style
    
    def update_creator_preferences(
        self,
        creator_id: str,
        preferences: Dict[str, Any]
    ):
        """Update creator preferences."""
        prefs = self._get_creator_preferences(creator_id)
        prefs.update_style_preferences(preferences)
        
        if self.config.save_creator_preferences:
            # In production, save to database
            pass
    
    def record_template_usage(
        self,
        creator_id: str,
        template_id: str,
        engagement_metrics: Dict[str, float]
    ):
        """Record template usage and engagement for learning."""
        prefs = self._get_creator_preferences(creator_id)
        prefs.record_engagement(template_id, engagement_metrics)
        
        # Update template popularity if configured
        if self.config.update_popularity_scores:
            template = self.get_template_by_id(template_id)
            if template:
                self._update_template_popularity(template, engagement_metrics)
    
    def get_template_analytics(self, template_id: str) -> Dict:
        """Get analytics for a template across all creators."""
        
        analytics = {
            "template_id": template_id,
            "total_uses": 0,
            "unique_creators": 0,
            "avg_engagement": 0.0,
            "trend": "stable"
        }
        
        template = self.get_template_by_id(template_id)
        if not template:
            return analytics
        
        # Collect usage data from all creators
        total_engagement = 0.0
        total_uses = 0
        unique_creators = set()
        
        for creator_id, prefs in self.creator_preferences.items():
            performance = prefs.get_template_performance(template_id)
            if performance:
                total_uses += 1
                unique_creators.add(creator_id)
                total_engagement += performance.get("avg_engagement_rate", 0.0)
        
        if total_uses > 0:
            analytics.update({
                "total_uses": total_uses,
                "unique_creators": len(unique_creators),
                "avg_engagement": total_engagement / total_uses
            })
        
        return analytics
    
    def _get_creator_preferences(self, creator_id: str) -> CreatorPreferences:
        """Get or create creator preferences."""
        if creator_id not in self.creator_preferences:
            self.creator_preferences[creator_id] = CreatorPreferences(creator_id)
        return self.creator_preferences[creator_id]
    
    def _score_template_for_creator(self, template: StyleTemplate, prefs: CreatorPreferences) -> float:
        """Score template based on creator preferences."""
        
        score = template.popularity_score  # Base score
        
        # Boost for favorites
        if template.template_id in prefs.preferences["favorite_templates"]:
            score += 0.3
        
        # Boost for preferred styles
        if any(style in template.name.lower() for style in prefs.preferences["preferred_styles"]):
            score += 0.2
        
        # Boost for animation intensity match
        template_intensity = "high" if template.entrance_animation.has_particles else "medium"
        if template_intensity == prefs.preferences["animation_intensity"]:
            score += 0.15
        
        # Boost for historical performance
        performance = prefs.get_template_performance(template.template_id)
        if performance.get("avg_engagement_rate", 0) > 0.6:
            score += 0.25
        
        return min(score, 1.0)
    
    def _apply_template_customizations(
        self,
        base_template: StyleTemplate,
        customizations: Dict
    ) -> StyleTemplate:
        """Apply customizations to create new template."""
        
        # Create a copy of the base template
        custom_template = StyleTemplate(
            template_id=base_template.template_id,
            name=base_template.name,
            description=base_template.description,
            creator=base_template.creator,
            popularity_score=base_template.popularity_score,
            target_platforms=base_template.target_platforms,
            content_types=base_template.content_types,
            target_demographics=base_template.target_demographics,
            color_scheme=base_template.color_scheme,
            typography=base_template.typography,
            position_style=base_template.position_style,
            entrance_animation=base_template.entrance_animation,
            exit_animation=base_template.exit_animation,
            image_size_ratio=base_template.image_size_ratio,
            border_radius=base_template.border_radius,
            border_width=base_template.border_width,
            has_background_blur=base_template.has_background_blur,
            background_opacity=base_template.background_opacity,
            has_ken_burns=base_template.has_ken_burns,
            ken_burns_intensity=base_template.ken_burns_intensity,
            has_parallax=base_template.has_parallax,
            parallax_speed=base_template.parallax_speed,
            has_pulse_to_beat=base_template.has_pulse_to_beat,
            has_emoji_reactions=base_template.has_emoji_reactions,
            emoji_set=base_template.emoji_set
        )
        
        # Apply customizations
        if "colors" in customizations:
            custom_template.color_scheme = self._create_custom_color_scheme(customizations["colors"])
        
        if "position" in customizations:
            custom_template.position_style = PositionStyle(customizations["position"])
        
        if "size" in customizations:
            custom_template.image_size_ratio = customizations["size"]
        
        if "border_radius" in customizations:
            custom_template.border_radius = customizations["border_radius"]
        
        if "effects" in customizations:
            effects = customizations["effects"]
            custom_template.has_ken_burns = effects.get("ken_burns", custom_template.has_ken_burns)
            custom_template.has_parallax = effects.get("parallax", custom_template.has_parallax)
            custom_template.has_pulse_to_beat = effects.get("pulse_to_beat", custom_template.has_pulse_to_beat)
        
        return custom_template
    
    def _create_custom_color_scheme(self, color_data: Dict) -> ColorScheme:
        """Create custom color scheme from user input."""
        return ColorScheme(
            name="Custom",
            primary=color_data.get("primary", "#FF0080"),
            secondary=color_data.get("secondary", "#00FFFF"),
            background=color_data.get("background", "#000000"),
            text_primary=color_data.get("text_primary", "#FFFFFF"),
            text_secondary=color_data.get("text_secondary", "#CCCCCC"),
            accent_glow=color_data.get("accent_glow", "#FF0080"),
            gradient_start=color_data.get("gradient_start", "#FF0080"),
            gradient_end=color_data.get("gradient_end", "#00FFFF"),
            highlight=color_data.get("highlight", "#FFFF00"),
            shadow=color_data.get("shadow", "#000000")
        )
    
    def _update_template_popularity(self, template: StyleTemplate, engagement_metrics: Dict[str, float]):
        """Update template popularity based on engagement."""
        
        # Simple popularity update based on engagement
        engagement_score = engagement_metrics.get("engagement_rate", 0.0)
        
        # Blend current popularity with recent performance
        blend_factor = 0.1  # How much to weight recent performance
        new_popularity = (
            template.popularity_score * (1 - blend_factor) +
            engagement_score * blend_factor
        )
        
        template.popularity_score = min(max(new_popularity, 0.0), 1.0) 