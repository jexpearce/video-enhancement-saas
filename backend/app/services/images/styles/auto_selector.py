"""
Automatic Style Selection - AI-Powered Template Matching.

Intelligently selects the most engaging style template based on:
- Content analysis (audio, video, entities)
- Platform optimization (TikTok, Instagram, etc.)
- Creator demographics and preferences  
- Trending patterns and engagement data
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime

from .models import (
    StyleTemplate,
    Platform,
    ContentType, 
    VIRAL_TEMPLATES,
    StyleConfig
)

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzes video content to determine optimal style."""
    
    def __init__(self):
        # Keywords for content type detection
        self.content_keywords = {
            ContentType.DANCE: [
                "dance", "dancing", "choreography", "moves", "tiktok dance",
                "viral dance", "challenge", "rhythm", "beat", "groove"
            ],
            ContentType.COMEDY: [
                "funny", "hilarious", "comedy", "joke", "laugh", "meme",
                "skit", "parody", "roast", "react", "reaction"
            ],
            ContentType.LIFESTYLE: [
                "lifestyle", "daily", "morning routine", "vlog", "aesthetic",
                "wellness", "self care", "style", "outfit", "home"
            ],
            ContentType.EDUCATION: [
                "learn", "tutorial", "how to", "explained", "facts", "tips",
                "guide", "lesson", "educational", "study", "academic"
            ],
            ContentType.BUSINESS: [
                "business", "entrepreneur", "startup", "marketing", "money",
                "success", "professional", "career", "finance", "investment"
            ],
            ContentType.TRAVEL: [
                "travel", "destination", "vacation", "adventure", "explore",
                "wanderlust", "trip", "journey", "beach", "city"
            ],
            ContentType.FOOD: [
                "food", "recipe", "cooking", "restaurant", "delicious",
                "taste", "foodie", "chef", "kitchen", "meal"
            ],
            ContentType.FITNESS: [
                "workout", "fitness", "gym", "exercise", "health", "training",
                "muscle", "cardio", "yoga", "pilates", "strength"
            ],
            ContentType.FASHION: [
                "fashion", "outfit", "style", "clothing", "designer", "trend",
                "lookbook", "accessories", "beauty", "makeup"
            ],
            ContentType.TECH: [
                "tech", "technology", "gadget", "app", "software", "coding",
                "ai", "innovation", "digital", "device", "review"
            ],
            ContentType.MUSIC: [
                "music", "song", "artist", "album", "concert", "performance",
                "singing", "musician", "band", "lyrics", "melody"
            ],
            ContentType.GAMING: [
                "gaming", "game", "gamer", "esports", "stream", "pc",
                "console", "gameplay", "review", "walkthrough"
            ]
        }
    
    def analyze_content_type(self, audio_transcript: str, video_title: str = "") -> ContentType:
        """Determine primary content type from audio and title."""
        
        # Combine text sources
        text = f"{audio_transcript} {video_title}".lower()
        
        # Score each content type
        scores = {}
        for content_type, keywords in self.content_keywords.items():
            score = 0
            for keyword in keywords:
                # Count keyword occurrences with position weighting
                occurrences = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                score += occurrences
                
                # Boost if keyword appears in title
                if keyword in video_title.lower():
                    score += 2
            
            scores[content_type] = score
        
        # Return highest scoring content type
        if scores:
            return max(scores, key=scores.get)
        
        return ContentType.LIFESTYLE  # Default fallback
    
    def detect_energy_level(self, audio_transcript: str) -> float:
        """Detect energy level from 0.0 (calm) to 1.0 (high energy)."""
        
        high_energy_words = [
            "amazing", "incredible", "awesome", "exciting", "crazy", "insane",
            "unbelievable", "wow", "omg", "fire", "lit", "epic", "wild"
        ]
        
        calm_words = [
            "peaceful", "calm", "relaxing", "gentle", "soft", "quiet",
            "meditation", "mindful", "serene", "tranquil"
        ]
        
        text = audio_transcript.lower()
        
        # Count energy indicators
        high_energy_score = sum(1 for word in high_energy_words if word in text)
        calm_score = sum(1 for word in calm_words if word in text)
        
        # Calculate energy level
        total_words = len(text.split())
        if total_words == 0:
            return 0.5
        
        energy_ratio = (high_energy_score - calm_score) / max(total_words / 50, 1)
        energy_level = max(0.0, min(1.0, 0.5 + energy_ratio))
        
        return energy_level
    
    def analyze_demographics(self, creator_age: int = None, creator_gender: str = None) -> List[str]:
        """Determine target demographics based on creator info."""
        
        demographics = []
        
        if creator_age:
            if creator_age <= 25:
                demographics.append("gen_z")
            elif creator_age <= 35:
                demographics.append("young_millennials")
            elif creator_age <= 45:
                demographics.append("millennials")
            else:
                demographics.append("gen_x")
        
        if creator_gender:
            if creator_gender.lower() in ["female", "woman"]:
                demographics.append("gen_z_female")
        
        # Default demographics if no info provided
        if not demographics:
            demographics = ["gen_z", "millennials"]
        
        return demographics


class TrendingAnalyzer:
    """Analyzes trending patterns to boost popular styles."""
    
    def __init__(self):
        # Mock trending data (in production, this would come from analytics)
        self.trending_features = {
            "neon_aesthetic": 0.95,
            "gradient_backgrounds": 0.88,
            "particle_effects": 0.82,
            "emoji_reactions": 0.76,
            "pulse_to_beat": 0.85,
            "glitch_effects": 0.79
        }
        
        self.platform_trends = {
            Platform.TIKTOK: {
                "preferred_positions": [
                    "floating", "center_overlay", "top_right"
                ],
                "popular_animations": [
                    "zoom_blast", "bounce_in", "glitch_pop"
                ],
                "trending_colors": ["neon", "gradient", "vibrant"]
            },
            Platform.INSTAGRAM: {
                "preferred_positions": [
                    "top_right", "side_panel", "floating"
                ],
                "popular_animations": [
                    "slide_up", "fade_in", "scale_up"
                ],
                "trending_colors": ["pastel", "gradient", "warm"]
            }
        }
    
    def get_trending_boost(self, template: StyleTemplate) -> float:
        """Calculate trending boost factor for template."""
        
        boost = 1.0
        
        # Check for trending features
        if template.entrance_animation.has_particles:
            boost += self.trending_features.get("particle_effects", 0) * 0.1
        
        if template.has_emoji_reactions:
            boost += self.trending_features.get("emoji_reactions", 0) * 0.1
        
        if template.has_pulse_to_beat:
            boost += self.trending_features.get("pulse_to_beat", 0) * 0.1
        
        if "neon" in template.color_scheme.name.lower():
            boost += self.trending_features.get("neon_aesthetic", 0) * 0.15
        
        if "gradient" in template.color_scheme.name.lower():
            boost += self.trending_features.get("gradient_backgrounds", 0) * 0.1
        
        return min(boost, 2.0)  # Cap at 2x boost


class AutoStyleSelector:
    """Intelligent automatic style selection system."""
    
    def __init__(self, config: StyleConfig = None):
        """Initialize with configuration."""
        self.config = config or StyleConfig()
        self.content_analyzer = ContentAnalyzer()
        self.trending_analyzer = TrendingAnalyzer()
        
        # Load available templates
        self.available_templates = list(VIRAL_TEMPLATES.values())
        
        logger.info(f"AutoStyleSelector initialized with {len(self.available_templates)} templates")
    
    async def select_optimal_style(
        self,
        audio_transcript: str,
        video_title: str = "",
        target_platform: Platform = Platform.TIKTOK,
        creator_age: int = None,
        creator_gender: str = None,
        creator_preferences: Dict = None
    ) -> Tuple[StyleTemplate, float]:
        """
        Select the most optimal style template.
        
        Returns:
            Tuple of (selected_template, confidence_score)
        """
        
        logger.info(f"Selecting optimal style for {target_platform.value} platform")
        
        # 1. Analyze content
        content_type = self.content_analyzer.analyze_content_type(audio_transcript, video_title)
        energy_level = self.content_analyzer.detect_energy_level(audio_transcript)
        demographics = self.content_analyzer.analyze_demographics(creator_age, creator_gender)
        
        logger.debug(f"Content analysis: type={content_type.value}, energy={energy_level:.2f}")
        
        # 2. Score all templates
        template_scores = []
        
        for template in self.available_templates:
            score = await self._score_template(
                template=template,
                content_type=content_type,
                energy_level=energy_level,
                target_platform=target_platform,
                demographics=demographics,
                creator_preferences=creator_preferences or {}
            )
            
            template_scores.append((template, score))
        
        # 3. Sort by score and select best
        template_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_template, best_score = template_scores[0]
        
        # 4. Check confidence threshold
        confidence = min(best_score, 1.0)
        
        if confidence < self.config.auto_selection_confidence_threshold:
            logger.warning(f"Low confidence ({confidence:.2f}) in auto-selection")
            # Fallback to highest popularity template
            best_template = max(self.available_templates, key=lambda t: t.popularity_score)
        
        logger.info(f"Selected template: {best_template.name} (confidence: {confidence:.2f})")
        
        return best_template, confidence
    
    async def _score_template(
        self,
        template: StyleTemplate,
        content_type: ContentType,
        energy_level: float,
        target_platform: Platform,
        demographics: List[str],
        creator_preferences: Dict
    ) -> float:
        """Score a template for the given context."""
        
        score = 0.0
        
        # 1. Content type matching (30% weight)
        if content_type in template.content_types:
            score += 0.3
        elif len(template.content_types) == 0:  # Universal template
            score += 0.15
        
        # 2. Platform optimization (25% weight)
        if target_platform in template.target_platforms:
            score += 0.25
        elif Platform.UNIVERSAL in template.target_platforms:
            score += 0.15
        
        # 3. Demographics matching (20% weight) 
        demographic_matches = sum(1 for demo in demographics if demo in template.target_demographics)
        if demographic_matches > 0:
            score += 0.2 * (demographic_matches / len(template.target_demographics))
        
        # 4. Energy level matching (15% weight)
        template_energy = self._estimate_template_energy(template)
        energy_diff = abs(energy_level - template_energy)
        score += 0.15 * (1.0 - energy_diff)
        
        # 5. Base popularity (10% weight)
        score += 0.1 * template.popularity_score
        
        # 6. Trending boost
        trending_boost = self.trending_analyzer.get_trending_boost(template)
        score *= trending_boost
        
        # 7. Creator preferences boost
        if creator_preferences:
            preference_boost = self._apply_creator_preferences(template, creator_preferences)
            score *= preference_boost
        
        return score
    
    def _estimate_template_energy(self, template: StyleTemplate) -> float:
        """Estimate energy level of a template."""
        
        energy = 0.5  # Base energy
        
        # Animation energy
        if template.entrance_animation.has_particles:
            energy += 0.2
        if template.entrance_animation.animation_type.value in ["zoom_blast", "bounce_in", "glitch_pop"]:
            energy += 0.15
        
        # Color energy
        if "neon" in template.color_scheme.name.lower():
            energy += 0.2
        if template.has_pulse_to_beat:
            energy += 0.15
        
        # Position energy
        if template.position_style.value in ["floating", "center_overlay"]:
            energy += 0.1
        
        return min(energy, 1.0)
    
    def _apply_creator_preferences(self, template: StyleTemplate, preferences: Dict) -> float:
        """Apply creator preferences as score multiplier."""
        
        multiplier = 1.0
        
        # Preferred style categories
        if "preferred_styles" in preferences:
            preferred = preferences["preferred_styles"]
            if any(style in template.name.lower() for style in preferred):
                multiplier += 0.3
        
        # Animation intensity preference
        if "animation_intensity" in preferences:
            pref_intensity = preferences["animation_intensity"]  # "low", "medium", "high"
            template_intensity = "high" if template.entrance_animation.has_particles else "medium"
            
            if pref_intensity == template_intensity:
                multiplier += 0.2
        
        # Color preferences
        if "color_preferences" in preferences:
            color_prefs = preferences["color_preferences"]
            if any(color in template.color_scheme.name.lower() for color in color_prefs):
                multiplier += 0.15
        
        return multiplier
    
    def get_style_recommendations(
        self,
        content_type: ContentType,
        target_platform: Platform,
        count: int = 3
    ) -> List[Tuple[StyleTemplate, str]]:
        """Get style recommendations with reasoning."""
        
        recommendations = []
        
        # Filter templates by criteria
        suitable_templates = [
            t for t in self.available_templates
            if content_type in t.content_types or len(t.content_types) == 0
        ]
        
        # Sort by engagement score
        suitable_templates.sort(key=lambda t: t.get_engagement_score(), reverse=True)
        
        for template in suitable_templates[:count]:
            # Generate reasoning
            reasons = []
            
            if content_type in template.content_types:
                reasons.append(f"Perfect for {content_type.value} content")
            
            if target_platform in template.target_platforms:
                reasons.append(f"Optimized for {target_platform.value}")
            
            if template.popularity_score > 0.8:
                reasons.append("Highly popular template")
            
            if template.has_pulse_to_beat:
                reasons.append("Syncs with music")
            
            if template.entrance_animation.has_particles:
                reasons.append("Eye-catching particle effects")
            
            reasoning = " • ".join(reasons)
            recommendations.append((template, reasoning))
        
        return recommendations
    
    async def analyze_template_performance(
        self,
        template_id: str,
        engagement_data: Dict
    ) -> Dict:
        """Analyze template performance and update popularity."""
        
        # Find template
        template = None
        for t in self.available_templates:
            if t.template_id == template_id:
                template = t
                break
        
        if not template:
            return {"error": "Template not found"}
        
        # Calculate performance metrics
        metrics = {
            "template_id": template_id,
            "current_popularity": template.popularity_score,
            "engagement_score": engagement_data.get("engagement_rate", 0.0),
            "view_duration": engagement_data.get("avg_view_duration", 0.0),
            "share_rate": engagement_data.get("share_rate", 0.0),
            "like_rate": engagement_data.get("like_rate", 0.0)
        }
        
        # Update popularity if enabled
        if self.config.update_popularity_scores:
            new_popularity = self._calculate_updated_popularity(template, engagement_data)
            template.popularity_score = new_popularity
            metrics["updated_popularity"] = new_popularity
        
        logger.info(f"Analyzed performance for template {template_id}")
        
        return metrics
    
    def _calculate_updated_popularity(self, template: StyleTemplate, engagement_data: Dict) -> float:
        """Calculate updated popularity score based on performance."""
        
        current_score = template.popularity_score
        
        # Weight recent performance
        recent_weight = 0.3
        historical_weight = 0.7
        
        # Calculate performance score from engagement metrics
        performance_score = (
            engagement_data.get("engagement_rate", 0.0) * 0.4 +
            engagement_data.get("share_rate", 0.0) * 0.3 +
            engagement_data.get("like_rate", 0.0) * 0.2 +
            min(engagement_data.get("avg_view_duration", 0.0) / 30.0, 1.0) * 0.1
        )
        
        # Blend with historical score
        new_score = (historical_weight * current_score + 
                    recent_weight * performance_score)
        
        return min(max(new_score, 0.0), 1.0) 