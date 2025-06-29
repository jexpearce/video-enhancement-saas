"""
ML-Powered Image Ranking Engine (Days 27-28).

This module implements sophisticated image ranking using machine learning,
feature extraction, contextual adjustments, and diversity penalties.
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from datetime import datetime, timezone

from .models import (
    RankedImage, 
    RankingFeatures, 
    RankingResult, 
    RankingConfig,
    VideoMetadata
)
from ..curation.models import CuratedImage, WordContext
from ...audio.entity_enricher import EnrichedEntity

logger = logging.getLogger(__name__)


class ImageRankingEngine:
    """
    ML-powered image ranking engine with contextual scoring.
    
    Features:
    - Comprehensive feature extraction (15+ features)
    - ML model for base scoring
    - Contextual adjustments based on video content
    - Diversity penalty to avoid visual repetition
    - Configurable weights and thresholds
    """
    
    def __init__(self, config: RankingConfig = None):
        """Initialize ranking engine with configuration."""
        self.config = config or RankingConfig()
        
        # Initialize ML model and scaler
        self.ranking_model = self._load_or_create_ranking_model()
        self.feature_scaler = StandardScaler()
        
        # Feature importance (learned from data)
        self.feature_importance = {}
        
        logger.info(f"Initialized ImageRankingEngine with model: {self.config.model_name}")
    
    async def rank_images_for_context(
        self,
        entity: EnrichedEntity,
        word_context: WordContext,
        candidate_images: List[CuratedImage],
        video_metadata: VideoMetadata
    ) -> RankingResult:
        """
        Rank images based on context, quality, and relevance.
        
        Args:
            entity: The entity being processed
            word_context: Context around the emphasized word
            candidate_images: Images to rank
            video_metadata: Video information for context
            
        Returns:
            RankingResult with ranked images and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Ranking {len(candidate_images)} images for entity: {entity.name}")
            
            if not candidate_images:
                return RankingResult(
                    entity=entity,
                    context=word_context,
                    ranked_images=[],
                    total_candidates=0,
                    processing_time_ms=0,
                    model_version=self.config.model_name,
                    feature_importance={},
                    score_distribution={},
                    diversity_metrics={}
                )
            
            # 1. Extract features for each image in parallel
            feature_tasks = [
                self._extract_ranking_features(img, entity, word_context, video_metadata)
                for img in candidate_images
            ]
            features_list = await asyncio.gather(*feature_tasks)
            
            # 2. Convert to feature matrix and normalize
            feature_matrix = np.array([f.to_feature_vector() for f in features_list])
            if len(feature_matrix) > 1:
                feature_matrix = self.feature_scaler.fit_transform(feature_matrix)
            
            # 3. Apply ML model for base scoring
            base_scores = self.ranking_model.predict(feature_matrix)
            
            # 4. Apply contextual adjustments and create ranked images
            ranked_images = []
            for i, (image, features, base_score) in enumerate(
                zip(candidate_images, features_list, base_scores)
            ):
                adjustments = await self._apply_contextual_adjustments(
                    image, features, entity, word_context, video_metadata
                )
                
                adjusted_score = base_score + sum(adjustments.values())
                
                ranked_image = RankedImage(
                    curated_image=image,
                    ranking_features=features,
                    base_score=float(base_score),
                    contextual_adjustments=adjustments,
                    diversity_penalty=0.0,
                    final_score=adjusted_score
                )
                ranked_images.append(ranked_image)
            
            # 5. Apply diversity penalty and sort
            ranked_images = self._apply_diversity_penalty(ranked_images)
            ranked_images.sort(key=lambda x: x.final_score, reverse=True)
            
            # 6. Create result with metrics
            processing_time = (time.time() - start_time) * 1000
            
            result = RankingResult(
                entity=entity,
                context=word_context,
                ranked_images=ranked_images,
                total_candidates=len(candidate_images),
                processing_time_ms=processing_time,
                model_version=self.config.model_name,
                feature_importance=self.feature_importance,
                score_distribution=self._calculate_score_statistics(ranked_images),
                diversity_metrics=self._calculate_diversity_metrics(ranked_images)
            )
            
            logger.info(f"Ranked {len(ranked_images)} images in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to rank images: {str(e)}")
            raise
    
    async def _extract_ranking_features(
        self,
        image: CuratedImage,
        entity: EnrichedEntity,
        word_context: WordContext,
        video_metadata: VideoMetadata
    ) -> RankingFeatures:
        """Extract comprehensive features for ranking."""
        
        try:
            # Use existing CLIP and quality scores from curation
            clip_similarity = image.relevance_score
            visual_quality = image.quality_score
            
            # Calculate additional semantic similarity
            semantic_similarity = self._calculate_semantic_similarity(
                image.source.title, entity.name
            )
            
            # Context matching
            context_match = self._calculate_context_match(
                image.source.description, word_context.sentence
            )
            
            # Entity type matching
            entity_type_match = self._check_entity_type_match(image, entity)
            
            # Resolution scoring
            resolution_score = self._calculate_resolution_score(
                image.source.width, image.source.height, video_metadata
            )
            
            # Aspect ratio scoring
            aspect_ratio_score = self._calculate_aspect_ratio_score(
                image.source.width, image.source.height, video_metadata.aspect_ratio
            )
            
            # Engagement metrics (if available)
            popularity_score = self._calculate_popularity_score(image.source.metadata)
            
            # Recency scoring
            recency_score = self._calculate_recency_score(image.source.metadata)
            
            # Visual analysis features
            color_vibrancy = self._analyze_color_vibrancy(image.source.metadata)
            composition_score = self._analyze_composition(image.source.metadata)
            text_space = self._has_text_space(image.source.metadata)
            
            # Face detection for person entities
            face_score = 0.0
            if entity.entity_type == "PERSON":
                face_score = image.source.metadata.get('face_quality_score', 0.0)
            
            # Visual signature for diversity
            visual_signature = self._compute_visual_signature(image)
            color_palette = self._extract_color_palette(image.source.metadata)
            visual_complexity = self._calculate_visual_complexity(image.source.metadata)
            
            return RankingFeatures(
                clip_similarity_score=clip_similarity,
                semantic_match_score=semantic_similarity,
                context_match_score=context_match,
                entity_type_match=entity_type_match,
                visual_quality_score=visual_quality,
                resolution_score=resolution_score,
                aspect_ratio_score=aspect_ratio_score,
                sharpness_score=image.source.metadata.get('sharpness_score', 0.5),
                popularity_score=popularity_score,
                recency_score=recency_score,
                color_vibrancy=color_vibrancy,
                composition_score=composition_score,
                text_space_available=text_space,
                face_detection_score=face_score,
                visual_signature=visual_signature,
                color_palette=color_palette,
                visual_complexity=visual_complexity
            )
            
        except Exception as e:
            logger.error(f"Failed to extract features for image {image.source.id}: {e}")
            # Return default features on error
            return self._get_default_features()
    
    async def _apply_contextual_adjustments(
        self,
        image: CuratedImage,
        features: RankingFeatures,
        entity: EnrichedEntity,
        word_context: WordContext,
        video_metadata: VideoMetadata
    ) -> Dict[str, float]:
        """Apply contextual adjustments to base score."""
        
        adjustments = {}
        
        # Entity type boost
        if features.entity_type_match:
            adjustments['entity_type_boost'] = self.config.entity_type_boost
        
        # Context relevance boost
        if features.context_match_score > 0.7:
            adjustments['context_relevance_boost'] = self.config.context_window_boost
        
        # Platform-specific adjustments
        if video_metadata.target_platform:
            platform_adjustment = self._get_platform_adjustment(
                image, video_metadata.target_platform
            )
            if platform_adjustment != 0:
                adjustments['platform_adjustment'] = platform_adjustment
        
        # Genre-specific adjustments
        if video_metadata.genre:
            genre_adjustment = self._get_genre_adjustment(image, video_metadata.genre)
            if genre_adjustment != 0:
                adjustments['genre_adjustment'] = genre_adjustment
        
        # Quality threshold penalty
        if features.visual_quality_score < self.config.min_quality_score:
            adjustments['quality_penalty'] = -0.2
        
        # Resolution penalty
        if features.resolution_score < self.config.min_resolution_score:
            adjustments['resolution_penalty'] = -0.15
        
        return adjustments
    
    def _apply_diversity_penalty(self, ranked_images: List[RankedImage]) -> List[RankedImage]:
        """Apply diversity penalty to avoid visual repetition."""
        
        if len(ranked_images) <= 1:
            return ranked_images
        
        # Track visual signatures we've seen
        seen_signatures = set()
        color_palette_counts = {}
        
        for i, ranked_image in enumerate(ranked_images):
            penalty = 0.0
            
            # Visual similarity penalty
            signature = ranked_image.ranking_features.visual_signature
            if signature in seen_signatures:
                penalty += self.config.diversity_penalty_strength
            seen_signatures.add(signature)
            
            # Color palette diversity penalty
            palette = tuple(ranked_image.ranking_features.color_palette)
            palette_count = color_palette_counts.get(palette, 0)
            if palette_count >= self.config.max_similar_colors:
                penalty += 0.1
            color_palette_counts[palette] = palette_count + 1
            
            # Apply penalty
            ranked_image.diversity_penalty = penalty
            ranked_image.final_score -= penalty
        
        return ranked_images
    
    def _load_or_create_ranking_model(self):
        """Load pre-trained model or create default one."""
        
        model_path = f"models/{self.config.model_name}.pkl"
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded ranking model from {model_path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load model {model_path}: {e}")
        
        # Create default Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train with synthetic data (in production, use real feedback data)
        self._train_default_model(model)
        
        logger.info("Created default Random Forest ranking model")
        return model
    
    def _train_default_model(self, model):
        """Train model with synthetic data (placeholder for real training)."""
        
        # Generate synthetic training data
        n_samples = 1000
        n_features = 15  # Number of features in RankingFeatures
        
        # Create feature matrix
        X = np.random.random((n_samples, n_features))
        
        # Create target scores based on feature importance
        # (In production, these would be real user feedback scores)
        weights = np.array([
            0.25,  # clip_similarity_score (most important)
            0.15,  # semantic_match_score
            0.10,  # context_match_score
            0.08,  # entity_type_match
            0.12,  # visual_quality_score
            0.05,  # resolution_score
            0.05,  # aspect_ratio_score
            0.04,  # sharpness_score
            0.06,  # popularity_score
            0.03,  # recency_score
            0.02,  # color_vibrancy
            0.02,  # composition_score
            0.01,  # text_space_available
            0.01,  # face_detection_score
            0.01   # visual_complexity
        ])
        
        y = np.dot(X, weights) + np.random.normal(0, 0.1, n_samples)
        y = np.clip(y, 0, 1)  # Ensure scores are in [0, 1]
        
        # Train model
        model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = dict(zip([
            'clip_similarity', 'semantic_match', 'context_match', 'entity_type_match',
            'visual_quality', 'resolution', 'aspect_ratio', 'sharpness',
            'popularity', 'recency', 'color_vibrancy', 'composition',
            'text_space', 'face_detection', 'visual_complexity'
        ], model.feature_importances_))
    
    def _calculate_semantic_similarity(self, image_title: str, entity_name: str) -> float:
        """Calculate semantic similarity between image title and entity."""
        if not image_title or not entity_name:
            return 0.0
        
        # Simple word overlap similarity (in production, use embeddings)
        title_words = set(image_title.lower().split())
        entity_words = set(entity_name.lower().split())
        
        if not title_words or not entity_words:
            return 0.0
        
        intersection = len(title_words.intersection(entity_words))
        union = len(title_words.union(entity_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_context_match(self, image_description: str, context_sentence: str) -> float:
        """Calculate how well image matches the context sentence."""
        if not image_description or not context_sentence:
            return 0.0
        
        # Simple word overlap (in production, use semantic similarity)
        desc_words = set(image_description.lower().split())
        context_words = set(context_sentence.lower().split())
        
        if not desc_words or not context_words:
            return 0.0
        
        intersection = len(desc_words.intersection(context_words))
        return min(intersection / 5.0, 1.0)  # Normalize to max 1.0
    
    def _check_entity_type_match(self, image: CuratedImage, entity: EnrichedEntity) -> bool:
        """Check if image type matches entity type."""
        
        # Get image categories from metadata
        image_categories = image.source.metadata.get('categories', [])
        if isinstance(image_categories, str):
            image_categories = [image_categories]
        
        # Check for type matches
        entity_type = entity.entity_type.lower()
        
        if entity_type == "person":
            return any(cat.lower() in ['person', 'people', 'portrait', 'face'] 
                      for cat in image_categories)
        elif entity_type == "location":
            return any(cat.lower() in ['place', 'location', 'city', 'country', 'landscape']
                      for cat in image_categories)
        elif entity_type == "organization":
            return any(cat.lower() in ['business', 'company', 'organization', 'building']
                      for cat in image_categories)
        
        return False
    
    def _calculate_resolution_score(self, width: int, height: int, video_metadata: VideoMetadata) -> float:
        """Score image resolution relative to video resolution."""
        if not width or not height:
            return 0.3
        
        # Calculate megapixels
        megapixels = (width * height) / 1_000_000
        
        # Score based on megapixels
        if megapixels >= 8:    # 4K+
            return 1.0
        elif megapixels >= 2:  # HD
            return 0.8
        elif megapixels >= 1:  # SD
            return 0.6
        else:                  # Low res
            return 0.3
    
    def _calculate_aspect_ratio_score(self, width: int, height: int, target_aspect: str) -> float:
        """Score how well image aspect ratio matches video."""
        if not width or not height:
            return 0.5
        
        image_ratio = width / height
        
        # Target ratios
        target_ratios = {
            "16:9": 16/9,
            "9:16": 9/16,
            "1:1": 1.0,
            "4:3": 4/3
        }
        
        target_ratio = target_ratios.get(target_aspect, 16/9)
        
        # Calculate difference
        ratio_diff = abs(image_ratio - target_ratio) / target_ratio
        
        # Score inversely proportional to difference
        return max(0.0, 1.0 - ratio_diff)
    
    def _calculate_popularity_score(self, metadata: Dict) -> float:
        """Calculate popularity score from engagement metrics."""
        likes = metadata.get('likes', 0)
        downloads = metadata.get('downloads', 0)
        views = metadata.get('views', 0)
        
        # Normalize using log scale
        score = 0.0
        if likes > 0:
            score += min(np.log1p(likes) / 10, 0.3)
        if downloads > 0:
            score += min(np.log1p(downloads) / 8, 0.4)
        if views > 0:
            score += min(np.log1p(views) / 15, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_recency_score(self, metadata: Dict) -> float:
        """Calculate recency score based on image age."""
        created_date = metadata.get('created_date')
        if not created_date:
            return 0.5  # Default for unknown date
        
        try:
            if isinstance(created_date, str):
                created_date = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            
            # Calculate age in days
            age_days = (datetime.now(timezone.utc) - created_date).days
            
            # Recent images get higher scores
            if age_days <= 30:
                return 1.0
            elif age_days <= 365:
                return 0.8
            elif age_days <= 1825:  # 5 years
                return 0.6
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _analyze_color_vibrancy(self, metadata: Dict) -> float:
        """Analyze color vibrancy from metadata."""
        return metadata.get('color_vibrancy', 0.5)
    
    def _analyze_composition(self, metadata: Dict) -> float:
        """Analyze composition quality from metadata."""
        return metadata.get('composition_score', 0.5)
    
    def _has_text_space(self, metadata: Dict) -> bool:
        """Check if image has space for text overlays."""
        return metadata.get('text_space_available', True)
    
    def _compute_visual_signature(self, image: CuratedImage) -> str:
        """Compute visual signature for similarity detection."""
        # In production, use perceptual hashing
        # For now, use a simple hash of image properties
        signature_data = f"{image.source.width}x{image.source.height}_{image.source.title}"
        return hashlib.md5(signature_data.encode()).hexdigest()[:16]
    
    def _extract_color_palette(self, metadata: Dict) -> List[str]:
        """Extract dominant color palette."""
        return metadata.get('dominant_colors', ['#808080'])  # Default gray
    
    def _calculate_visual_complexity(self, metadata: Dict) -> float:
        """Calculate visual complexity score."""
        return metadata.get('visual_complexity', 0.5)
    
    def _get_platform_adjustment(self, image: CuratedImage, platform: str) -> float:
        """Get platform-specific adjustment."""
        # Platform-specific preferences
        adjustments = {
            'tiktok': 0.1 if image.source.height > image.source.width else 0.0,  # Prefer vertical
            'instagram': 0.05,  # Slightly prefer square
            'youtube': 0.1 if image.source.width > image.source.height else 0.0  # Prefer horizontal
        }
        
        return adjustments.get(platform.lower(), 0.0)
    
    def _get_genre_adjustment(self, image: CuratedImage, genre: str) -> float:
        """Get genre-specific adjustment."""
        # Genre-specific preferences (simplified)
        if genre == 'news':
            # Prefer more formal, clear images
            return 0.05 if 'professional' in image.source.title.lower() else 0.0
        elif genre == 'entertainment':
            # Prefer more vibrant, dynamic images
            return 0.05 if any(word in image.source.title.lower() 
                             for word in ['colorful', 'vibrant', 'dynamic']) else 0.0
        
        return 0.0
    
    def _get_default_features(self) -> RankingFeatures:
        """Return default features when extraction fails."""
        return RankingFeatures(
            clip_similarity_score=0.0,
            semantic_match_score=0.0,
            context_match_score=0.0,
            entity_type_match=False,
            visual_quality_score=0.5,
            resolution_score=0.5,
            aspect_ratio_score=0.5,
            sharpness_score=0.5,
            popularity_score=0.0,
            recency_score=0.5,
            color_vibrancy=0.5,
            composition_score=0.5,
            text_space_available=True,
            face_detection_score=0.0,
            visual_signature="default",
            color_palette=["#808080"],
            visual_complexity=0.5
        )
    
    def _calculate_score_statistics(self, ranked_images: List[RankedImage]) -> Dict[str, float]:
        """Calculate score distribution statistics."""
        if not ranked_images:
            return {}
        
        scores = [img.final_score for img in ranked_images]
        return {
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'median_score': float(np.median(scores))
        }
    
    def _calculate_diversity_metrics(self, ranked_images: List[RankedImage]) -> Dict[str, float]:
        """Calculate diversity metrics."""
        if not ranked_images:
            return {}
        
        # Count unique visual signatures
        unique_signatures = len(set(img.ranking_features.visual_signature 
                                  for img in ranked_images))
        
        # Count unique color palettes
        unique_palettes = len(set(tuple(img.ranking_features.color_palette) 
                                for img in ranked_images))
        
        total_images = len(ranked_images)
        
        return {
            'signature_diversity': unique_signatures / total_images,
            'color_diversity': unique_palettes / total_images,
            'average_diversity_penalty': float(np.mean([img.diversity_penalty 
                                                       for img in ranked_images]))
        } 