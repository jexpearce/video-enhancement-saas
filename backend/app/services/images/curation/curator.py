"""
ML-Powered Image Curator - Phase 2 Days 23-24

Advanced image curation system using:
- CLIP model for image-text similarity scoring
- Face detection for person entities
- Computer vision quality assessment
- Context-aware relevance scoring
"""

import torch
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import io
import hashlib

import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import cv2
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

from ..providers.base import ImageResult, ImageLicense
from ...nlp.entity_enricher import EnrichedEntity

logger = logging.getLogger(__name__)

@dataclass
class CuratedImage:
    """An image that has been curated using ML and quality assessment."""
    entity_id: str
    image: ImageResult
    relevance_score: float
    curation_metadata: Dict
    face_detection_result: Optional[Dict] = None
    quality_assessment: Optional[Dict] = None
    clip_similarity: Optional[float] = None
    
    @property
    def final_score(self) -> float:
        """Calculate final weighted score for ranking."""
        base_score = self.relevance_score
        
        # Add quality bonus
        quality_bonus = self.curation_metadata.get('quality_bonus', 0.0)
        
        # Add face detection bonus for person entities
        face_bonus = self.curation_metadata.get('face_bonus', 0.0)
        
        # Add CLIP similarity boost
        clip_bonus = self.curation_metadata.get('clip_bonus', 0.0)
        
        return min(1.0, base_score + quality_bonus + face_bonus + clip_bonus)

@dataclass
class WordContext:
    """Context information for an emphasized word."""
    word: str
    surrounding_text: str
    timestamp: float
    emphasis_score: float

class ImageCurator:
    """ML-powered image curation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the image curator.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        
        # Initialize CLIP model for image-text similarity
        self._init_clip_model()
        
        # Initialize face detection
        self._init_face_detection()
        
        # Curation parameters
        self.max_images_per_entity = self.config.get('max_images_per_entity', 5)
        self.min_relevance_threshold = self.config.get('min_relevance_threshold', 0.3)
        self.diversity_penalty_threshold = self.config.get('diversity_penalty', 0.85)
        
        # Scoring weights
        self.scoring_weights = {
            'clip_similarity': 0.4,
            'original_relevance': 0.3,
            'quality_score': 0.2,
            'entity_specific': 0.1
        }
        
    def _init_clip_model(self):
        """Initialize CLIP model for image-text similarity."""
        try:
            model_name = self.config.get('clip_model', "openai/clip-vit-base-patch32")
            logger.info(f"Loading CLIP model: {model_name}")
            
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.clip_model.eval()
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
            
    def _init_face_detection(self):
        """Initialize face detection for person entities."""
        try:
            # Initialize MediaPipe face detection
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for close-range, 1 for full-range
                min_detection_confidence=0.6
            )
            
            logger.info("Face detection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_detection = None
            
    async def curate_entity_images(
        self,
        entity: EnrichedEntity,
        candidate_images: List[ImageResult],
        word_context: Optional[WordContext] = None
    ) -> List[CuratedImage]:
        """
        Select and rank the best images for an entity using ML.
        
        Args:
            entity: The entity to find images for
            candidate_images: List of candidate images from providers
            word_context: Context about how the entity was emphasized
            
        Returns:
            List of curated images ranked by relevance
        """
        try:
            logger.info(f"Curating images for entity: {entity.text}")
            
            if not candidate_images:
                logger.warning(f"No candidate images provided for entity: {entity.text}")
                return []
            
            # 1. Filter images by basic criteria
            valid_images = await self._filter_valid_images(candidate_images)
            logger.info(f"Valid images after filtering: {len(valid_images)}")
            
            if not valid_images:
                return []
            
            # 2. Calculate CLIP-based relevance scores
            clip_scores = await self._calculate_clip_relevance(entity, valid_images, word_context)
            
            # 3. Apply entity-type specific scoring
            entity_scores = await self._calculate_entity_specific_scores(entity, valid_images)
            
            # 4. Combine all scores
            curated_images = []
            for i, image in enumerate(valid_images):
                # Calculate weighted final score
                final_score = self._combine_scores(
                    clip_score=clip_scores[i],
                    original_relevance=image.relevance_score,
                    quality_score=image.quality_score,
                    entity_specific=entity_scores[i]
                )
                
                # Create curation metadata
                curation_metadata = {
                    'clip_score': clip_scores[i],
                    'entity_specific_score': entity_scores[i],
                    'quality_bonus': max(0, image.quality_score - 0.5) * 0.1,
                    'clip_bonus': max(0, clip_scores[i] - 0.7) * 0.1,
                    'curation_timestamp': datetime.utcnow(),
                    'curation_version': '1.0'
                }
                
                curated_image = CuratedImage(
                    entity_id=getattr(entity, 'id', entity.text),
                    image=image,
                    relevance_score=final_score,
                    curation_metadata=curation_metadata,
                    clip_similarity=clip_scores[i]
                )
                
                curated_images.append(curated_image)
            
            # 5. Apply diversity penalty and select top images
            diverse_images = self._select_diverse_images(curated_images, self.max_images_per_entity)
            
            # 6. Sort by final score
            diverse_images.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.info(f"Successfully curated {len(diverse_images)} images for {entity.text}")
            return diverse_images
            
        except Exception as e:
            logger.error(f"Error curating images for entity {entity.text}: {e}")
            return []
    
    async def _filter_valid_images(self, images: List[ImageResult]) -> List[ImageResult]:
        """Filter images by basic quality and legal criteria."""
        valid_images = []
        
        for image in images:
            # Check license compliance
            if not self._is_license_compliant(image.license):
                continue
                
            # Check basic quality thresholds
            if image.quality_score < self.min_relevance_threshold:
                continue
                
            # Check image dimensions (minimum viable size)
            if image.width < 400 or image.height < 300:
                continue
                
            # Check if image URL is accessible (basic validation)
            if not image.image_url or not image.thumbnail_url:
                continue
                
            valid_images.append(image)
            
        return valid_images
    
    def _is_license_compliant(self, license: ImageLicense) -> bool:
        """Check if image license is compliant for commercial use."""
        compliant_licenses = {
            ImageLicense.CREATIVE_COMMONS_ZERO,
            ImageLicense.CREATIVE_COMMONS_BY,
            ImageLicense.PUBLIC_DOMAIN,
            ImageLicense.COMMERCIAL_ALLOWED
        }
        return license in compliant_licenses
    
    async def _calculate_clip_relevance(
        self,
        entity: EnrichedEntity,
        images: List[ImageResult],
        word_context: Optional[WordContext] = None
    ) -> List[float]:
        """Calculate relevance scores using CLIP model."""
        if not self.clip_model or not self.clip_processor:
            logger.warning("CLIP model not available, using fallback scores")
            return [image.relevance_score for image in images]
        
        try:
            # Prepare text queries for CLIP
            text_queries = self._prepare_clip_queries(entity, word_context)
            
            # Process images and calculate similarities
            similarities = []
            
            for image in images:
                try:
                    # Download and process image
                    image_data = await self._download_image(image.thumbnail_url)
                    pil_image = Image.open(io.BytesIO(image_data))
                    
                    # Calculate similarity with each text query
                    image_similarities = []
                    
                    for query in text_queries:
                        # Process inputs
                        inputs = self.clip_processor(
                            text=[query],
                            images=[pil_image],
                            return_tensors="pt",
                            padding=True
                        )
                        
                        # Calculate similarity
                        with torch.no_grad():
                            outputs = self.clip_model(**inputs)
                            similarity = torch.cosine_similarity(
                                outputs.text_embeds,
                                outputs.image_embeds
                            ).item()
                            
                        image_similarities.append(similarity)
                    
                    # Use maximum similarity across queries
                    max_similarity = max(image_similarities)
                    similarities.append(max_similarity)
                    
                except Exception as e:
                    logger.warning(f"Failed to process image {image.image_url}: {e}")
                    similarities.append(0.0)
            
            # Normalize similarities to [0, 1] range
            if similarities:
                min_sim = min(similarities)
                max_sim = max(similarities)
                if max_sim > min_sim:
                    similarities = [(s - min_sim) / (max_sim - min_sim) for s in similarities]
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating CLIP relevance: {e}")
            return [image.relevance_score for image in images]
    
    def _prepare_clip_queries(self, entity: EnrichedEntity, word_context: Optional[WordContext]) -> List[str]:
        """Prepare text queries for CLIP similarity calculation."""
        queries = []
        
        # Base entity text
        queries.append(entity.text)
        
        # Canonical name if different
        if entity.canonical_name and entity.canonical_name != entity.text:
            queries.append(entity.canonical_name)
        
        # Entity-type specific queries
        if entity.entity_type == "PERSON":
            queries.extend([
                f"{entity.text} portrait",
                f"{entity.text} headshot",
                f"{entity.text} professional photo"
            ])
        elif entity.entity_type in ["LOCATION", "GPE", "LOC"]:
            queries.extend([
                f"{entity.text} landmark",
                f"{entity.text} skyline",
                f"{entity.text} aerial view"
            ])
        elif entity.entity_type == "ORG":
            queries.extend([
                f"{entity.text} logo",
                f"{entity.text} building",
                f"{entity.text} headquarters"
            ])
        
        # Add context from surrounding words if available
        if word_context and word_context.surrounding_text:
            context_query = f"{entity.text} {word_context.surrounding_text}"
            queries.append(context_query)
        
        return queries[:5]  # Limit to 5 queries for efficiency
    
    async def _calculate_entity_specific_scores(
        self,
        entity: EnrichedEntity,
        images: List[ImageResult]
    ) -> List[float]:
        """Calculate entity-type specific scores."""
        if entity.entity_type == "PERSON":
            return await self._score_person_images(images)
        elif entity.entity_type in ["LOCATION", "GPE", "LOC"]:
            return await self._score_location_images(images)
        elif entity.entity_type == "ORG":
            return await self._score_organization_images(images)
        else:
            return [0.5 for _ in images]  # Neutral score for other types
    
    async def _score_person_images(self, images: List[ImageResult]) -> List[float]:
        """Score images for person entities using face detection."""
        scores = []
        
        for image in images:
            score = 0.5  # Base score
            
            try:
                if self.face_detection:
                    # Download and analyze image
                    image_data = await self._download_image(image.thumbnail_url)
                    
                    # Convert to cv2 format
                    nparr = np.frombuffer(image_data, np.uint8)
                    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    results = self.face_detection.process(rgb_image)
                    
                    if results.detections:
                        num_faces = len(results.detections)
                        
                        if num_faces == 1:
                            # Single person portrait - excellent
                            score = 1.0
                            
                            # Check face quality/size
                            detection = results.detections[0]
                            bbox = detection.location_data.relative_bounding_box
                            face_area = bbox.width * bbox.height
                            
                            if face_area > 0.1:  # Face takes up >10% of image
                                score = 1.0
                            elif face_area > 0.05:  # Face takes up >5% of image
                                score = 0.8
                            else:
                                score = 0.6
                                
                        elif num_faces <= 3:
                            # Small group - good
                            score = 0.7
                        else:
                            # Large group - less ideal
                            score = 0.4
                    else:
                        # No faces detected - poor for person entity
                        score = 0.2
                        
            except Exception as e:
                logger.warning(f"Face detection failed for image: {e}")
                score = 0.3  # Fallback score
                
            scores.append(score)
            
        return scores
    
    async def _score_location_images(self, images: List[ImageResult]) -> List[float]:
        """Score images for location entities."""
        scores = []
        
        for image in images:
            score = 0.5  # Base score
            
            # Analyze image title and description for location indicators
            title_lower = (image.title or "").lower()
            desc_lower = (image.description or "").lower()
            
            location_keywords = [
                'landmark', 'skyline', 'aerial', 'view', 'panorama',
                'cityscape', 'landscape', 'architecture', 'building',
                'monument', 'statue', 'bridge', 'tower'
            ]
            
            keyword_count = sum(1 for keyword in location_keywords 
                              if keyword in title_lower or keyword in desc_lower)
            
            # Boost score based on relevant keywords
            score += min(0.4, keyword_count * 0.1)
            
            # Prefer landscape orientation for locations
            aspect_ratio = image.width / image.height if image.height > 0 else 1
            if 1.5 <= aspect_ratio <= 2.5:  # Landscape orientation
                score += 0.1
            
            scores.append(min(1.0, score))
            
        return scores
    
    async def _score_organization_images(self, images: List[ImageResult]) -> List[float]:
        """Score images for organization entities."""
        scores = []
        
        for image in images:
            score = 0.5  # Base score
            
            # Analyze for organization-relevant content
            title_lower = (image.title or "").lower()
            desc_lower = (image.description or "").lower()
            
            org_keywords = [
                'logo', 'building', 'headquarters', 'office',
                'corporate', 'company', 'brand', 'business'
            ]
            
            keyword_count = sum(1 for keyword in org_keywords 
                              if keyword in title_lower or keyword in desc_lower)
            
            score += min(0.3, keyword_count * 0.1)
            
            # Prefer square aspect ratios for logos
            aspect_ratio = image.width / image.height if image.height > 0 else 1
            if 0.8 <= aspect_ratio <= 1.2:  # Square-ish
                score += 0.1
            
            scores.append(min(1.0, score))
            
        return scores
    
    def _combine_scores(self, clip_score: float, original_relevance: float,
                       quality_score: float, entity_specific: float) -> float:
        """Combine different scores using weighted average."""
        weights = self.scoring_weights
        
        combined = (
            weights['clip_similarity'] * clip_score +
            weights['original_relevance'] * original_relevance +
            weights['quality_score'] * quality_score +
            weights['entity_specific'] * entity_specific
        )
        
        return min(1.0, combined)
    
    def _select_diverse_images(self, curated_images: List[CuratedImage], count: int) -> List[CuratedImage]:
        """Select diverse images to avoid visual repetition."""
        if len(curated_images) <= count:
            return curated_images
        
        selected = []
        remaining = curated_images.copy()
        
        # Always select the highest scoring image first
        remaining.sort(key=lambda x: x.final_score, reverse=True)
        selected.append(remaining.pop(0))
        
        # Select remaining images with diversity consideration
        while len(selected) < count and remaining:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                # Calculate diversity score (lower = more similar)
                diversity_score = self._calculate_diversity_score(candidate, selected)
                
                # Combine relevance and diversity
                combined_score = candidate.final_score * 0.7 + diversity_score * 0.3
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _calculate_diversity_score(self, candidate: CuratedImage, selected: List[CuratedImage]) -> float:
        """Calculate how visually diverse a candidate is from selected images."""
        if not selected:
            return 1.0
        
        # Simple diversity calculation based on image properties
        min_diversity = 1.0
        
        for selected_image in selected:
            # Compare aspect ratios
            candidate_ratio = candidate.image.width / candidate.image.height
            selected_ratio = selected_image.image.width / selected_image.image.height
            ratio_diff = abs(candidate_ratio - selected_ratio)
            
            # Compare titles/descriptions for content similarity
            title_similarity = self._text_similarity(
                candidate.image.title or "",
                selected_image.image.title or ""
            )
            
            # Calculate overall similarity (lower = more diverse)
            similarity = (
                max(0, 1 - ratio_diff) * 0.3 +
                title_similarity * 0.7
            )
            
            diversity = 1 - similarity
            min_diversity = min(min_diversity, diversity)
        
        return min_diversity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _download_image(self, url: str) -> bytes:
        """Download image data from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise
    
    def get_curation_stats(self, curated_images: List[CuratedImage]) -> Dict:
        """Get statistics about the curation process."""
        if not curated_images:
            return {}
        
        scores = [img.final_score for img in curated_images]
        clip_scores = [img.clip_similarity for img in curated_images if img.clip_similarity]
        
        return {
            'total_images': len(curated_images),
            'avg_final_score': np.mean(scores),
            'max_final_score': max(scores),
            'min_final_score': min(scores),
            'avg_clip_similarity': np.mean(clip_scores) if clip_scores else 0.0,
            'images_with_faces': sum(1 for img in curated_images 
                                   if img.face_detection_result and 
                                   img.face_detection_result.get('faces_detected', 0) > 0),
            'curation_version': '1.0'
        } 