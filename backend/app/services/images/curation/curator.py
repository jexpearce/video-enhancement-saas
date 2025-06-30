"""
ML-Powered Image Curator - Phase 2 Days 23-24

Advanced image curation system using:
- CLIP model for image-text similarity scoring (with graceful fallbacks)
- Face detection for person entities (with error handling)
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
from sklearn.metrics.pairwise import cosine_similarity

from ..providers.base import ImageResult, ImageLicense
from ...nlp.entity_enricher import EnrichedEntity

logger = logging.getLogger(__name__)

# ML imports with error handling
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CLIP not available: {e}")
    CLIP_AVAILABLE = False

try:
    import cv2
    import mediapipe as mp
    CV2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenCV/MediaPipe not available: {e}")
    CV2_AVAILABLE = False

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
        """Calculate final weighted score for ranking with ultra-robust type safety."""
        try:
            # Ensure base_score is a float with multiple fallbacks
            if self.relevance_score is not None:
                try:
                    base_score = float(self.relevance_score)
                except (ValueError, TypeError):
                    base_score = 0.0
            else:
                base_score = 0.0
            
            # Add quality bonus - ensure it's a float with fallbacks
            quality_bonus_raw = self.curation_metadata.get('quality_bonus', 0.0)
            try:
                quality_bonus = float(quality_bonus_raw) if quality_bonus_raw is not None else 0.0
            except (ValueError, TypeError):
                quality_bonus = 0.0
            
            # Add face detection bonus for person entities - ensure it's a float with fallbacks
            face_bonus_raw = self.curation_metadata.get('face_bonus', 0.0)
            try:
                face_bonus = float(face_bonus_raw) if face_bonus_raw is not None else 0.0
            except (ValueError, TypeError):
                face_bonus = 0.0
            
            # Add CLIP similarity boost - ensure it's a float with fallbacks
            clip_bonus_raw = self.curation_metadata.get('clip_bonus', 0.0)
            try:
                clip_bonus = float(clip_bonus_raw) if clip_bonus_raw is not None else 0.0
            except (ValueError, TypeError):
                clip_bonus = 0.0
            
            result = base_score + quality_bonus + face_bonus + clip_bonus
            return min(1.0, max(0.0, result))  # Clamp between 0 and 1
            
        except Exception as e:
            # Ultimate fallback - return a valid float
            logger.warning(f"Error calculating final_score: {e}, returning default 0.5")
            return 0.5

@dataclass
class WordContext:
    """Context information for an emphasized word."""
    word: str
    surrounding_text: str
    timestamp: float
    emphasis_score: float

class ImageCurator:
    """ML-powered image curation system with robust error handling."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the image curator.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        
        # Model availability flags
        self.clip_available = False
        self.face_detection_available = False
        
        # Initialize models with error handling
        self._init_clip_model()
        self._init_face_detection()
        
        # Curation parameters
        self.max_images_per_entity = self.config.get('max_images_per_entity', 5)
        self.min_relevance_threshold = self.config.get('min_relevance_threshold', 0.3)
        self.diversity_penalty_threshold = self.config.get('diversity_penalty', 0.85)
        
        # Scoring weights (adjusted based on available models)
        self._init_scoring_weights()
        
        # Cache for model predictions
        self._clip_cache = {}
        self._face_cache = {}
        
    def _init_clip_model(self):
        """Initialize CLIP model with comprehensive error handling."""
        if not CLIP_AVAILABLE:
            logger.warning("CLIP dependencies not available. Using text-based fallback.")
            self.clip_model = None
            self.clip_processor = None
            return
            
        try:
            model_name = self.config.get('clip_model', "openai/clip-vit-base-patch32")
            logger.info(f"Loading CLIP model: {model_name}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model 
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            
            # Move to device if CUDA available
            if device == "cuda":
                self.clip_model = self.clip_model.to(device)
            
            # Set to evaluation mode
            self.clip_model.eval()
            self.device = device
            
            # Test the model with a simple query
            test_successful = self._test_clip_model()
            
            if test_successful:
                self.clip_available = True
                logger.info("CLIP model loaded and tested successfully")
            else:
                logger.error("CLIP model failed initial test")
                self._disable_clip_model()
                
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self._disable_clip_model()
            
    def _test_clip_model(self) -> bool:
        """Test CLIP model with a simple query."""
        try:
            if not self.clip_model or not self.clip_processor:
                return False
                
            # Simple test
            inputs = self.clip_processor(
                text=["a photo"], 
                images=None, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                _ = self.clip_model.get_text_features(**inputs)
                
            return True
            
        except Exception as e:
            logger.error(f"CLIP model test failed: {e}")
            return False
            
    def _disable_clip_model(self):
        """Disable CLIP model and clean up."""
        self.clip_model = None
        self.clip_processor = None
        self.clip_available = False
        logger.warning("CLIP model disabled. Using text-based similarity fallback.")
            
    def _init_face_detection(self):
        """Initialize face detection with error handling."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV/MediaPipe not available. Face detection disabled.")
            self.face_detection = None
            return
            
        try:
            # Initialize MediaPipe face detection
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for close-range, 1 for full-range
                min_detection_confidence=0.6
            )
            
            # Test face detection
            test_successful = self._test_face_detection()
            
            if test_successful:
                self.face_detection_available = True
                logger.info("Face detection initialized successfully")
            else:
                logger.error("Face detection failed initial test")
                self.face_detection = None
                
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_detection = None
            
    def _test_face_detection(self) -> bool:
        """Test face detection with a simple image."""
        try:
            if not self.face_detection:
                return False
                
            # Create a simple test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = self.face_detection.process(test_image)
            
            return True
            
        except Exception as e:
            logger.error(f"Face detection test failed: {e}")
            return False
            
    def _init_scoring_weights(self):
        """Initialize scoring weights based on available models."""
        if self.clip_available:
            # CLIP available - use ML-heavy weighting
            self.scoring_weights = {
                'clip_similarity': 0.4,
                'original_relevance': 0.3,
                'quality_score': 0.2,
                'entity_specific': 0.1
            }
        else:
            # No CLIP - rely more on text similarity and quality
            self.scoring_weights = {
                'clip_similarity': 0.0,  # Not available
                'original_relevance': 0.5,  # Increased weight
                'quality_score': 0.3,      # Increased weight
                'entity_specific': 0.2     # Increased weight
            }
            
        logger.info(f"Scoring weights configured: {self.scoring_weights}")

    async def curate_entity_images(
        self,
        entity: EnrichedEntity,
        candidate_images: List[ImageResult],
        word_context: Optional[WordContext] = None
    ) -> List[CuratedImage]:
        """
        Select and rank the best images for an entity using ML with fallbacks.
        
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
            
            # 2. Calculate relevance scores (with fallbacks)
            clip_scores = await self._calculate_clip_relevance_safe(entity, valid_images, word_context)
            
            # 3. Apply entity-type specific scoring (with fallbacks)
            entity_scores = await self._calculate_entity_specific_scores_safe(entity, valid_images)
            
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
                    'clip_bonus': max(0, clip_scores[i] - 0.7) * 0.1 if self.clip_available else 0.0,
                    'curation_timestamp': datetime.utcnow(),
                    'curation_version': '1.0',
                    'models_used': {
                        'clip': self.clip_available,
                        'face_detection': self.face_detection_available
                    }
                }
                
                curated_image = CuratedImage(
                    entity_id=getattr(entity, 'id', entity.text),
                    image=image,
                    relevance_score=final_score,
                    curation_metadata=curation_metadata,
                    clip_similarity=clip_scores[i]
                )
                
                curated_images.append(curated_image)
            
            # 5. Select diverse set of images
            selected_images = self._select_diverse_images(
                curated_images, 
                self.max_images_per_entity
            )
            
            logger.info(f"Curated {len(selected_images)} images for entity: {entity.text}")
            
            return selected_images
            
        except Exception as e:
            logger.error(f"Image curation failed for entity {entity.text}: {e}")
            
            # Return basic ranking based on original scores
            return self._fallback_curation(entity, candidate_images)

    def _fallback_curation(self, entity: EnrichedEntity, images: List[ImageResult]) -> List[CuratedImage]:
        """Fallback curation when ML models fail."""
        try:
            logger.warning(f"Using fallback curation for entity: {entity.text}")
            
            # Sort by original quality and relevance scores
            sorted_images = sorted(
                images, 
                key=lambda img: (img.quality_score + img.relevance_score) / 2, 
                reverse=True
            )
            
            curated_images = []
            for i, image in enumerate(sorted_images[:self.max_images_per_entity]):
                curation_metadata = {
                    'fallback_used': True,
                    'curation_timestamp': datetime.utcnow(),
                    'models_used': {'clip': False, 'face_detection': False}
                }
                
                curated_image = CuratedImage(
                    entity_id=getattr(entity, 'id', entity.text),
                    image=image,
                    relevance_score=image.relevance_score,
                    curation_metadata=curation_metadata
                )
                
                curated_images.append(curated_image)
            
            return curated_images
            
        except Exception as e:
            logger.error(f"Fallback curation failed: {e}")
            return []

    async def _calculate_clip_relevance_safe(
        self,
        entity: EnrichedEntity,
        images: List[ImageResult],
        word_context: Optional[WordContext] = None
    ) -> List[float]:
        """Calculate CLIP relevance with error handling and fallbacks."""
        
        if not self.clip_available:
            logger.debug("CLIP not available, using text similarity fallback")
            return self._text_similarity_fallback(entity, images, word_context)
            
        try:
            return await self._calculate_clip_relevance(entity, images, word_context)
            
        except Exception as e:
            logger.error(f"CLIP similarity calculation failed: {e}")
            logger.warning("Falling back to text similarity")
            
            # Disable CLIP for future requests
            self._disable_clip_model()
            
            return self._text_similarity_fallback(entity, images, word_context)

    def _text_similarity_fallback(
        self,
        entity: EnrichedEntity,
        images: List[ImageResult],
        word_context: Optional[WordContext] = None
    ) -> List[float]:
        """Text-based similarity fallback when CLIP is unavailable."""
        
        # Prepare search terms
        search_terms = [entity.text.lower()]
        if hasattr(entity, 'aliases') and entity.aliases:
            search_terms.extend([alias.lower() for alias in entity.aliases])
        if word_context:
            search_terms.append(word_context.word.lower())
            
        scores = []
        for image in images:
            # Calculate text similarity with image title and description
            image_text = f"{image.title} {getattr(image, 'description', '')}".lower()
            
            # Find best matching term
            max_similarity = 0.0
            for term in search_terms:
                similarity = self._simple_text_similarity(term, image_text)
                max_similarity = max(max_similarity, similarity)
                
            scores.append(max_similarity)
            
        return scores
        
    def _simple_text_similarity(self, term: str, text: str) -> float:
        """Simple text similarity calculation."""
        if term in text:
            return 0.8  # High similarity for exact match
        
        # Word overlap similarity
        term_words = set(term.split())
        text_words = set(text.split())
        
        if term_words and text_words:
            overlap = len(term_words.intersection(text_words))
            return min(0.7, overlap / len(term_words))
            
        return 0.1  # Base similarity

    async def _calculate_entity_specific_scores_safe(
        self,
        entity: EnrichedEntity,
        images: List[ImageResult]
    ) -> List[float]:
        """Calculate entity-specific scores with error handling."""
        
        try:
            return await self._calculate_entity_specific_scores(entity, images)
            
        except Exception as e:
            logger.error(f"Entity-specific scoring failed: {e}")
            
            # Return neutral scores
            return [0.5] * len(images)

    async def _filter_valid_images(self, images: List[ImageResult]) -> List[ImageResult]:
        """Filter images by basic quality and compliance criteria."""
        valid_images = []
        
        for image in images:
            try:
                # Check minimum dimensions
                if image.width < 400 or image.height < 300:
                    continue
                
                # Check aspect ratio (avoid extreme ratios)
                aspect_ratio = image.width / image.height
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    continue
                
                # Check license compliance
                if hasattr(image, 'license') and not self._is_license_compliant(getattr(image, 'license', 'unknown')):
                    continue
                
                # Check quality score threshold
                if image.quality_score < self.min_relevance_threshold:
                    continue
                
                valid_images.append(image)
                
            except Exception as e:
                logger.warning(f"Error filtering image {image.url}: {e}")
                continue
        
        return valid_images
    
    def _is_license_compliant(self, license: str) -> bool:
        """Check if image license is compliant for commercial use."""
        compliant_licenses = [
            "free", "cc0", "public_domain", "creative_commons_zero",
            "commercial_allowed", "unsplash", "pexels"
        ]
        
        return license.lower() in compliant_licenses

    async def _calculate_clip_relevance(
        self,
        entity: EnrichedEntity,
        images: List[ImageResult],
        word_context: Optional[WordContext] = None
    ) -> List[float]:
        """Calculate CLIP-based relevance scores."""
        
        if not self.clip_available:
            return [0.5] * len(images)  # Neutral scores
        
        try:
            # Prepare text queries for CLIP
            queries = self._prepare_clip_queries(entity, word_context)
            
            # Calculate similarities for each image
            similarities = []
            
            for image in images:
                try:
                    # Check cache first
                    cache_key = f"{image.url}:{':'.join(queries)}"
                    if cache_key in self._clip_cache:
                        similarities.append(self._clip_cache[cache_key])
                        continue
                    
                    # Download and process image
                    image_data = await self._download_image_safe(image.url)
                    if not image_data:
                        similarities.append(0.1)  # Low score for failed downloads
                        continue
                    
                    # Calculate CLIP similarity
                    similarity = await self._calculate_clip_similarity(image_data, queries)
                    
                    # Cache result
                    self._clip_cache[cache_key] = similarity
                    similarities.append(similarity)
                    
                except Exception as e:
                    logger.warning(f"CLIP similarity failed for image {image.url}: {e}")
                    similarities.append(0.1)
                    
            return similarities
            
        except Exception as e:
            logger.error(f"CLIP relevance calculation failed: {e}")
            return [0.5] * len(images)

    async def _download_image_safe(self, url: str) -> Optional[bytes]:
        """Safely download image with error handling."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.warning(f"Failed to download image {url}: {e}")
            return None

    async def _calculate_clip_similarity(self, image_data: bytes, queries: List[str]) -> float:
        """Calculate CLIP similarity with error handling."""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=queries, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
                # Calculate similarity
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Calculate similarity (take max across all queries)
                similarities = torch.matmul(text_embeds, image_embeds.T)
                max_similarity = similarities.max().item()
                
            return max(0.0, min(1.0, max_similarity))
            
        except Exception as e:
            logger.error(f"CLIP similarity calculation error: {e}")
            return 0.5  # Neutral score on error

    def _prepare_clip_queries(self, entity: EnrichedEntity, word_context: Optional[WordContext]) -> List[str]:
        """Prepare text queries for CLIP model."""
        queries = [f"a photo of {entity.text}"]
        
        # Add entity-specific queries
        if hasattr(entity, 'type'):
            if entity.type == 'PERSON':
                queries.extend([
                    f"portrait of {entity.text}",
                    f"{entity.text} person"
                ])
            elif entity.type in ['ORG', 'ORGANIZATION']:
                queries.extend([
                    f"{entity.text} logo",
                    f"{entity.text} company"
                ])
            elif entity.type in ['GPE', 'LOC', 'LOCATION']:
                queries.extend([
                    f"{entity.text} place",
                    f"{entity.text} landmark"
                ])
        
        # Add context-based queries
        if word_context:
            queries.append(f"{entity.text} {word_context.surrounding_text}")
        
        return queries[:3]  # Limit to 3 queries for efficiency

    async def _calculate_entity_specific_scores(
        self,
        entity: EnrichedEntity,
        images: List[ImageResult]
    ) -> List[float]:
        """Calculate entity-type specific scores."""
        
        if hasattr(entity, 'type'):
            if entity.type == 'PERSON':
                return await self._score_person_images_safe(images)
            elif entity.type in ['GPE', 'LOC', 'LOCATION']:
                return await self._score_location_images_safe(images)
            elif entity.type in ['ORG', 'ORGANIZATION']:
                return await self._score_organization_images_safe(images)
        
        # Default scoring
        return [0.5] * len(images)

    async def _score_person_images_safe(self, images: List[ImageResult]) -> List[float]:
        """Score person images with face detection (with fallbacks)."""
        
        if not self.face_detection_available:
            logger.debug("Face detection not available, using heuristic scoring")
            return self._score_person_images_heuristic(images)
        
        try:
            return await self._score_person_images(images)
        except Exception as e:
            logger.error(f"Face detection scoring failed: {e}")
            return self._score_person_images_heuristic(images)

    def _score_person_images_heuristic(self, images: List[ImageResult]) -> List[float]:
        """Heuristic scoring for person images when face detection fails."""
        scores = []
        for image in images:
            score = 0.5  # Base score
            
            # Boost for portrait orientation
            if hasattr(image, 'width') and hasattr(image, 'height'):
                aspect_ratio = image.width / image.height
                if 0.6 <= aspect_ratio <= 1.0:  # Portrait-ish
                    score += 0.2
            
            # Boost for person-related keywords in title
            if hasattr(image, 'title'):
                title_lower = image.title.lower()
                person_keywords = ['person', 'portrait', 'face', 'people', 'man', 'woman']
                if any(keyword in title_lower for keyword in person_keywords):
                    score += 0.3
            
            scores.append(min(1.0, score))
            
        return scores

    async def _score_location_images_safe(self, images: List[ImageResult]) -> List[float]:
        """Score location images safely."""
        try:
            return await self._score_location_images(images)
        except Exception as e:
            logger.error(f"Location scoring failed: {e}")
            return [0.5] * len(images)

    async def _score_organization_images_safe(self, images: List[ImageResult]) -> List[float]:
        """Score organization images safely."""
        try:
            return await self._score_organization_images(images)
        except Exception as e:
            logger.error(f"Organization scoring failed: {e}")
            return [0.5] * len(images)

    async def _score_person_images(self, images: List[ImageResult]) -> List[float]:
        """Score person images using face detection."""
        scores = []
        
        for image in images:
            try:
                score = 0.5  # Base score
                
                # Download and analyze image
                image_data = await self._download_image_safe(image.url)
                if not image_data:
                    scores.append(score)
                    continue
                
                # Face detection
                face_result = await self._detect_faces(image_data)
                if face_result and face_result.get('face_count', 0) > 0:
                    face_quality = face_result.get('face_quality_score', 0.5)
                    score += 0.3 * face_quality
                
                scores.append(min(1.0, score))
                
            except Exception as e:
                logger.warning(f"Person image scoring failed for {image.url}: {e}")
                scores.append(0.5)
        
        return scores

    async def _detect_faces(self, image_data: bytes) -> Optional[Dict]:
        """Detect faces in image with error handling."""
        try:
            if not self.face_detection:
                return None
            
            # Convert to OpenCV format
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_detection.process(image_rgb)
            
            if results.detections:
                face_count = len(results.detections)
                
                # Calculate quality score based on face size and confidence
                total_confidence = sum(detection.score[0] for detection in results.detections)
                avg_confidence = total_confidence / face_count
                
                return {
                    'face_count': face_count,
                    'face_quality_score': avg_confidence,
                    'faces': results.detections
                }
            
            return {'face_count': 0, 'face_quality_score': 0.0}
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None

    async def _score_location_images(self, images: List[ImageResult]) -> List[float]:
        """Score location images based on visual characteristics."""
        scores = []
        
        for image in images:
            score = 0.5  # Base score
            
            # Prefer landscape orientation for locations
            if hasattr(image, 'width') and hasattr(image, 'height'):
                aspect_ratio = image.width / image.height
                if aspect_ratio >= 1.2:  # Landscape
                    score += 0.2
            
            # Boost for location keywords
            if hasattr(image, 'title'):
                title_lower = image.title.lower()
                location_keywords = ['landscape', 'city', 'building', 'place', 'view', 'scenic']
                if any(keyword in title_lower for keyword in location_keywords):
                    score += 0.3
            
            scores.append(min(1.0, score))
        
        return scores

    async def _score_organization_images(self, images: List[ImageResult]) -> List[float]:
        """Score organization images."""
        scores = []
        
        for image in images:
            score = 0.5  # Base score
            
            # Boost for corporate keywords
            if hasattr(image, 'title'):
                title_lower = image.title.lower()
                org_keywords = ['logo', 'corporate', 'company', 'business', 'office', 'building']
                if any(keyword in title_lower for keyword in org_keywords):
                    score += 0.3
            
            scores.append(min(1.0, score))
        
        return scores

    def _combine_scores(self, clip_score: float, original_relevance: float,
                       quality_score: float, entity_specific: float) -> float:
        """Combine multiple scores using configured weights."""
        
        final_score = (
            clip_score * self.scoring_weights['clip_similarity'] +
            original_relevance * self.scoring_weights['original_relevance'] +
            quality_score * self.scoring_weights['quality_score'] +
            entity_specific * self.scoring_weights['entity_specific']
        )
        
        return min(1.0, max(0.0, final_score))

    def _select_diverse_images(self, curated_images: List[CuratedImage], count: int) -> List[CuratedImage]:
        """Select diverse set of high-quality images with safe sorting."""
        if len(curated_images) <= count:
            return self._safe_sort_by_final_score(curated_images)
        
        selected = []
        candidates = self._safe_sort_by_final_score(curated_images)
        
        while len(selected) < count and candidates:
            best_candidate = None
            best_score = -1
            
            for candidate in candidates:
                try:
                    # Calculate diversity score
                    diversity_score = self._calculate_diversity_score(candidate, selected)
                    
                    # Get final score safely
                    final_score = candidate.final_score
                    
                    # Ensure both are floats
                    diversity_score = float(diversity_score) if diversity_score is not None else 0.0
                    final_score = float(final_score) if final_score is not None else 0.0
                    
                    combined_score = final_score * 0.7 + diversity_score * 0.3
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate
                        
                except Exception as e:
                    logger.warning(f"Error calculating combined score for candidate: {e}")
                    continue
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _safe_sort_by_final_score(self, curated_images: List[CuratedImage]) -> List[CuratedImage]:
        """Safely sort images by final_score with error handling."""
        try:
            def safe_score_key(x: CuratedImage) -> float:
                try:
                    score = x.final_score
                    return float(score) if score is not None else 0.0
                except Exception as e:
                    logger.warning(f"Error getting final_score for image: {e}")
                    return 0.0
            
            return sorted(curated_images, key=safe_score_key, reverse=True)
        except Exception as e:
            logger.error(f"Error sorting images by final_score: {e}")
            return curated_images  # Return unsorted if sorting fails

    def _calculate_diversity_score(self, candidate: CuratedImage, selected: List[CuratedImage]) -> float:
        """Calculate diversity score for image selection."""
        if not selected:
            return 1.0
        
        # Simple diversity based on image dimensions and provider
        min_diversity = 1.0
        
        for selected_image in selected:
            # Aspect ratio diversity
            candidate_ratio = candidate.image.width / candidate.image.height
            selected_ratio = selected_image.image.width / selected_image.image.height
            ratio_diff = abs(candidate_ratio - selected_ratio)
            
            # Provider diversity
            provider_penalty = 0.1 if candidate.image.source == selected_image.image.source else 0.0
            
            diversity = max(0.0, 1.0 - ratio_diff - provider_penalty)
            min_diversity = min(min_diversity, diversity)
        
        return min_diversity

    def get_curation_stats(self, curated_images: List[CuratedImage]) -> Dict:
        """Get statistics about the curation process."""
        if not curated_images:
            return {'total_images': 0, 'models_used': {'clip': False, 'face_detection': False}}
        
        clip_scores = [img.clip_similarity for img in curated_images if img.clip_similarity is not None]
        relevance_scores = [img.relevance_score for img in curated_images]
        
        stats = {
            'total_images': len(curated_images),
            'average_relevance': sum(relevance_scores) / len(relevance_scores),
            'models_used': {
                'clip': self.clip_available,
                'face_detection': self.face_detection_available
            }
        }
        
        if clip_scores:
            stats['average_clip_similarity'] = sum(clip_scores) / len(clip_scores)
        
        return stats 