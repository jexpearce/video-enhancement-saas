"""
Advanced Image Search and Quality Assessment Engine - Phase 2

Real API integration with multi-source providers:
- Unsplash for high-quality photography
- Pexels for diverse stock imagery  
- Pixabay for free-to-use content
- Wikimedia for educational/reference material

Features:
- Intelligent query optimization
- Multi-provider result aggregation
- Quality scoring and ranking
- Entity-specific search strategies
- Rate limiting and error handling
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

# Import real providers
from .providers.unsplash import UnsplashProvider
from .providers.base import ImageResult as ProviderImageResult, SearchRequest as ProviderSearchRequest, ImageLicense
from .providers.base import RateLimitError, ImageUnavailableError

logger = logging.getLogger(__name__)

@dataclass
class ImageResult:
    """Image search result with quality metrics."""
    url: str                          # Image URL
    thumbnail_url: str               # Thumbnail URL
    width: int                       # Image width
    height: int                      # Image height
    title: str                       # Image title/description
    photographer: str                # Photographer credit
    source: str                      # API source (unsplash, pexels, etc.)
    
    # Quality metrics
    quality_score: float = 0.0       # Overall quality score (0-1)
    relevance_score: float = 0.0     # Relevance to search query (0-1)
    resolution_score: float = 0.0    # Resolution quality (0-1)
    aesthetic_score: float = 0.0     # Aesthetic quality (0-1)
    
    # Metadata
    search_query: str = ""           # Original search query
    entity_name: str = ""            # Entity this image represents
    entity_type: str = ""            # Type of entity
    download_url: str = ""           # High-res download URL
    license: str = "free"            # License type
    
    # Timing for video sync
    suggested_start_time: float = 0.0  # When to show in video
    suggested_duration: float = 3.0    # How long to show

@dataclass
class SearchRequest:
    """Image search request configuration."""
    query: str                       # Search query
    entity_name: str                 # Entity name
    entity_type: str                 # Entity type
    max_results: int = 10           # Max results per API
    min_width: int = 800            # Minimum image width
    min_height: int = 600           # Minimum image height
    orientation: str = "landscape"   # landscape, portrait, squarish
    quality_threshold: float = 0.6   # Minimum quality score

class ImageSearcher:
    """Advanced image search engine with multi-source aggregation."""
    
    def __init__(self):
        """Initialize image searcher with providers and configurations."""
        
        # Initialize real providers
        self._init_providers()
        
        # Search configuration
        self.provider_weights = {
            'unsplash': 0.5,    # High-quality photography
            'pexels': 0.3,      # Diverse stock imagery
            'pixabay': 0.2      # Free content (fallback)
        }
        
        # Quality scoring weights
        self.quality_weights = {
            'resolution': 0.3,
            'relevance': 0.4,
            'aesthetic': 0.3
        }
        
        # Rate limiting tracking  
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'calls': [], 'max_calls': 50, 'window': 3600})
        
        # Entity-specific search strategies
        self._init_entity_search_configs()
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'total_images_found': 0,
            'average_quality_score': 0.0,
            'provider_performance': defaultdict(int)
        }
        
    def _init_providers(self):
        """Initialize real image providers with API keys."""
        
        # Initialize Unsplash provider
        unsplash_api_key = os.getenv('UNSPLASH_API_KEY')
        if unsplash_api_key:
            try:
                self.unsplash_provider = UnsplashProvider(unsplash_api_key)
                logger.info("Unsplash provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Unsplash provider: {e}")
                self.unsplash_provider = None
        else:
            logger.warning("UNSPLASH_API_KEY not found, Unsplash provider disabled")
            self.unsplash_provider = None
            
        # TODO: Initialize other providers when implemented
        # pexels_api_key = os.getenv('PEXELS_API_KEY')
        # pixabay_api_key = os.getenv('PIXABAY_API_KEY')
        
        # Track which providers are available
        self.available_providers = []
        if self.unsplash_provider:
            self.available_providers.append('unsplash')
            
        if not self.available_providers:
            logger.warning("No image providers available! Add API keys to environment.")

    def _init_entity_search_configs(self):
        """Initialize entity-specific search configurations."""
        
        self.entity_search_configs = {
            'PERSON': {
                'query_expansions': ['portrait', 'headshot', 'professional'],
                'min_resolution': (800, 600),
                'preferred_orientation': 'portrait',
                'quality_boost': 0.1
            },
            'ORG': {
                'query_expansions': ['logo', 'headquarters', 'corporate'],
                'min_resolution': (1024, 768),
                'preferred_orientation': 'landscape',
                'quality_boost': 0.05
            },
            'GPE': {
                'query_expansions': ['landmark', 'aerial', 'scenic'],
                'min_resolution': (1200, 800),
                'preferred_orientation': 'landscape',
                'quality_boost': 0.1
            },
            'LOC': {
                'query_expansions': ['destination', 'travel', 'scenic'],
                'min_resolution': (1200, 800),
                'preferred_orientation': 'landscape',
                'quality_boost': 0.1
            },
            'EVENT': {
                'query_expansions': ['ceremony', 'gathering', 'celebration'],
                'min_resolution': (1024, 768),
                'preferred_orientation': 'landscape',
                'quality_boost': 0.05
            },
            'MISC': {
                'query_expansions': ['concept', 'abstract', 'symbolic'],
                'min_resolution': (800, 600),
                'preferred_orientation': 'landscape',
                'quality_boost': 0.0
            }
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Close provider sessions if needed
        if self.unsplash_provider:
            try:
                await self.unsplash_provider.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.error(f"Error closing Unsplash provider: {e}")

    async def search_images(self, search_request: SearchRequest) -> List[ImageResult]:
        """
        Search for images across multiple providers with quality scoring.
        
        Args:
            search_request: SearchRequest configuration
            
        Returns:
            List of ImageResult objects, ranked by quality
        """
        
        try:
            start_time = time.time()
            self.search_stats['total_searches'] += 1
            
            logger.info(f"Starting image search for: {search_request.query}")
            
            # Optimize search query
            optimized_query = self._optimize_search_query(
                search_request.query, 
                search_request.entity_type
            )
            
            # Search across available providers
            all_results = []
            
            # Search Unsplash
            if 'unsplash' in self.available_providers:
                unsplash_results = await self._search_unsplash(optimized_query, search_request)
                all_results.extend(unsplash_results)
                self.search_stats['provider_performance']['unsplash'] += len(unsplash_results)
            
            # TODO: Add other providers when implemented
            # if 'pexels' in self.available_providers:
            #     pexels_results = await self._search_pexels(optimized_query, search_request)
            #     all_results.extend(pexels_results)
            
            # Score and rank all results
            ranked_results = self._score_and_rank_results(all_results, search_request)
            
            # Filter by quality threshold
            quality_results = [
                result for result in ranked_results 
                if result.quality_score >= search_request.quality_threshold
            ]
            
            # Update statistics
            search_time = time.time() - start_time
            self.search_stats['successful_searches'] += 1
            self.search_stats['total_images_found'] += len(quality_results)
            
            if quality_results:
                avg_quality = sum(r.quality_score for r in quality_results) / len(quality_results)
                self.search_stats['average_quality_score'] = avg_quality
            
            logger.info(
                f"Search completed in {search_time:.2f}s: "
                f"found {len(quality_results)} quality images"
            )
            
            return quality_results[:search_request.max_results]
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    def _optimize_search_query(self, query: str, entity_type: str) -> str:
        """Optimize search query based on entity type."""
        
        if entity_type in self.entity_search_configs:
            config = self.entity_search_configs[entity_type]
            expansions = config.get('query_expansions', [])
            
            # Add first expansion term for better results
            if expansions:
                query = f"{query} {expansions[0]}"
        
        return query.strip()
    
    async def _search_unsplash(self, query: str, request: SearchRequest) -> List[ImageResult]:
        """Search Unsplash API with real provider."""
        
        if not self.unsplash_provider:
            logger.warning("Unsplash provider not available")
            return []
            
        if not self._check_rate_limit('unsplash'):
            logger.warning("Unsplash rate limit exceeded")
            return []
        
        try:
            # Use real Unsplash provider
            provider_results = await self.unsplash_provider.search(query, request.max_results)
            
            # Convert provider results to our format
            results = []
            for provider_result in provider_results:
                result = self._convert_provider_result(provider_result, request)
                if result:
                    results.append(result)
            
            self._update_rate_limit('unsplash')
            logger.info(f"Unsplash search returned {len(results)} results")
            return results
            
        except RateLimitError as e:
            logger.warning(f"Unsplash rate limit hit: {e}")
            return []
        except Exception as e:
            logger.error(f"Unsplash search error: {e}")
            return []

    def _convert_provider_result(self, provider_result: ProviderImageResult, request: SearchRequest) -> Optional[ImageResult]:
        """Convert provider ImageResult to our ImageResult format."""
        
        try:
            result = ImageResult(
                url=provider_result.image_url,
                thumbnail_url=provider_result.thumbnail_url,
                width=provider_result.width,
                height=provider_result.height,
                title=provider_result.title or f"{request.query} - {provider_result.provider}",
                photographer=provider_result.author or "Unknown",
                source=provider_result.provider,
                search_query=request.query,
                entity_name=request.entity_name,
                entity_type=request.entity_type,
                download_url=provider_result.image_url,
                license="free" if provider_result.license in [ImageLicense.CREATIVE_COMMONS_ZERO, ImageLicense.PUBLIC_DOMAIN] else "attribution"
            )
            
            # Use provider's quality scores as base
            result.quality_score = provider_result.quality_score
            result.relevance_score = provider_result.relevance_score
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert provider result: {e}")
            return None

    def _score_and_rank_results(self, results: List[ImageResult], request: SearchRequest) -> List[ImageResult]:
        """Score and rank image results by quality."""
        
        entity_config = self.entity_search_configs.get(request.entity_type, {})
        
        for result in results:
            # Resolution score
            resolution_score = self._calculate_resolution_score(
                result.width, result.height, entity_config
            )
            result.resolution_score = resolution_score
            
            # Relevance score (use provider's score if available, otherwise calculate)
            if result.relevance_score == 0.0:
                relevance_score = self._calculate_relevance_score(
                    result.title, result.search_query, request.entity_type
                )
                result.relevance_score = relevance_score
            
            # Aesthetic score (use provider's score if available, otherwise mock)
            if result.aesthetic_score == 0.0:
                aesthetic_score = self._calculate_aesthetic_score(result, entity_config)
                result.aesthetic_score = aesthetic_score
            
            # Overall quality score (combine all factors)
            if result.quality_score == 0.0:
                result.quality_score = (
                    resolution_score * self.quality_weights['resolution'] +
                    result.relevance_score * self.quality_weights['relevance'] +
                    result.aesthetic_score * self.quality_weights['aesthetic']
                )
        
        # Sort by quality score
        results.sort(key=lambda r: r.quality_score, reverse=True)
        return results

    def _calculate_resolution_score(self, width: int, height: int, entity_config: Dict) -> float:
        """Calculate resolution quality score."""
        
        min_width, min_height = entity_config.get('min_resolution', (800, 600))
        
        # Score based on meeting minimum requirements
        width_score = min(1.0, width / min_width) if min_width > 0 else 1.0
        height_score = min(1.0, height / min_height) if min_height > 0 else 1.0
        
        # Bonus for high resolution
        if width >= 1920 and height >= 1080:
            return min(1.0, (width_score + height_score) / 2 + 0.2)
        
        return (width_score + height_score) / 2

    def _calculate_relevance_score(self, title: str, query: str, entity_type: str) -> float:
        """Calculate relevance score based on title matching."""
        
        title_lower = title.lower()
        query_lower = query.lower()
        
        # Exact query match
        if query_lower in title_lower:
            return 1.0
        
        # Word overlap scoring
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        
        if query_words and title_words:
            overlap = len(query_words.intersection(title_words))
            return min(1.0, overlap / len(query_words))
        
        return 0.3  # Base relevance score
    
    def _calculate_aesthetic_score(self, result: ImageResult, entity_config: Dict) -> float:
        """Calculate aesthetic quality score (simplified heuristic)."""
        
        # Aspect ratio preference
        aspect_ratio = result.width / result.height
        preferred_orientation = entity_config.get('preferred_orientation', 'landscape')
        
        score = 0.5  # Base score
        
        if preferred_orientation == 'portrait' and 0.6 <= aspect_ratio <= 0.8:
            score += 0.3
        elif preferred_orientation == 'landscape' and 1.3 <= aspect_ratio <= 1.8:
            score += 0.3
        elif preferred_orientation == 'square' and 0.9 <= aspect_ratio <= 1.1:
            score += 0.3
        
        # Quality boost for entity type
        quality_boost = entity_config.get('quality_boost', 0.0)
        score += quality_boost
        
        return min(1.0, score)

    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if we can make an API call without hitting rate limits."""
        
        now = time.time()
        rate_info = self.rate_limits[api_name]
        
        # Remove old calls outside the time window
        rate_info['calls'] = [
            call_time for call_time in rate_info['calls'] 
            if now - call_time < rate_info['window']
        ]
        
        return len(rate_info['calls']) < rate_info['max_calls']

    def _update_rate_limit(self, api_name: str):
        """Update rate limit tracking after successful API call."""
        self.rate_limits[api_name]['calls'].append(time.time())

    def get_search_statistics(self) -> Dict:
        """Get search performance statistics."""
        
        return {
            'total_searches': self.search_stats['total_searches'],
            'successful_searches': self.search_stats['successful_searches'],
            'success_rate': (
                self.search_stats['successful_searches'] / max(1, self.search_stats['total_searches'])
            ),
            'total_images_found': self.search_stats['total_images_found'],
            'average_quality_score': self.search_stats['average_quality_score'],
            'provider_performance': dict(self.search_stats['provider_performance']),
            'available_providers': self.available_providers
        }

    async def search_for_entity(self, entity, emphasized_time: float = 0.0) -> List[ImageResult]:
        """
        Search for images for a specific entity with timing context.
        
        Args:
            entity: Entity object with name, type, etc.
            emphasized_time: Time in video when entity is emphasized
            
        Returns:
            List of ranked ImageResult objects
        """
        
        search_request = SearchRequest(
            query=entity.text,
            entity_name=entity.text,
            entity_type=entity.type,
            max_results=5,
            quality_threshold=0.6
        )
        
        results = await self.search_images(search_request)
        
        # Set timing information
        for result in results:
            result.suggested_start_time = emphasized_time - 0.5  # Start slightly before
            result.suggested_duration = 3.0  # Show for 3 seconds
        
        return results 