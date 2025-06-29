"""
Image Search Service

Provides comprehensive image search capabilities across multiple stock image APIs:
- Unsplash (free, high-quality photos)
- Pexels (free stock photos)
- Pixabay (free images, photos, vectors)
- Getty Images (premium stock - future)

Optimized for finding images that match our entity recognition results.
"""

import logging
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    """Advanced image search service with multiple API integrations."""
    
    def __init__(self):
        """Initialize the image searcher."""
        
        # API configurations
        self.api_configs = {
            'unsplash': {
                'base_url': 'https://api.unsplash.com',
                'access_key': 'demo',  # Replace with real key
                'rate_limit': 50,      # Requests per hour
                'enabled': True
            },
            'pexels': {
                'base_url': 'https://api.pexels.com/v1',
                'api_key': 'demo',     # Replace with real key
                'rate_limit': 200,     # Requests per hour
                'enabled': True
            },
            'pixabay': {
                'base_url': 'https://pixabay.com/api',
                'api_key': 'demo',     # Replace with real key
                'rate_limit': 100,     # Requests per hour
                'enabled': True
            }
        }
        
        # Image quality assessment weights
        self.quality_weights = {
            'resolution': 0.3,      # Resolution quality
            'aesthetic': 0.3,       # Visual aesthetics
            'relevance': 0.4        # Search relevance
        }
        
        # Entity-specific search configurations
        self._init_entity_search_configs()
        
        # Rate limiting tracking
        self.rate_limits = {api: {'count': 0, 'reset_time': time.time() + 3600} 
                           for api in self.api_configs.keys()}
        
        # HTTP session for efficient requests
        self.session = None
        
    def _init_entity_search_configs(self):
        """Initialize entity-specific search configurations."""
        
        self.entity_search_configs = {
            'PERSON': {
                'preferred_orientation': 'portrait',
                'quality_modifiers': ['portrait', 'professional', 'headshot'],
                'exclude_terms': ['cartoon', 'drawing', 'illustration'],
                'min_resolution': (600, 800),  # width, height
                'aesthetic_weight': 0.4
            },
            'ORGANIZATION': {
                'preferred_orientation': 'landscape', 
                'quality_modifiers': ['logo', 'building', 'headquarters', 'corporate'],
                'exclude_terms': ['person', 'people', 'crowd'],
                'min_resolution': (800, 600),
                'aesthetic_weight': 0.3
            },
            'LOCATION': {
                'preferred_orientation': 'landscape',
                'quality_modifiers': ['landmark', 'aerial', 'skyline', 'scenic'],
                'exclude_terms': ['people', 'person', 'crowd'],
                'min_resolution': (1200, 800),
                'aesthetic_weight': 0.5
            },
            'PRODUCT': {
                'preferred_orientation': 'squarish',
                'quality_modifiers': ['product', 'professional', 'clean', 'isolated'],
                'exclude_terms': ['person', 'people', 'lifestyle'],
                'min_resolution': (800, 800),
                'aesthetic_weight': 0.4
            },
            'BRAND': {
                'preferred_orientation': 'squarish',
                'quality_modifiers': ['logo', 'brand', 'official', 'clean'],
                'exclude_terms': ['person', 'people', 'unofficial'],
                'min_resolution': (600, 600),
                'aesthetic_weight': 0.3
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def search_images(self, search_request: SearchRequest) -> List[ImageResult]:
        """
        Search for images across multiple APIs.
        
        Args:
            search_request: Search configuration
            
        Returns:
            List of image results sorted by quality
        """
        try:
            logger.info(f"Searching images for: {search_request.query}")
            
            # Optimize search query for entity type
            optimized_query = self._optimize_search_query(
                search_request.query, 
                search_request.entity_type
            )
            
            # Search across enabled APIs in parallel
            search_tasks = []
            
            if self.api_configs['unsplash']['enabled']:
                search_tasks.append(self._search_unsplash(optimized_query, search_request))
            
            if self.api_configs['pexels']['enabled']:
                search_tasks.append(self._search_pexels(optimized_query, search_request))
            
            if self.api_configs['pixabay']['enabled']:
                search_tasks.append(self._search_pixabay(optimized_query, search_request))
            
            # Execute searches in parallel
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and filter results
            all_results = []
            for results in results_lists:
                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    logger.warning(f"Search error: {results}")
            
            # Score and rank results
            scored_results = self._score_and_rank_results(all_results, search_request)
            
            # Filter by quality threshold
            filtered_results = [
                result for result in scored_results 
                if result.quality_score >= search_request.quality_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} quality images for '{search_request.query}'")
            return filtered_results[:search_request.max_results]
            
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []
    
    def _optimize_search_query(self, query: str, entity_type: str) -> str:
        """Optimize search query for entity type."""
        
        config = self.entity_search_configs.get(entity_type, {})
        modifiers = config.get('quality_modifiers', [])
        exclude_terms = config.get('exclude_terms', [])
        
        # Add quality modifiers
        if modifiers:
            # Pick top 2 modifiers to avoid over-constraining
            query += f" {' '.join(modifiers[:2])}"
        
        # Add exclusions (for APIs that support it)
        if exclude_terms:
            # Note: Actual implementation would depend on API syntax
            pass
        
        return query.strip()
    
    async def _search_unsplash(self, query: str, request: SearchRequest) -> List[ImageResult]:
        """Search Unsplash API."""
        
        if not self._check_rate_limit('unsplash'):
            logger.warning("Unsplash rate limit exceeded")
            return []
        
        try:
            # Mock Unsplash API response (replace with real API)
            results = self._create_mock_unsplash_results(query, request)
            self._update_rate_limit('unsplash')
            return results
            
        except Exception as e:
            logger.error(f"Unsplash search error: {e}")
            return []
    
    async def _search_pexels(self, query: str, request: SearchRequest) -> List[ImageResult]:
        """Search Pexels API."""
        
        if not self._check_rate_limit('pexels'):
            logger.warning("Pexels rate limit exceeded")
            return []
        
        try:
            # Mock Pexels API response (replace with real API)
            results = self._create_mock_pexels_results(query, request)
            self._update_rate_limit('pexels')
            return results
            
        except Exception as e:
            logger.error(f"Pexels search error: {e}")
            return []
    
    async def _search_pixabay(self, query: str, request: SearchRequest) -> List[ImageResult]:
        """Search Pixabay API."""
        
        if not self._check_rate_limit('pixabay'):
            logger.warning("Pixabay rate limit exceeded")
            return []
        
        try:
            # Mock Pixabay API response (replace with real API)
            results = self._create_mock_pixabay_results(query, request)
            self._update_rate_limit('pixabay')
            return results
            
        except Exception as e:
            logger.error(f"Pixabay search error: {e}")
            return []
    
    def _create_mock_unsplash_results(self, query: str, request: SearchRequest) -> List[ImageResult]:
        """Create mock Unsplash results for testing."""
        
        results = []
        entity_config = self.entity_search_configs.get(request.entity_type, {})
        
        # Generate realistic mock results
        for i in range(min(request.max_results, 5)):
            width = np.random.randint(1200, 2400)
            height = np.random.randint(800, 1600)
            
            result = ImageResult(
                url=f"https://images.unsplash.com/photo-{query.replace(' ', '-')}-{i}",
                thumbnail_url=f"https://images.unsplash.com/photo-{query.replace(' ', '-')}-{i}?w=400",
                width=width,
                height=height,
                title=f"{query.title()} - Professional Photo",
                photographer=f"Photographer {i+1}",
                source="unsplash",
                search_query=query,
                entity_name=request.entity_name,
                entity_type=request.entity_type,
                download_url=f"https://images.unsplash.com/photo-{query.replace(' ', '-')}-{i}?dl=1",
                license="free"
            )
            
            results.append(result)
        
        return results
    
    def _create_mock_pexels_results(self, query: str, request: SearchRequest) -> List[ImageResult]:
        """Create mock Pexels results for testing."""
        
        results = []
        
        for i in range(min(request.max_results, 4)):
            width = np.random.randint(1000, 2000)
            height = np.random.randint(600, 1200)
            
            result = ImageResult(
                url=f"https://images.pexels.com/photos/{query.replace(' ', '-')}-{i}.jpeg",
                thumbnail_url=f"https://images.pexels.com/photos/{query.replace(' ', '-')}-{i}.jpeg?w=300",
                width=width,
                height=height,
                title=f"{query.title()} - Stock Photo",
                photographer=f"Pexels Photographer {i+1}",
                source="pexels",
                search_query=query,
                entity_name=request.entity_name,
                entity_type=request.entity_type,
                download_url=f"https://images.pexels.com/photos/{query.replace(' ', '-')}-{i}.jpeg",
                license="free"
            )
            
            results.append(result)
        
        return results
    
    def _create_mock_pixabay_results(self, query: str, request: SearchRequest) -> List[ImageResult]:
        """Create mock Pixabay results for testing."""
        
        results = []
        
        for i in range(min(request.max_results, 3)):
            width = np.random.randint(800, 1600)
            height = np.random.randint(600, 1200)
            
            result = ImageResult(
                url=f"https://cdn.pixabay.com/photo/{query.replace(' ', '-')}-{i}.jpg",
                thumbnail_url=f"https://cdn.pixabay.com/photo/{query.replace(' ', '-')}-{i}_150.jpg",
                width=width,
                height=height,
                title=f"{query.title()} - Free Image",
                photographer=f"Pixabay User {i+1}",
                source="pixabay",
                search_query=query,
                entity_name=request.entity_name,
                entity_type=request.entity_type,
                download_url=f"https://cdn.pixabay.com/photo/{query.replace(' ', '-')}-{i}.jpg",
                license="free"
            )
            
            results.append(result)
        
        return results
    
    def _score_and_rank_results(self, results: List[ImageResult], request: SearchRequest) -> List[ImageResult]:
        """Score and rank image results by quality."""
        
        entity_config = self.entity_search_configs.get(request.entity_type, {})
        
        for result in results:
            # Resolution score
            resolution_score = self._calculate_resolution_score(
                result.width, result.height, entity_config
            )
            result.resolution_score = resolution_score
            
            # Relevance score (based on query matching)
            relevance_score = self._calculate_relevance_score(
                result.title, result.search_query, request.entity_type
            )
            result.relevance_score = relevance_score
            
            # Aesthetic score (mock - would use ML model in production)
            aesthetic_score = self._calculate_aesthetic_score(result, entity_config)
            result.aesthetic_score = aesthetic_score
            
            # Overall quality score
            result.quality_score = (
                resolution_score * self.quality_weights['resolution'] +
                relevance_score * self.quality_weights['relevance'] +
                aesthetic_score * self.quality_weights['aesthetic']
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
        
        # Word overlap
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        overlap = len(query_words.intersection(title_words))
        
        if len(query_words) > 0:
            word_match_score = overlap / len(query_words)
        else:
            word_match_score = 0.0
        
        # Entity type relevance
        entity_keywords = {
            'PERSON': ['person', 'portrait', 'face', 'people'],
            'ORGANIZATION': ['company', 'business', 'corporate', 'office'],
            'LOCATION': ['place', 'location', 'city', 'country', 'landscape'],
            'PRODUCT': ['product', 'item', 'device', 'gadget'],
            'BRAND': ['brand', 'logo', 'company']
        }
        
        type_keywords = entity_keywords.get(entity_type, [])
        type_relevance = any(keyword in title_lower for keyword in type_keywords)
        
        return min(1.0, word_match_score + (0.2 if type_relevance else 0))
    
    def _calculate_aesthetic_score(self, result: ImageResult, entity_config: Dict) -> float:
        """Calculate aesthetic quality score (mock implementation)."""
        
        # Mock aesthetic scoring - in production would use ML model
        base_score = 0.7  # Default aesthetic score
        
        # Boost for professional sources
        if result.source == 'unsplash':
            base_score += 0.1
        elif result.source == 'pexels':
            base_score += 0.05
        
        # Boost for certain keywords in title
        quality_keywords = ['professional', 'high quality', 'clean', 'clear', 'sharp']
        title_lower = result.title.lower()
        
        for keyword in quality_keywords:
            if keyword in title_lower:
                base_score += 0.05
        
        return min(1.0, base_score)
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API rate limit allows more requests."""
        
        current_time = time.time()
        rate_info = self.rate_limits[api_name]
        
        # Reset if hour has passed
        if current_time >= rate_info['reset_time']:
            rate_info['count'] = 0
            rate_info['reset_time'] = current_time + 3600
        
        # Check if under limit
        api_config = self.api_configs[api_name]
        return rate_info['count'] < api_config['rate_limit']
    
    def _update_rate_limit(self, api_name: str):
        """Update rate limit counter."""
        self.rate_limits[api_name]['count'] += 1
    
    def get_search_statistics(self) -> Dict:
        """Get image search statistics."""
        
        stats = {
            'apis_enabled': sum(1 for config in self.api_configs.values() if config['enabled']),
            'rate_limits': {
                api: {
                    'requests_used': info['count'],
                    'limit': self.api_configs[api]['rate_limit'],
                    'reset_time': info['reset_time']
                }
                for api, info in self.rate_limits.items()
            },
            'entity_types_supported': list(self.entity_search_configs.keys()),
            'quality_weights': self.quality_weights
        }
        
        return stats
    
    async def search_for_entity(self, entity, emphasized_time: float = 0.0) -> List[ImageResult]:
        """
        Convenience method to search for a specific entity.
        
        Args:
            entity: Entity object from NLP system
            emphasized_time: When this entity was emphasized in audio
            
        Returns:
            List of relevant image results
        """
        
        # Create search request from entity
        search_request = SearchRequest(
            query=getattr(entity, 'search_queries', [entity.canonical_name])[0] if hasattr(entity, 'search_queries') and entity.search_queries else entity.canonical_name,
            entity_name=entity.canonical_name,
            entity_type=entity.entity_type if hasattr(entity, 'entity_type') else getattr(entity, 'entity_type', 'MISC'),
            max_results=5,
            min_width=800,
            min_height=600,
            quality_threshold=0.6
        )
        
        # Search for images
        results = await self.search_images(search_request)
        
        # Set timing information
        for result in results:
            result.suggested_start_time = emphasized_time
            result.suggested_duration = 3.0  # Default 3 seconds
        
        return results 