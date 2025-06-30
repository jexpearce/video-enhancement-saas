"""
Unsplash Integration with Rate Limiting - Phase 2

Advanced Unsplash image provider with:
- Intelligent query expansion
- Sophisticated rate limiting
- Quality scoring and filtering
- Commercial usage compliance
"""

import aiohttp
from typing import List, Dict, Optional
import asyncio
import time
import logging
from datetime import datetime, timedelta
import hashlib

from .base import (
    ImageProvider,
    ImageResult, 
    ImageLicense,
    SearchRequest,
    RateLimitError,
    ImageUnavailableError
)

logger = logging.getLogger(__name__)

class RateLimiter:
    """Sophisticated rate limiter for API compliance"""
    
    def __init__(self, calls_per_hour: int = 50):
        self.calls_per_hour = calls_per_hour
        self.calls = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire rate limit permission"""
        async with self.lock:
            now = datetime.utcnow()
            
            # Remove calls older than 1 hour
            cutoff = now - timedelta(hours=1)
            self.calls = [call_time for call_time in self.calls if call_time > cutoff]
            
            if len(self.calls) >= self.calls_per_hour:
                # Calculate wait time
                oldest_call = min(self.calls)
                wait_seconds = (oldest_call + timedelta(hours=1) - now).total_seconds()
                raise RateLimitError(f"Rate limit exceeded. Retry in {wait_seconds:.0f} seconds")
                
            self.calls.append(now)

class UnsplashProvider(ImageProvider):
    """Unsplash image provider with advanced features"""
    
    def __init__(self, access_key: str):
        self.access_key = access_key
        self.base_url = "https://api.unsplash.com"
        self.headers = {"Authorization": f"Client-ID {access_key}"}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiter (Unsplash free tier: 50 requests/hour)
        self.rate_limiter = RateLimiter(calls_per_hour=50)
        
        # Query expansion mappings
        self.expansion_mappings = {
            'person': {
                'terms': ['portrait', 'headshot', 'professional', 'face'],
                'orientation': 'portrait'
            },
            'location': {
                'terms': ['landscape', 'aerial', 'scenic', 'destination'],
                'orientation': 'landscape'
            },
            'organization': {
                'terms': ['corporate', 'building', 'office', 'headquarters'],
                'orientation': 'landscape'
            },
            'product': {
                'terms': ['product', 'isolated', 'clean', 'commercial'],
                'orientation': 'square'
            },
            'event': {
                'terms': ['gathering', 'ceremony', 'celebration', 'crowd'],
                'orientation': 'landscape'
            },
            'concept': {
                'terms': ['abstract', 'conceptual', 'symbolic', 'metaphor'],
                'orientation': 'landscape'
            }
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    @property
    def name(self) -> str:
        return "unsplash"
        
    @property
    def rate_limit(self) -> Dict[str, int]:
        return {
            "requests_per_hour": 50,
            "requests_per_minute": 10,
            "concurrent_requests": 3
        }
        
    async def search(self, query: str, count: int = 10) -> List[ImageResult]:
        """
        Search Unsplash with intelligent query expansion.
        
        Args:
            query: Search query string
            count: Maximum number of results
            
        Returns:
            List of ImageResult objects
        """
        if not self.session:
            async with self:
                return await self._perform_search(query, count)
        else:
            return await self._perform_search(query, count)
            
    async def _perform_search(self, query: str, count: int) -> List[ImageResult]:
        """Perform search with rate limiting and error handling"""
        
        try:
            # Check rate limit
            await self.rate_limiter.acquire()
            
            # Expand query for better results
            expanded_query, orientation = self._expand_query_intelligently(query)
            
            # Build search parameters
            params = {
                "query": expanded_query,
                "per_page": min(count, 30),  # Unsplash max per page
                "orientation": orientation,
                "content_filter": "high",  # Avoid NSFW content
                "order_by": "relevance"
            }
            
            if not self.session:
                raise RuntimeError("Session not initialized")
                
            async with self.session.get(
                f"{self.base_url}/search/photos",
                params=params
            ) as response:
                if response.status == 403:
                    raise RateLimitError("Unsplash API rate limit exceeded")
                elif response.status == 401:
                    raise RateLimitError("Invalid Unsplash API key")
                    
                response.raise_for_status()
                data = await response.json()
                
            # Process results
            images = []
            for photo in data.get('results', []):
                try:
                    image = self._process_photo_result(photo, query)
                    if image and self._passes_quality_filter(image):
                        images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to process Unsplash photo: {e}")
                    continue
                    
            # Sort by combined relevance and quality score with safe type conversion
            def safe_combined_score(x: ImageResult) -> float:
                try:
                    relevance = float(x.relevance_score) if x.relevance_score is not None else 0.0
                    quality = float(x.quality_score) if x.quality_score is not None else 0.0
                    return relevance * 0.6 + quality * 0.4
                except (ValueError, TypeError):
                    return 0.0
            
            images.sort(key=safe_combined_score, reverse=True)
            
            return images[:count]
            
        except RateLimitError:
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Unsplash API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Unsplash search: {e}")
            return []
            
    def _expand_query_intelligently(self, query: str) -> tuple[str, str]:
        """
        Intelligently expand query based on detected entity type.
        
        Returns:
            Tuple of (expanded_query, preferred_orientation)
        """
        query_lower = query.lower()
        
        # Detect entity type from query content
        entity_type = self._detect_entity_type(query_lower)
        
        if entity_type in self.expansion_mappings:
            mapping = self.expansion_mappings[entity_type]
            
            # Add 1-2 relevant expansion terms
            expansion_terms = mapping['terms'][:2]
            expanded_query = f"{query} {' '.join(expansion_terms)}"
            orientation = mapping['orientation']
        else:
            # Default expansion
            expanded_query = query
            orientation = "landscape"
            
        return expanded_query, orientation
        
    def _detect_entity_type(self, query: str) -> str:
        """Detect entity type from query content"""
        
        # Simple keyword-based detection (in production, use NLP)
        person_keywords = ['person', 'people', 'man', 'woman', 'politician', 'celebrity']
        location_keywords = ['country', 'city', 'place', 'building', 'mountain', 'ocean']
        org_keywords = ['company', 'corporation', 'organization', 'government']
        product_keywords = ['product', 'device', 'tool', 'car', 'phone']
        event_keywords = ['conference', 'meeting', 'protest', 'celebration', 'festival']
        
        if any(keyword in query for keyword in person_keywords):
            return 'person'
        elif any(keyword in query for keyword in location_keywords):
            return 'location'
        elif any(keyword in query for keyword in org_keywords):
            return 'organization'
        elif any(keyword in query for keyword in product_keywords):
            return 'product'
        elif any(keyword in query for keyword in event_keywords):
            return 'event'
        else:
            return 'concept'
            
    def _process_photo_result(self, photo: Dict, original_query: str) -> ImageResult:
        """Process a single Unsplash photo result"""
        
        # Extract metadata
        description = photo.get('description') or photo.get('alt_description', '')
        
        # Calculate relevance score
        relevance_score = self._calculate_unsplash_relevance(photo, original_query)
        
        # Calculate quality score  
        quality_score = self._calculate_unsplash_quality(photo)
        
        # Calculate popularity score
        popularity_score = self._calculate_popularity_score(photo)
        
        return ImageResult(
            provider="unsplash",
            image_url=photo['urls']['full'],
            thumbnail_url=photo['urls']['regular'],
            title=description[:100] if description else f"Photo by {photo['user']['name']}",
            description=description,
            author=photo['user']['name'],
            author_url=photo['user']['links']['html'],
            license=ImageLicense.COMMERCIAL_ALLOWED,
            license_url="https://unsplash.com/license",
            width=photo['width'],
            height=photo['height'],
            relevance_score=relevance_score,
            quality_score=quality_score,
            popularity_score=popularity_score,
            created_at=self._parse_unsplash_date(photo.get('created_at')),
            metadata={
                'likes': photo['likes'],
                'downloads': photo.get('downloads', 0),
                'color': photo['color'],
                'blur_hash': photo['blur_hash'],
                'user_id': photo['user']['id'],
                'photo_id': photo['id'],
                'tags': [tag['title'] for tag in photo.get('tags', [])],
                'unsplash_url': photo['links']['html']
            }
        )
        
    def _calculate_unsplash_relevance(self, photo: Dict, query: str) -> float:
        """Calculate relevance score for Unsplash photo"""
        
        score = 0.0
        query_terms = query.lower().split()
        
        # Description matching
        description = (photo.get('description') or photo.get('alt_description', '')).lower()
        if description:
            matches = sum(1 for term in query_terms if term in description)
            score += (matches / len(query_terms)) * 0.4
            
        # Tags matching  
        tags = [tag['title'].lower() for tag in photo.get('tags', [])]
        if tags:
            tag_text = ' '.join(tags)
            matches = sum(1 for term in query_terms if term in tag_text)
            score += (matches / len(query_terms)) * 0.3
            
        # Color relevance (if query mentions colors)
        color_keywords = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple']
        photo_color = photo.get('color', '').lower()
        for term in query_terms:
            if term in color_keywords and term in photo_color:
                score += 0.2
                break
                
        # User quality bonus (established photographers)
        if photo['user'].get('total_likes', 0) > 10000:
            score += 0.1
            
        return min(1.0, score)
        
    def _calculate_unsplash_quality(self, photo: Dict) -> float:
        """Calculate quality score for Unsplash photo"""
        
        # Start with base quality from resolution
        quality = self.calculate_base_quality_score(
            photo['width'],
            photo['height'],
            {}
        )
        
        # Engagement metrics
        likes = photo['likes']
        downloads = photo.get('downloads', 0)
        
        # Normalize engagement scores
        if likes > 1000:
            quality += 0.2
        elif likes > 100:
            quality += 0.1
        elif likes > 10:
            quality += 0.05
            
        if downloads > 1000:
            quality += 0.1
        elif downloads > 100:
            quality += 0.05
            
        # User reputation
        user = photo['user']
        user_photos = user.get('total_photos', 0)
        user_likes = user.get('total_likes', 0)
        
        if user_photos > 100 and user_likes > 10000:
            quality += 0.1  # Professional photographer
        elif user_photos > 50 and user_likes > 1000:
            quality += 0.05  # Semi-professional
            
        return min(1.0, quality)
        
    def _calculate_popularity_score(self, photo: Dict) -> float:
        """Calculate popularity score based on engagement"""
        
        likes = photo['likes']
        downloads = photo.get('downloads', 0)
        
        # Logarithmic scoring to avoid extreme values
        import math
        
        likes_score = min(1.0, math.log1p(likes) / 10)
        downloads_score = min(1.0, math.log1p(downloads) / 8)
        
        return (likes_score + downloads_score) / 2
        
    def _passes_quality_filter(self, image: ImageResult) -> bool:
        """Check if image passes minimum quality thresholds"""
        
        # Minimum resolution
        if image.width < 800 or image.height < 600:
            return False
            
        # Minimum quality score with safe type conversion
        try:
            quality_score = float(image.quality_score) if image.quality_score is not None else 0.0
            if quality_score < 0.3:
                return False
        except (ValueError, TypeError):
            return False
            
        # Must have description or tags
        if not image.description and not image.metadata.get('tags'):
            return False
            
        return True
        
    def _parse_unsplash_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse Unsplash date format"""
        if not date_str:
            return None
            
        try:
            # Unsplash format: 2023-01-15T10:30:00Z
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
            
    async def validate_license(self, image: ImageResult) -> bool:
        """Validate Unsplash license (always commercial allowed)"""
        return image.license == ImageLicense.COMMERCIAL_ALLOWED
        
    async def check_availability(self, image_url: str) -> bool:
        """Check if Unsplash image URL is accessible"""
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    async with session.head(image_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        return response.status == 200
            else:
                async with self.session.head(image_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return response.status == 200
        except Exception:
            return False 