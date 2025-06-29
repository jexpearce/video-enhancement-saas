"""
Pre-Curated Image Cache Manager - Phase 2 Days 23-24

Redis-backed caching system for popular entity images with:
- ML-based quality filtering
- Intelligent cache warming
- Efficient storage and retrieval
- TTL management based on popularity
"""

import pickle
import json
import logging
from typing import Optional, List, Dict, Set, Any
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .curator import CuratedImage, ImageCurator
from ..providers.base import ImageResult
from ...nlp.entity_enricher import EnrichedEntity

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cached curated images for an entity."""
    entity_id: str
    entity_text: str
    entity_type: str
    curated_images: List[CuratedImage]
    cache_timestamp: datetime
    access_count: int
    popularity_score: float
    expires_at: datetime

class ImageCacheManager:
    """Advanced caching system for pre-curated images."""
    
    def __init__(self, redis_client: Any, config: Optional[Dict] = None):
        """
        Initialize the cache manager.
        
        Args:
            redis_client: Redis client instance
            config: Configuration for caching behavior
        """
        self.redis = redis_client
        self.config = config or {}
        
        # Cache configuration
        self.default_ttl = self.config.get('default_ttl', 86400 * 7)  # 7 days
        self.popular_entity_ttl = self.config.get('popular_ttl', 86400 * 30)  # 30 days
        self.max_images_per_entity = self.config.get('max_images_per_entity', 5)
        self.cache_warming_threshold = self.config.get('warming_threshold', 10)
        
        # Cache key prefixes
        self.entity_cache_prefix = "curated_images:entity:"
        self.popularity_prefix = "entity_popularity:"
        self.access_count_prefix = "entity_access:"
        self.warming_queue_key = "cache_warming_queue"
        
        # In-memory LRU cache for frequently accessed items
        self.local_cache: Dict[str, CacheEntry] = {}
        self.local_cache_size = self.config.get('local_cache_size', 100)
        
    async def get_curated_images(
        self,
        entity_id: str,
        entity_text: str,
        entity_type: str
    ) -> Optional[List[CuratedImage]]:
        """
        Get curated images for an entity from cache.
        
        Args:
            entity_id: Unique entity identifier
            entity_text: Entity text
            entity_type: Entity type (PERSON, LOCATION, etc.)
            
        Returns:
            List of curated images if cached, None otherwise
        """
        cache_key = f"{self.entity_cache_prefix}{entity_id}"
        
        try:
            # Check local cache first
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                if entry.expires_at > datetime.utcnow():
                    await self._increment_access_count(entity_id)
                    logger.debug(f"Cache hit (local) for entity: {entity_text}")
                    return entry.curated_images
                else:
                    # Expired, remove from local cache
                    del self.local_cache[cache_key]
            
            # Check Redis cache
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                entry_dict = pickle.loads(cached_data)
                entry = self._dict_to_cache_entry(entry_dict)
                
                if entry.expires_at > datetime.utcnow():
                    # Valid cache entry
                    await self._increment_access_count(entity_id)
                    
                    # Add to local cache
                    self._add_to_local_cache(cache_key, entry)
                    
                    logger.debug(f"Cache hit (Redis) for entity: {entity_text}")
                    return entry.curated_images
                else:
                    # Expired, remove from Redis
                    await self.redis.delete(cache_key)
            
            logger.debug(f"Cache miss for entity: {entity_text}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached images for {entity_text}: {e}")
            return None
    
    async def cache_curated_images(
        self,
        entity: EnrichedEntity,
        curated_images: List[CuratedImage]
    ) -> bool:
        """
        Cache curated images for an entity.
        
        Args:
            entity: The entity
            curated_images: List of curated images to cache
            
        Returns:
            True if successfully cached
        """
        try:
            entity_id = getattr(entity, 'id', entity.text)
            cache_key = f"{self.entity_cache_prefix}{entity_id}"
            
            # Calculate TTL based on entity popularity
            popularity_score = await self._get_popularity_score(entity_id)
            ttl = await self._calculate_ttl(popularity_score)
            
            # Create cache entry
            entry = CacheEntry(
                entity_id=entity_id,
                entity_text=entity.text,
                entity_type=entity.entity_type,
                curated_images=curated_images,
                cache_timestamp=datetime.utcnow(),
                access_count=1,
                popularity_score=popularity_score,
                expires_at=datetime.utcnow() + timedelta(seconds=ttl)
            )
            
            # Serialize and cache in Redis
            entry_dict = self._cache_entry_to_dict(entry)
            serialized_data = pickle.dumps(entry_dict)
            
            await self.redis.setex(cache_key, ttl, serialized_data)
            
            # Add to local cache
            self._add_to_local_cache(cache_key, entry)
            
            logger.info(f"Cached {len(curated_images)} images for entity: {entity.text}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching images for {entity.text}: {e}")
            return False
    
    async def warm_cache_for_popular_entities(self, curator: ImageCurator) -> int:
        """
        Pre-warm cache for popular entities.
        
        Args:
            curator: ImageCurator instance for processing
            
        Returns:
            Number of entities processed
        """
        try:
            # Get popular entities that need warming
            entities_to_warm = await self._get_entities_for_warming()
            
            if not entities_to_warm:
                logger.info("No entities need cache warming")
                return 0
            
            processed_count = 0
            
            for entity_data in entities_to_warm:
                try:
                    # Create enriched entity from cached data
                    entity = self._dict_to_enriched_entity(entity_data)
                    
                    # Check if already cached and fresh
                    cached = await self.get_curated_images(
                        entity_data['id'],
                        entity_data['text'],
                        entity_data['entity_type']
                    )
                    
                    if cached:
                        continue  # Already fresh in cache
                    
                    # Get candidate images (this would integrate with MultiSourceImageRetriever)
                    candidate_images = await self._get_candidate_images_for_entity(entity)
                    
                    if candidate_images:
                        # Curate images
                        curated_images = await curator.curate_entity_images(
                            entity,
                            candidate_images
                        )
                        
                        if curated_images:
                            # Cache the results
                            await self.cache_curated_images(entity, curated_images)
                            processed_count += 1
                            
                except Exception as e:
                    logger.error(f"Error warming cache for entity {entity_data.get('text', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Cache warming completed. Processed {processed_count} entities.")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error during cache warming: {e}")
            return 0
    
    async def invalidate_entity_cache(self, entity_id: str) -> bool:
        """
        Invalidate cached images for a specific entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if successfully invalidated
        """
        try:
            cache_key = f"{self.entity_cache_prefix}{entity_id}"
            
            # Remove from Redis
            await self.redis.delete(cache_key)
            
            # Remove from local cache
            if cache_key in self.local_cache:
                del self.local_cache[cache_key]
            
            logger.info(f"Invalidated cache for entity: {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache for entity {entity_id}: {e}")
            return False
    
    async def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics."""
        try:
            # Get cache size information
            total_keys = len(await self.redis.keys(f"{self.entity_cache_prefix}*"))
            
            # Get popular entities
            popular_entities = await self._get_most_accessed_entities(limit=10)
            
            # Calculate hit rates (simplified)
            total_accesses = 0
            for entity_data in popular_entities:
                entity_id = entity_data['entity_id']
                access_count = await self.redis.get(f"{self.access_count_prefix}{entity_id}")
                total_accesses += int(access_count or 0)
            
            return {
                'total_cached_entities': total_keys,
                'local_cache_size': len(self.local_cache),
                'popular_entities': popular_entities,
                'total_accesses': total_accesses,
                'cache_memory_usage': self._estimate_cache_memory_usage()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}
    
    async def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries."""
        try:
            # Get all entity cache keys
            cache_keys = await self.redis.keys(f"{self.entity_cache_prefix}*")
            
            cleaned_count = 0
            
            for key in cache_keys:
                try:
                    cached_data = await self.redis.get(key)
                    if cached_data:
                        entry_dict = pickle.loads(cached_data)
                        expires_at = datetime.fromisoformat(entry_dict['expires_at'])
                        
                        if expires_at <= datetime.utcnow():
                            await self.redis.delete(key)
                            
                            # Remove from local cache too
                            if key in self.local_cache:
                                del self.local_cache[key]
                                
                            cleaned_count += 1
                            
                except Exception as e:
                    logger.warning(f"Error checking expiry for key {key}: {e}")
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0
    
    async def _increment_access_count(self, entity_id: str):
        """Increment access count for an entity."""
        access_key = f"{self.access_count_prefix}{entity_id}"
        await self.redis.incr(access_key)
        await self.redis.expire(access_key, 86400 * 30)  # Keep for 30 days
    
    async def _get_popularity_score(self, entity_id: str) -> float:
        """Get popularity score for an entity."""
        try:
            popularity_key = f"{self.popularity_prefix}{entity_id}"
            score = await self.redis.get(popularity_key)
            return float(score) if score else 0.5  # Default popularity
        except:
            return 0.5
    
    async def _calculate_ttl(self, popularity_score: float) -> int:
        """Calculate TTL based on popularity score."""
        if popularity_score >= 0.8:
            return self.popular_entity_ttl
        elif popularity_score >= 0.6:
            return self.default_ttl * 2
        else:
            return self.default_ttl
    
    def _add_to_local_cache(self, cache_key: str, entry: CacheEntry):
        """Add entry to local cache with LRU eviction."""
        if len(self.local_cache) >= self.local_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.local_cache.keys(),
                key=lambda k: self.local_cache[k].cache_timestamp
            )
            del self.local_cache[oldest_key]
        
        self.local_cache[cache_key] = entry
    
    def _cache_entry_to_dict(self, entry: CacheEntry) -> Dict:
        """Convert CacheEntry to dictionary for serialization."""
        entry_dict = asdict(entry)
        
        # Convert datetime objects to ISO strings
        entry_dict['cache_timestamp'] = entry.cache_timestamp.isoformat()
        entry_dict['expires_at'] = entry.expires_at.isoformat()
        
        # Convert CuratedImage objects to dictionaries
        entry_dict['curated_images'] = [
            {
                'entity_id': img.entity_id,
                'image': asdict(img.image),
                'relevance_score': img.relevance_score,
                'curation_metadata': img.curation_metadata,
                'face_detection_result': img.face_detection_result,
                'quality_assessment': img.quality_assessment,
                'clip_similarity': img.clip_similarity
            }
            for img in entry.curated_images
        ]
        
        return entry_dict
    
    def _dict_to_cache_entry(self, entry_dict: Dict) -> CacheEntry:
        """Convert dictionary to CacheEntry object."""
        # Convert ISO strings back to datetime objects
        entry_dict['cache_timestamp'] = datetime.fromisoformat(entry_dict['cache_timestamp'])
        entry_dict['expires_at'] = datetime.fromisoformat(entry_dict['expires_at'])
        
        # Convert dictionaries back to CuratedImage objects
        curated_images = []
        for img_dict in entry_dict['curated_images']:
            # Reconstruct ImageResult
            image_result = ImageResult(**img_dict['image'])
            
            # Reconstruct CuratedImage
            curated_image = CuratedImage(
                entity_id=img_dict['entity_id'],
                image=image_result,
                relevance_score=img_dict['relevance_score'],
                curation_metadata=img_dict['curation_metadata'],
                face_detection_result=img_dict.get('face_detection_result'),
                quality_assessment=img_dict.get('quality_assessment'),
                clip_similarity=img_dict.get('clip_similarity')
            )
            
            curated_images.append(curated_image)
        
        entry_dict['curated_images'] = curated_images
        
        return CacheEntry(**entry_dict)
    
    async def _get_entities_for_warming(self) -> List[Dict]:
        """Get list of entities that need cache warming."""
        # This would typically query a database of popular entities
        # For now, return a mock list
        return [
            {
                'id': 'biden',
                'text': 'Joe Biden',
                'entity_type': 'PERSON',
                'popularity_score': 0.95
            },
            {
                'id': 'tesla',
                'text': 'Tesla',
                'entity_type': 'ORG',
                'popularity_score': 0.85
            },
            {
                'id': 'new_york',
                'text': 'New York',
                'entity_type': 'LOCATION',
                'popularity_score': 0.80
            }
        ]
    
    def _dict_to_enriched_entity(self, entity_data: Dict) -> EnrichedEntity:
        """Convert dictionary to EnrichedEntity object."""
        # Create a minimal EnrichedEntity for cache warming
        return EnrichedEntity(
            text=entity_data['text'],
            entity_type=entity_data['entity_type'],
            confidence=1.0,
            image_potential='EXCELLENT',
            canonical_name=entity_data['text']
        )
    
    async def _get_candidate_images_for_entity(self, entity: EnrichedEntity) -> List[ImageResult]:
        """Get candidate images for entity (would integrate with MultiSourceImageRetriever)."""
        # This would call the MultiSourceImageRetriever
        # For now, return empty list as placeholder
        return []
    
    async def _get_most_accessed_entities(self, limit: int = 10) -> List[Dict]:
        """Get most accessed entities for statistics."""
        try:
            access_keys = await self.redis.keys(f"{self.access_count_prefix}*")
            entity_access_data = []
            
            for key in access_keys:
                try:
                    entity_id = key.decode().replace(self.access_count_prefix, '')
                    access_count = int(await self.redis.get(key) or 0)
                    
                    entity_access_data.append({
                        'entity_id': entity_id,
                        'access_count': access_count
                    })
                except:
                    continue
            
            # Sort by access count and return top entities
            entity_access_data.sort(key=lambda x: x['access_count'], reverse=True)
            return entity_access_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting most accessed entities: {e}")
            return []
    
    def _estimate_cache_memory_usage(self) -> Dict:
        """Estimate memory usage of local cache."""
        try:
            import sys
            
            total_size = 0
            for entry in self.local_cache.values():
                total_size += sys.getsizeof(entry)
                total_size += sum(sys.getsizeof(img) for img in entry.curated_images)
            
            return {
                'local_cache_bytes': total_size,
                'local_cache_mb': round(total_size / (1024 * 1024), 2),
                'entries_count': len(self.local_cache)
            }
        except:
            return {'error': 'Could not estimate memory usage'} 