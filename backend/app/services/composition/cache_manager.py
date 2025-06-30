"""
Cache Manager for Video Composition

Implements Redis-based caching to improve performance:
- Cache animation timelines
- Cache image assets and metadata
- Cache style selections
- Cache FFmpeg filter graphs
"""

import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import timedelta
import redis
import asyncio
from dataclasses import asdict

logger = logging.getLogger(__name__)

class CompositionCacheManager:
    """
    Redis-based cache manager for video composition operations.
    
    Provides multi-level caching to reduce redundant computations:
    - Level 1: Animation timelines (1 hour TTL)
    - Level 2: Image asset metadata (24 hours TTL)  
    - Level 3: Style selections (12 hours TTL)
    - Level 4: FFmpeg filter graphs (30 minutes TTL)
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize cache manager with Redis connection."""
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None
        
        # Cache TTL settings
        self.ttl_settings = {
            'animation_timeline': 3600,      # 1 hour
            'image_metadata': 86400,         # 24 hours  
            'style_selection': 43200,        # 12 hours
            'filter_graph': 1800,            # 30 minutes
            'composition_result': 7200       # 2 hours
        }
        
        # Key prefixes
        self.key_prefixes = {
            'animation': 'composition:animation:',
            'image': 'composition:image:',
            'style': 'composition:style:',
            'filter': 'composition:filter:',
            'result': 'composition:result:'
        }
    
    def _is_available(self) -> bool:
        """Check if Redis is available."""
        return self.redis_client is not None
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate deterministic cache key from data."""
        
        # Create stable hash from data
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        
        return f"{self.key_prefixes[prefix]}{data_hash}"
    
    async def get_animation_timeline(
        self,
        emphasis_points: List[Dict],
        style: Dict,
        video_duration: float
    ) -> Optional[Dict]:
        """Get cached animation timeline."""
        
        if not self._is_available():
            return None
        
        try:
            cache_key = self._generate_cache_key('animation', {
                'emphasis_count': len(emphasis_points),
                'emphasis_scores': [p.get('emphasis_score', 0) for p in emphasis_points[:5]],
                'style_template': style.get('template_name', ''),
                'video_duration': round(video_duration, 1)
            })
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                timeline = json.loads(cached_data)
                logger.debug(f"Cache hit for animation timeline: {cache_key}")
                return timeline
            
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving animation timeline from cache: {e}")
            return None
    
    async def cache_animation_timeline(
        self,
        emphasis_points: List[Dict],
        style: Dict,
        video_duration: float,
        timeline: Dict
    ) -> None:
        """Cache animation timeline."""
        
        if not self._is_available():
            return
        
        try:
            cache_key = self._generate_cache_key('animation', {
                'emphasis_count': len(emphasis_points),
                'emphasis_scores': [p.get('emphasis_score', 0) for p in emphasis_points[:5]],
                'style_template': style.get('template_name', ''),
                'video_duration': round(video_duration, 1)
            })
            
            # Store with TTL
            self.redis_client.setex(
                cache_key,
                self.ttl_settings['animation_timeline'],
                json.dumps(timeline)
            )
            
            logger.debug(f"Cached animation timeline: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Error caching animation timeline: {e}")
    
    async def get_image_metadata(self, image_url: str) -> Optional[Dict]:
        """Get cached image metadata."""
        
        if not self._is_available():
            return None
        
        try:
            cache_key = self._generate_cache_key('image', {'url': image_url})
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                metadata = json.loads(cached_data)
                logger.debug(f"Cache hit for image metadata: {image_url[:50]}...")
                return metadata
            
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving image metadata from cache: {e}")
            return None
    
    async def cache_image_metadata(self, image_url: str, metadata: Dict) -> None:
        """Cache image metadata."""
        
        if not self._is_available():
            return
        
        try:
            cache_key = self._generate_cache_key('image', {'url': image_url})
            
            # Store with TTL
            self.redis_client.setex(
                cache_key,
                self.ttl_settings['image_metadata'],
                json.dumps(metadata)
            )
            
            logger.debug(f"Cached image metadata: {image_url[:50]}...")
            
        except Exception as e:
            logger.warning(f"Error caching image metadata: {e}")
    
    async def get_style_selection(
        self,
        content_hash: str,
        preferences: Dict
    ) -> Optional[Dict]:
        """Get cached style selection."""
        
        if not self._is_available():
            return None
        
        try:
            cache_key = self._generate_cache_key('style', {
                'content_hash': content_hash,
                'platform': preferences.get('platform', 'tiktok'),
                'mood': preferences.get('mood', 'energetic')
            })
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                style = json.loads(cached_data)
                logger.debug(f"Cache hit for style selection: {content_hash[:16]}")
                return style
            
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving style selection from cache: {e}")
            return None
    
    async def cache_style_selection(
        self,
        content_hash: str,
        preferences: Dict,
        style: Dict
    ) -> None:
        """Cache style selection."""
        
        if not self._is_available():
            return
        
        try:
            cache_key = self._generate_cache_key('style', {
                'content_hash': content_hash,
                'platform': preferences.get('platform', 'tiktok'),
                'mood': preferences.get('mood', 'energetic')
            })
            
            # Store with TTL
            self.redis_client.setex(
                cache_key,
                self.ttl_settings['style_selection'],
                json.dumps(style)
            )
            
            logger.debug(f"Cached style selection: {content_hash[:16]}")
            
        except Exception as e:
            logger.warning(f"Error caching style selection: {e}")
    
    async def get_filter_graph(
        self,
        overlay_config: Dict,
        style: Dict
    ) -> Optional[str]:
        """Get cached FFmpeg filter graph."""
        
        if not self._is_available():
            return None
        
        try:
            cache_key = self._generate_cache_key('filter', {
                'overlay_count': overlay_config.get('overlay_count', 0),
                'animations': overlay_config.get('animations', []),
                'style_template': style.get('template_name', '')
            })
            
            cached_filter = self.redis_client.get(cache_key)
            if cached_filter:
                logger.debug(f"Cache hit for filter graph")
                return cached_filter
            
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving filter graph from cache: {e}")
            return None
    
    async def cache_filter_graph(
        self,
        overlay_config: Dict,
        style: Dict,
        filter_graph: str
    ) -> None:
        """Cache FFmpeg filter graph."""
        
        if not self._is_available():
            return
        
        try:
            cache_key = self._generate_cache_key('filter', {
                'overlay_count': overlay_config.get('overlay_count', 0),
                'animations': overlay_config.get('animations', []),
                'style_template': style.get('template_name', '')
            })
            
            # Store with TTL
            self.redis_client.setex(
                cache_key,
                self.ttl_settings['filter_graph'],
                filter_graph
            )
            
            logger.debug(f"Cached filter graph")
            
        except Exception as e:
            logger.warning(f"Error caching filter graph: {e}")
    
    async def invalidate_job_cache(self, job_id: str) -> None:
        """Invalidate all cache entries for a specific job."""
        
        if not self._is_available():
            return
        
        try:
            # Find all keys related to this job
            patterns = [f"{prefix}*{job_id}*" for prefix in self.key_prefixes.values()]
            
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.debug(f"Invalidated {len(keys)} cache entries for job {job_id}")
            
        except Exception as e:
            logger.warning(f"Error invalidating cache for job {job_id}: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        
        if not self._is_available():
            return {'status': 'unavailable'}
        
        try:
            info = self.redis_client.info()
            
            # Count keys by prefix
            key_counts = {}
            for name, prefix in self.key_prefixes.items():
                keys = self.redis_client.keys(f"{prefix}*")
                key_counts[name] = len(keys)
            
            return {
                'status': 'available',
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'key_counts': key_counts,
                'total_keys': sum(key_counts.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def cleanup_expired_keys(self) -> Dict[str, int]:
        """Clean up expired keys (manual cleanup)."""
        
        if not self._is_available():
            return {'cleaned': 0}
        
        try:
            cleaned_counts = {}
            total_cleaned = 0
            
            for name, prefix in self.key_prefixes.items():
                keys = self.redis_client.keys(f"{prefix}*")
                cleaned = 0
                
                for key in keys:
                    ttl = self.redis_client.ttl(key)
                    if ttl == -1:  # No expiration set
                        # Set appropriate expiration
                        ttl_seconds = self.ttl_settings.get(name.split('_')[0], 3600)
                        self.redis_client.expire(key, ttl_seconds)
                        cleaned += 1
                
                cleaned_counts[name] = cleaned
                total_cleaned += cleaned
            
            logger.info(f"Cleaned up {total_cleaned} cache keys")
            return {'cleaned': total_cleaned, 'by_type': cleaned_counts}
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return {'cleaned': 0, 'error': str(e)}
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Closed Redis connection") 