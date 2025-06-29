"""
Multi-Source Image Retriever - Phase 2

Unified interface that combines multiple image providers with:
- Intelligent provider selection
- Result aggregation and deduplication
- Fallback handling and load balancing
- Quality-based ranking across sources
"""

import asyncio
from typing import List, Dict, Optional, Set
import logging
from datetime import datetime
from dataclasses import dataclass

from .providers.base import (
    ImageProvider,
    ImageResult, 
    ImageLicense,
    SearchRequest,
    RateLimitError,
    ProviderError
)
from .providers.wikimedia import WikimediaProvider
from .providers.unsplash import UnsplashProvider

logger = logging.getLogger(__name__)

@dataclass
class ProviderConfig:
    """Configuration for an image provider"""
    provider: ImageProvider
    weight: float = 1.0  # Relative importance
    max_results: int = 10
    enabled: bool = True
    priority: int = 1  # Lower = higher priority

@dataclass
class RetrievalResult:
    """Result from multi-source retrieval"""
    images: List[ImageResult]
    provider_stats: Dict[str, Dict]
    total_time: float
    query_used: str

class MultiSourceImageRetriever:
    """Advanced multi-source image retrieval system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.providers: List[ProviderConfig] = []
        self.config = config or {}
        
        # Default quality thresholds
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        self.min_relevance_score = self.config.get('min_relevance_score', 0.2)
        self.max_results_per_provider = self.config.get('max_results_per_provider', 15)
        
        # Deduplication settings
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        
    def add_provider(self, provider: ImageProvider, weight: float = 1.0, 
                    max_results: int = 10, priority: int = 1):
        """Add an image provider to the retrieval system"""
        
        provider_config = ProviderConfig(
            provider=provider,
            weight=weight,
            max_results=max_results,
            priority=priority
        )
        
        self.providers.append(provider_config)
        
        # Sort by priority
        self.providers.sort(key=lambda x: x.priority)
        
        logger.info(f"Added provider {provider.name} with weight {weight}")
        
    async def search(self, query: str, count: int = 20, 
                    entity_type: Optional[str] = None) -> RetrievalResult:
        """
        Search across all configured providers.
        
        Args:
            query: Search query
            count: Total number of results desired
            entity_type: Type of entity for query optimization
            
        Returns:
            RetrievalResult with aggregated results
        """
        start_time = datetime.utcnow()
        
        # Optimize query based on entity type
        optimized_query = self._optimize_query(query, entity_type)
        
        # Plan provider execution
        execution_plan = self._create_execution_plan(count)
        
        # Execute searches across providers
        provider_results = await self._execute_searches(
            optimized_query, 
            execution_plan,
            entity_type
        )
        
        # Aggregate and process results
        aggregated_results = self._aggregate_results(provider_results)
        
        # Deduplicate images
        deduplicated_results = self._deduplicate_images(aggregated_results)
        
        # Final ranking and selection
        final_results = self._rank_and_select(deduplicated_results, count)
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Generate provider statistics
        provider_stats = self._generate_stats(provider_results, total_time)
        
        return RetrievalResult(
            images=final_results,
            provider_stats=provider_stats,
            total_time=total_time,
            query_used=optimized_query
        )
        
    def _optimize_query(self, query: str, entity_type: Optional[str]) -> str:
        """Optimize query based on entity type and provider capabilities"""
        
        # Entity-specific optimizations
        if entity_type == "PERSON":
            # For people, add context that helps find portraits
            if "portrait" not in query.lower() and "photo" not in query.lower():
                query = f"{query} portrait"
        elif entity_type in ["GPE", "LOC"]:
            # For locations, prefer landmark/scenic photos
            if not any(word in query.lower() for word in ["landmark", "scenic", "landscape"]):
                query = f"{query} landmark"
        elif entity_type == "ORG":
            # For organizations, prefer official/professional imagery
            if "logo" not in query.lower() and "building" not in query.lower():
                query = f"{query} building"
                
        return query
        
    def _create_execution_plan(self, total_count: int) -> Dict[str, int]:
        """Create execution plan for providers based on weights"""
        
        if not self.providers:
            return {}
            
        # Calculate weighted distribution
        total_weight = sum(p.weight for p in self.providers if p.enabled)
        if total_weight == 0:
            return {}
            
        plan = {}
        remaining_count = total_count
        
        for provider_config in self.providers:
            if not provider_config.enabled:
                continue
                
            # Calculate allocation based on weight
            allocation = int((provider_config.weight / total_weight) * total_count)
            allocation = min(allocation, provider_config.max_results, remaining_count)
            
            if allocation > 0:
                plan[provider_config.provider.name] = allocation
                remaining_count -= allocation
                
        # Distribute any remaining slots to highest priority providers
        for provider_config in self.providers:
            if remaining_count <= 0:
                break
            if provider_config.enabled and provider_config.provider.name in plan:
                additional = min(remaining_count, provider_config.max_results - plan[provider_config.provider.name])
                plan[provider_config.provider.name] += additional
                remaining_count -= additional
                
        return plan
        
    async def _execute_searches(self, query: str, execution_plan: Dict[str, int],
                              entity_type: Optional[str]) -> Dict[str, List[ImageResult]]:
        """Execute searches across providers in parallel"""
        
        tasks = []
        provider_map = {p.provider.name: p.provider for p in self.providers}
        
        for provider_name, count in execution_plan.items():
            if provider_name in provider_map:
                provider = provider_map[provider_name]
                task = self._search_with_provider(provider, query, count)
                tasks.append((provider_name, task))
                
        # Execute all searches in parallel
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        for (provider_name, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Provider {provider_name} failed: {result}")
                results[provider_name] = []
            else:
                results[provider_name] = result
                
        return results
        
    async def _search_with_provider(self, provider: ImageProvider, 
                                   query: str, count: int) -> List[ImageResult]:
        """Search with a single provider with error handling"""
        
        try:
            results = await provider.search(query, count)
            
            # Filter by quality thresholds
            filtered_results = [
                img for img in results
                if img.quality_score >= self.min_quality_score
                and img.relevance_score >= self.min_relevance_score
            ]
            
            logger.info(f"Provider {provider.name}: {len(filtered_results)}/{len(results)} images passed filters")
            return filtered_results
            
        except RateLimitError as e:
            logger.warning(f"Provider {provider.name} rate limited: {e}")
            return []
        except ProviderError as e:
            logger.error(f"Provider {provider.name} error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error with provider {provider.name}: {e}")
            return []
            
    def _aggregate_results(self, provider_results: Dict[str, List[ImageResult]]) -> List[ImageResult]:
        """Aggregate results from all providers"""
        
        all_results = []
        for provider_name, results in provider_results.items():
            for result in results:
                # Add provider metadata
                result.metadata['source_provider'] = provider_name
                result.metadata['retrieval_time'] = datetime.utcnow().isoformat()
                all_results.append(result)
                
        return all_results
        
    def _deduplicate_images(self, images: List[ImageResult]) -> List[ImageResult]:
        """Remove duplicate images based on URL and visual similarity"""
        
        seen_urls: Set[str] = set()
        unique_images = []
        
        # Sort by quality score first (keep highest quality duplicates)
        images.sort(key=lambda x: x.quality_score, reverse=True)
        
        for image in images:
            # Check URL deduplication
            if image.image_url in seen_urls:
                continue
                
            # Check visual similarity (simplified - in production use perceptual hashing)
            is_similar = self._is_visually_similar(image, unique_images)
            if is_similar:
                continue
                
            seen_urls.add(image.image_url)
            unique_images.append(image)
            
        logger.info(f"Deduplication: {len(unique_images)}/{len(images)} unique images")
        return unique_images
        
    def _is_visually_similar(self, image: ImageResult, existing_images: List[ImageResult]) -> bool:
        """Check if image is visually similar to existing images (simplified)"""
        
        # Simplified similarity check based on title and dimensions
        for existing in existing_images[-10:]:  # Only check recent images for performance
            # Similar titles might indicate similar content
            if self._text_similarity(image.title or "", existing.title or "") > 0.8:
                return True
                
            # Similar aspect ratios and same provider might indicate similar content
            if (image.provider == existing.provider and
                abs((image.width / image.height) - (existing.width / existing.height)) < 0.1):
                return True
                
        return False
        
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified)"""
        
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
        
    def _rank_and_select(self, images: List[ImageResult], count: int) -> List[ImageResult]:
        """Final ranking and selection of images"""
        
        # Multi-factor scoring
        for image in images:
            # Combine scores with weights
            image.metadata['final_score'] = (
                image.relevance_score * 0.4 +
                image.quality_score * 0.3 +
                image.popularity_score * 0.2 +
                self._provider_bonus(image.provider) * 0.1
            )
            
        # Sort by final score
        images.sort(key=lambda x: x.metadata['final_score'], reverse=True)
        
        return images[:count]
        
    def _provider_bonus(self, provider_name: str) -> float:
        """Get provider-specific bonus score"""
        
        provider_bonuses = {
            'wikimedia': 0.9,  # High quality, verified licensing
            'unsplash': 0.8,   # Professional photography, commercial use
        }
        
        return provider_bonuses.get(provider_name, 0.5)
        
    def _generate_stats(self, provider_results: Dict[str, List[ImageResult]], 
                       total_time: float) -> Dict[str, Dict]:
        """Generate statistics about provider performance"""
        
        stats = {}
        
        for provider_name, results in provider_results.items():
            avg_quality = sum(img.quality_score for img in results) / len(results) if results else 0
            avg_relevance = sum(img.relevance_score for img in results) / len(results) if results else 0
            
            stats[provider_name] = {
                'result_count': len(results),
                'avg_quality_score': round(avg_quality, 3),
                'avg_relevance_score': round(avg_relevance, 3),
                'success_rate': 1.0 if results else 0.0
            }
            
        stats['total_execution_time'] = round(total_time, 3)
        return stats
        
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all configured providers"""
        
        health_status = {}
        
        for provider_config in self.providers:
            try:
                # Simple availability check
                test_url = "https://example.com/test.jpg"
                is_healthy = await provider_config.provider.check_availability(test_url)
                health_status[provider_config.provider.name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {provider_config.provider.name}: {e}")
                health_status[provider_config.provider.name] = False
                
        return health_status 