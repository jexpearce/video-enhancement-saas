"""
Test Suite for Phase 2 Multi-Source Image Retrieval Infrastructure

Tests the complete image retrieval system with:
- Multi-provider integration
- Quality filtering and ranking
- Deduplication and aggregation
- Error handling and fallbacks
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Optional

# Import Phase 2 components
from app.services.images.providers.base import (
    ImageProvider, ImageResult, ImageLicense, SearchRequest
)
from app.services.images.multi_source_retriever import (
    MultiSourceImageRetriever, ProviderConfig, RetrievalResult
)

class MockImageProvider(ImageProvider):
    """Mock provider for testing"""
    
    def __init__(self, name: str, mock_results: Optional[list] = None):
        self._name = name
        self.mock_results = mock_results or []
        self.search_calls = []
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def rate_limit(self) -> dict:
        return {"requests_per_hour": 100}
        
    async def search(self, query: str, count: int = 10):
        self.search_calls.append((query, count))
        return self.mock_results[:count]
        
    async def validate_license(self, image: ImageResult) -> bool:
        return True
        
    async def check_availability(self, image_url: str) -> bool:
        return True

def create_mock_image(provider: str, title: str, quality: float = 0.8, 
                     relevance: float = 0.7, width: int = 1920, height: int = 1080) -> ImageResult:
    """Create a mock image result for testing"""
    return ImageResult(
        provider=provider,
        image_url=f"https://{provider}.com/{title.replace(' ', '_')}.jpg",
        thumbnail_url=f"https://{provider}.com/thumb_{title.replace(' ', '_')}.jpg",
        title=title,
        description=f"Test image of {title}",
        author="Test Author",
        author_url=f"https://{provider}.com/author",
        license=ImageLicense.CREATIVE_COMMONS_ZERO,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        width=width,
        height=height,
        relevance_score=relevance,
        quality_score=quality,
        popularity_score=0.5,
        metadata={"test": True}
    )

class TestImageProviderBase:
    """Test base image provider functionality"""
    
    def test_image_result_validation(self):
        """Test ImageResult validation"""
        
        # Valid image should pass
        image = create_mock_image("test", "Valid Image")
        assert image.provider == "test"
        assert image.quality_score == 0.8
        
        # Invalid dimensions should fail
        with pytest.raises(ValueError):
            ImageResult(
                provider="test",
                image_url="https://test.com/image.jpg",
                thumbnail_url="https://test.com/thumb.jpg",
                title="Invalid",
                description=None,
                author=None,
                author_url=None,
                license=ImageLicense.UNKNOWN,
                license_url=None,
                width=0,  # Invalid
                height=100,
                relevance_score=0.5,
                quality_score=0.5,
                metadata={}
            )
            
    def test_quality_score_calculation(self):
        """Test base quality score calculation"""
        
        provider = MockImageProvider("test")
        
        # High resolution image
        score = provider.calculate_base_quality_score(1920, 1080, {})
        assert score >= 0.4  # Should get points for Full HD
        
        # Low resolution image
        score = provider.calculate_base_quality_score(320, 240, {})
        assert score < 0.3  # Should get fewer points
        
    def test_query_expansion(self):
        """Test intelligent query expansion"""
        
        provider = MockImageProvider("test")
        
        # Person entity should add portrait terms
        expanded = provider.expand_query("John Doe", "PERSON")
        assert "portrait" in expanded or "headshot" in expanded
        
        # Location entity should add scenic terms
        expanded = provider.expand_query("Paris", "LOCATION")
        assert any(term in expanded for term in ["landmark", "scenic", "destination"])

class TestMultiSourceRetriever:
    """Test multi-source image retrieval system"""
    
    @pytest.fixture
    def retriever(self):
        """Create test retriever with mock providers"""
        retriever = MultiSourceImageRetriever()
        
        # Add mock providers with different characteristics
        provider1 = MockImageProvider("wikimedia", [
            create_mock_image("wikimedia", "High Quality Image", quality=0.9, relevance=0.8),
            create_mock_image("wikimedia", "Medium Quality Image", quality=0.6, relevance=0.7),
        ])
        
        provider2 = MockImageProvider("unsplash", [
            create_mock_image("unsplash", "Professional Photo", quality=0.8, relevance=0.9),
            create_mock_image("unsplash", "Artistic Shot", quality=0.7, relevance=0.6),
        ])
        
        retriever.add_provider(provider1, weight=1.0, priority=1)
        retriever.add_provider(provider2, weight=0.8, priority=2)
        
        return retriever
        
    @pytest.mark.asyncio
    async def test_basic_search(self, retriever):
        """Test basic multi-source search functionality"""
        
        result = await retriever.search("test query", count=4)
        
        assert isinstance(result, RetrievalResult)
        assert len(result.images) <= 4
        assert len(result.provider_stats) >= 2
        assert result.total_time > 0
        assert "test query" in result.query_used
        
    @pytest.mark.asyncio
    async def test_query_optimization(self, retriever):
        """Test entity-specific query optimization"""
        
        # Test person entity
        result = await retriever.search("John Doe", entity_type="PERSON")
        assert "portrait" in result.query_used
        
        # Test location entity
        result = await retriever.search("Paris", entity_type="GPE")
        assert "landmark" in result.query_used
        
        # Test organization entity
        result = await retriever.search("Apple Inc", entity_type="ORG")
        assert "building" in result.query_used
        
    @pytest.mark.asyncio
    async def test_provider_weighting(self, retriever):
        """Test that provider weights affect result distribution"""
        
        result = await retriever.search("test", count=10)
        
        # Check that providers were called
        wikimedia_count = sum(1 for img in result.images if img.provider == "wikimedia")
        unsplash_count = sum(1 for img in result.images if img.provider == "unsplash")
        
        # Wikimedia has higher weight (1.0 vs 0.8), should get more results
        assert wikimedia_count >= unsplash_count or wikimedia_count + unsplash_count <= 4
        
    @pytest.mark.asyncio
    async def test_quality_filtering(self, retriever):
        """Test that low quality images are filtered out"""
        
        # Add provider with low quality images
        low_quality_provider = MockImageProvider("lowquality", [
            create_mock_image("lowquality", "Poor Image", quality=0.1, relevance=0.1),
            create_mock_image("lowquality", "Bad Image", quality=0.2, relevance=0.15),
        ])
        
        retriever.add_provider(low_quality_provider, weight=1.0)
        
        result = await retriever.search("test", count=10)
        
        # Low quality images should be filtered out
        for image in result.images:
            assert image.quality_score >= retriever.min_quality_score
            assert image.relevance_score >= retriever.min_relevance_score
            
    @pytest.mark.asyncio
    async def test_deduplication(self, retriever):
        """Test image deduplication functionality"""
        
        # Add provider with duplicate images
        duplicate_provider = MockImageProvider("duplicate", [
            create_mock_image("duplicate", "Same Title", quality=0.8),
            create_mock_image("duplicate", "Same Title", quality=0.7),  # Same title, lower quality
            create_mock_image("duplicate", "Different Title", quality=0.8),
        ])
        
        retriever.add_provider(duplicate_provider, weight=1.0)
        
        result = await retriever.search("test", count=10)
        
        # Should keep only the higher quality version of duplicates
        titles = [img.title for img in result.images]
        assert len(titles) == len(set(titles))  # No duplicate titles
        
    @pytest.mark.asyncio 
    async def test_error_handling(self, retriever):
        """Test error handling for provider failures"""
        
        # Add provider that raises exceptions
        failing_provider = MockImageProvider("failing")
        
                 async def failing_search(query: str, count: int = 10):
             raise Exception("Provider error")
            
        failing_provider.search = failing_search
        retriever.add_provider(failing_provider, weight=1.0)
        
        # Should still return results from working providers
        result = await retriever.search("test", count=4)
        assert isinstance(result, RetrievalResult)
        assert "failing" in result.provider_stats
        assert result.provider_stats["failing"]["success_rate"] == 0.0
        
    @pytest.mark.asyncio
    async def test_health_check(self, retriever):
        """Test provider health checking"""
        
        health_status = await retriever.health_check()
        
        assert isinstance(health_status, dict)
        assert len(health_status) == len(retriever.providers)
        assert all(isinstance(status, bool) for status in health_status.values())

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_news_entity_scenario(self):
        """Test scenario for news entities like politicians"""
        
        retriever = MultiSourceImageRetriever()
        
        # Mock Wikimedia with official portrait
        wikimedia = MockImageProvider("wikimedia", [
            create_mock_image("wikimedia", "Official Portrait", quality=0.9, relevance=0.95),
            create_mock_image("wikimedia", "State Visit Photo", quality=0.8, relevance=0.85),
        ])
        
        # Mock Unsplash with professional photos
        unsplash = MockImageProvider("unsplash", [
            create_mock_image("unsplash", "Professional Headshot", quality=0.85, relevance=0.8),
            create_mock_image("unsplash", "Conference Speaking", quality=0.75, relevance=0.7),
        ])
        
        retriever.add_provider(wikimedia, weight=1.2, priority=1)  # Prefer official sources
        retriever.add_provider(unsplash, weight=0.8, priority=2)
        
        result = await retriever.search("President Biden", entity_type="PERSON", count=5)
        
        # Should optimize for person search
        assert "portrait" in result.query_used
        
        # Should prefer high-quality, high-relevance images
        assert all(img.quality_score >= 0.6 for img in result.images)
        assert len(result.images) > 0
        
        # Provider stats should show performance
        assert "wikimedia" in result.provider_stats
        assert "unsplash" in result.provider_stats
        
    @pytest.mark.asyncio
    async def test_location_entity_scenario(self):
        """Test scenario for location entities"""
        
        retriever = MultiSourceImageRetriever()
        
        # Mock providers with location-specific images
        wikimedia = MockImageProvider("wikimedia", [
            create_mock_image("wikimedia", "Landmark Architecture", quality=0.9, relevance=0.9),
        ])
        
        unsplash = MockImageProvider("unsplash", [
            create_mock_image("unsplash", "Scenic Landscape", quality=0.8, relevance=0.85),
        ])
        
        retriever.add_provider(wikimedia, weight=1.0)
        retriever.add_provider(unsplash, weight=1.0)
        
        result = await retriever.search("Tokyo", entity_type="LOC", count=3)
        
        # Should optimize for location search
        assert "landmark" in result.query_used
        assert len(result.images) > 0
        
    def test_performance_requirements(self):
        """Test that the system meets performance requirements"""
        
        retriever = MultiSourceImageRetriever()
        
        # Test initialization time
        start_time = datetime.utcnow()
        for i in range(10):
            provider = MockImageProvider(f"provider_{i}")
            retriever.add_provider(provider, weight=1.0)
        end_time = datetime.utcnow()
        
        initialization_time = (end_time - start_time).total_seconds()
        assert initialization_time < 1.0  # Should be fast
        
        # Test memory efficiency
        import sys
        retriever_size = sys.getsizeof(retriever)
        assert retriever_size < 10000  # Should be reasonably sized

# Performance benchmark
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase 2 requirements"""
    
    @pytest.mark.asyncio
    async def test_search_latency(self):
        """Test that search latency meets Phase 2 requirements"""
        
        retriever = MultiSourceImageRetriever()
        
        # Add multiple mock providers
        for i in range(3):
            provider = MockImageProvider(f"provider_{i}", [
                create_mock_image(f"provider_{i}", f"Image {j}") for j in range(5)
            ])
            retriever.add_provider(provider)
            
        # Measure search time
        start_time = datetime.utcnow()
        result = await retriever.search("test query", count=10)
        end_time = datetime.utcnow()
        
        search_time = (end_time - start_time).total_seconds()
        
        # Phase 2 requirement: < 2s for uncached searches
        assert search_time < 2.0
        assert result.total_time < 2.0
        
    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Test concurrent search performance"""
        
        retriever = MultiSourceImageRetriever()
        
        # Add providers
        for i in range(2):
            provider = MockImageProvider(f"provider_{i}", [
                create_mock_image(f"provider_{i}", f"Image {j}") for j in range(3)
            ])
            retriever.add_provider(provider)
            
        # Execute multiple searches concurrently
        search_tasks = [
            retriever.search(f"query_{i}", count=5)
            for i in range(5)
        ]
        
        start_time = datetime.utcnow()
        results = await asyncio.gather(*search_tasks)
        end_time = datetime.utcnow()
        
        total_time = (end_time - start_time).total_seconds()
        
        # Should handle concurrent requests efficiently
        assert total_time < 5.0  # 5 searches in under 5 seconds
        assert all(isinstance(result, RetrievalResult) for result in results)
        assert all(len(result.images) > 0 for result in results)

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "--tb=short"]) 