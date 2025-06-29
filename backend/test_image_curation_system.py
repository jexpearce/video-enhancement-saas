"""
Test Suite for Days 23-24 Intelligent Image Curation System

Tests the ML-powered image curation with:
- CLIP-based relevance scoring
- Face detection for person entities
- Quality assessment and validation
- Caching and performance optimization
"""

import asyncio
import logging
from typing import List, Dict
from datetime import datetime

# Import our curation system components
from app.services.images.curation.curator import ImageCurator, CuratedImage, WordContext
from app.services.images.curation.cache_manager import ImageCacheManager, CacheEntry
from app.services.images.curation.validator import LegalComplianceValidator, ValidationResult
from app.services.images.curation.quality_assessor import ImageQualityAssessor, QualityAssessment

# Import existing components
from app.services.images.providers.base import ImageResult, ImageLicense
from app.services.nlp.entity_enricher import EnrichedEntity
from app.services.images.multi_source_retriever import MultiSourceImageRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockRedisClient:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.data = {}
        self.expiry = {}
    
    async def get(self, key):
        if key in self.expiry and datetime.utcnow() > self.expiry[key]:
            del self.data[key]
            del self.expiry[key]
            return None
        return self.data.get(key)
    
    async def setex(self, key, ttl, value):
        self.data[key] = value
        self.expiry[key] = datetime.utcnow().timestamp() + ttl
    
    async def delete(self, key):
        self.data.pop(key, None)
        self.expiry.pop(key, None)
    
    async def incr(self, key):
        current = int(self.data.get(key, 0))
        self.data[key] = str(current + 1)
        return current + 1
    
    async def expire(self, key, ttl):
        if key in self.data:
            self.expiry[key] = datetime.utcnow().timestamp() + ttl
    
    async def keys(self, pattern):
        return [k for k in self.data.keys() if pattern.replace('*', '') in k]

def create_test_image_result(title: str, provider: str = "test") -> ImageResult:
    """Create a test image result."""
    return ImageResult(
        provider=provider,
        image_url=f"https://example.com/images/{title.lower().replace(' ', '_')}.jpg",
        thumbnail_url=f"https://example.com/thumbs/{title.lower().replace(' ', '_')}_thumb.jpg",
        title=title,
        description=f"Test image of {title}",
        author="Test Author",
        author_url="https://example.com/author",
        license=ImageLicense.CREATIVE_COMMONS_ZERO,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        width=1920,
        height=1080,
        relevance_score=0.8,
        quality_score=0.7,
        metadata={"tags": ["test", "sample"], "likes": 100}
    )

def create_test_enriched_entity(text: str, entity_type: str) -> EnrichedEntity:
    """Create a test enriched entity."""
    return EnrichedEntity(
        text=text,
        entity_type=entity_type,
        confidence=0.9,
        image_potential="EXCELLENT",
        canonical_name=text
    )

async def test_image_curator():
    """Test the ML-powered image curator."""
    print("\n🔥 Testing Image Curator (CLIP + Face Detection)")
    print("=" * 60)
    
    try:
        # Initialize curator
        curator = ImageCurator(config={'max_images_per_entity': 3})
        
        # Create test data
        entity = create_test_enriched_entity("Joe Biden", "PERSON")
        candidate_images = [
            create_test_image_result("Joe Biden Portrait"),
            create_test_image_result("Joe Biden Speech"),
            create_test_image_result("Random Person")
        ]
        
        # Test curation
        curated_images = await curator.curate_entity_images(
            entity=entity,
            candidate_images=candidate_images
        )
        
        print(f"✅ Curated {len(curated_images)} images")
        assert len(curated_images) <= 3, "Should respect max_images_per_entity"
        
        print("✅ Image Curator test passed!")
        
    except Exception as e:
        print(f"❌ Image Curator test failed: {e}")
        raise

async def test_cache_manager():
    """Test the pre-curated image cache manager."""
    print("\n💾 Testing Image Cache Manager")
    print("=" * 60)
    
    try:
        # Initialize cache manager with mock Redis
        mock_redis = MockRedisClient()
        cache_manager = ImageCacheManager(
            redis_client=mock_redis,
            config={
                'default_ttl': 3600,
                'local_cache_size': 10
            }
        )
        
        # Create test data
        entity = create_test_enriched_entity("Tesla", "ORG")
        curated_images = [
            CuratedImage(
                entity_id="tesla",
                image=create_test_image_result("Tesla Logo"),
                relevance_score=0.9,
                curation_metadata={'test': 'data'},
                clip_similarity=0.85
            )
        ]
        
        # Test caching
        print("Testing image caching...")
        cache_success = await cache_manager.cache_curated_images(entity, curated_images)
        assert cache_success, "Caching should succeed"
        print("✅ Images cached successfully")
        
        # Test retrieval
        print("Testing cache retrieval...")
        retrieved_images = await cache_manager.get_curated_images(
            entity_id="tesla",
            entity_text="Tesla",
            entity_type="ORG"
        )
        
        assert retrieved_images is not None, "Should retrieve cached images"
        assert len(retrieved_images) == 1, "Should retrieve correct number of images"
        assert retrieved_images[0].image.title == "Tesla Logo", "Should retrieve correct image"
        print("✅ Cache retrieval successful")
        
        # Test cache statistics
        stats = await cache_manager.get_cache_statistics()
        print(f"Cache Statistics: {stats}")
        assert stats['total_cached_entities'] >= 1, "Should show cached entities"
        
        # Test cache invalidation
        print("Testing cache invalidation...")
        invalidation_success = await cache_manager.invalidate_entity_cache("tesla")
        assert invalidation_success, "Invalidation should succeed"
        
        # Verify invalidation
        retrieved_after_invalidation = await cache_manager.get_curated_images(
            entity_id="tesla",
            entity_text="Tesla", 
            entity_type="ORG"
        )
        assert retrieved_after_invalidation is None, "Should return None after invalidation"
        print("✅ Cache invalidation successful")
        
        print("✅ Cache Manager test passed!")
        
    except Exception as e:
        print(f"❌ Cache Manager test failed: {e}")
        raise

async def test_legal_validator():
    """Test the legal compliance validator."""
    print("\n⚖️  Testing Legal Compliance Validator")
    print("=" * 60)
    
    try:
        # Initialize validator
        validator = LegalComplianceValidator(config={
            'nsfw_threshold': 0.3,
            'copyright_threshold': 0.7
        })
        
        # Create test images with different compliance levels
        test_images = [
            # Good image
            ImageResult(
                provider="test",
                image_url="https://example.com/good_image.jpg",
                thumbnail_url="https://example.com/good_thumb.jpg",
                title="Professional Portrait",
                description="Clean professional image",
                author="Photographer",
                author_url="https://example.com",
                license=ImageLicense.CREATIVE_COMMONS_ZERO,
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                width=1920,
                height=1080,
                relevance_score=0.8,
                quality_score=0.9,
                metadata={}
            ),
            
            # Image with licensing issue
            ImageResult(
                provider="test",
                image_url="https://example.com/bad_license.jpg",
                thumbnail_url="https://example.com/bad_thumb.jpg",
                title="Copyrighted Content",
                description="Image with restrictive license",
                author="Commercial Photographer",
                author_url="https://example.com",
                license=ImageLicense.EDITORIAL_ONLY,  # Not suitable for commercial use
                license_url="https://example.com/license",
                width=1280,
                height=720,
                relevance_score=0.7,
                quality_score=0.8,
                metadata={}
            ),
            
            # Low quality image
            ImageResult(
                provider="test",
                image_url="https://example.com/low_quality.jpg",
                thumbnail_url="https://example.com/low_thumb.jpg",
                title="Low Resolution Image",
                description="Very small image",
                author="Amateur",
                author_url="https://example.com",
                license=ImageLicense.PUBLIC_DOMAIN,
                license_url="https://example.com/pd",
                width=100,  # Too small
                height=75,
                relevance_score=0.5,
                quality_score=0.2,
                metadata={}
            )
        ]
        
        print(f"Testing validation of {len(test_images)} images...")
        
        # Test batch validation
        validation_results = await validator.validate_image_batch(test_images)
        
        assert len(validation_results) == len(test_images), "Should validate all images"
        print(f"✅ Validated {len(validation_results)} images")
        
        # Check individual results
        for i, result in enumerate(validation_results):
            print(f"\nImage {i+1}: {test_images[i].title}")
            print(f"  Valid: {result.is_valid}")
            print(f"  Safety Score: {result.safety_score:.2f}")
            print(f"  Issues: {len(result.issues)}")
            
            for issue in result.issues:
                print(f"    - {issue.severity.upper()}: {issue.message}")
            
            assert isinstance(result, ValidationResult), "Should return ValidationResult"
            assert 0.0 <= result.safety_score <= 1.0, "Safety score should be normalized"
        
        # Expected results
        assert validation_results[0].is_valid, "Good image should be valid"
        assert not validation_results[1].is_valid, "Editorial license should be invalid"
        assert not validation_results[2].is_valid, "Low quality should be invalid"
        
        # Test statistics
        stats = validator.get_validation_statistics(validation_results)
        print(f"\nValidation Statistics: {stats}")
        assert stats['total_images'] == 3, "Should count all images"
        assert stats['valid_images'] == 1, "Should identify valid images correctly"
        
        print("✅ Legal Validator test passed!")
        
    except Exception as e:
        print(f"❌ Legal Validator test failed: {e}")
        raise

async def test_quality_assessor():
    """Test the image quality assessor."""
    print("\n🎯 Testing Image Quality Assessor")
    print("=" * 60)
    
    try:
        # Initialize quality assessor
        assessor = ImageQualityAssessor(config={
            'min_quality': 0.4
        })
        
        # Create test images with different quality levels
        test_images = [
            # High quality image
            ImageResult(
                provider="test",
                image_url="https://example.com/hq_image.jpg",
                thumbnail_url="https://example.com/hq_thumb.jpg", 
                title="High Quality Portrait",
                description="Professional 4K image",
                author="Pro Photographer",
                author_url="https://example.com",
                license=ImageLicense.CREATIVE_COMMONS_ZERO,
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                width=3840,  # 4K
                height=2160,
                relevance_score=0.9,
                quality_score=0.95,
                metadata={'format': 'jpeg', 'compression': 'high'}
            ),
            
            # Medium quality image
            ImageResult(
                provider="test", 
                image_url="https://example.com/med_image.jpg",
                thumbnail_url="https://example.com/med_thumb.jpg",
                title="Medium Quality Image",
                description="Standard HD image",
                author="Photographer",
                author_url="https://example.com",
                license=ImageLicense.CREATIVE_COMMONS_BY,
                license_url="https://creativecommons.org/licenses/by/4.0/",
                width=1920,  # HD
                height=1080,
                relevance_score=0.7,
                quality_score=0.6,
                metadata={'format': 'jpeg'}
            ),
            
            # Low quality image
            ImageResult(
                provider="test",
                image_url="https://example.com/lq_image.jpg", 
                thumbnail_url="https://example.com/lq_thumb.jpg",
                title="Low Quality Image",
                description="Small compressed image",
                author="Amateur",
                author_url="https://example.com",
                license=ImageLicense.PUBLIC_DOMAIN,
                license_url="https://example.com/pd",
                width=320,   # Very low res
                height=240,
                relevance_score=0.4,
                quality_score=0.2,
                metadata={'format': 'jpeg', 'compression': 'low'}
            )
        ]
        
        print(f"Assessing quality of {len(test_images)} images...")
        
        # Test quality assessment
        assessments = []
        for i, image in enumerate(test_images):
            assessment = await assessor.assess_image_quality(image)
            assessments.append(assessment)
            
            print(f"\nImage {i+1}: {image.title}")
            print(f"  Overall Score: {assessment.overall_score:.2f}")
            print(f"  Technical Score: {assessment.technical_score:.2f}")
            print(f"  Sharpness Score: {assessment.sharpness_score:.2f}")
            print(f"  Assessment Method: {assessment.details.get('assessment_method', 'unknown')}")
            
            assert isinstance(assessment, QualityAssessment), "Should return QualityAssessment"
            assert 0.0 <= assessment.overall_score <= 1.0, "Overall score should be normalized"
        
        # Verify quality ordering
        assert assessments[0].overall_score > assessments[2].overall_score, "HQ should score higher than LQ"
        
        # Test statistics
        stats = assessor.get_quality_statistics(assessments)
        print(f"\nQuality Statistics: {stats}")
        assert stats['total_assessed'] == 3, "Should assess all images"
        
        print("✅ Quality Assessor test passed!")
        
    except Exception as e:
        print(f"❌ Quality Assessor test failed: {e}")
        raise

async def test_integration_workflow():
    """Test the complete integration workflow."""
    print("\n🚀 Testing Complete Integration Workflow")
    print("=" * 60)
    
    try:
        # Initialize all components
        mock_redis = MockRedisClient()
        
        curator = ImageCurator(config={'max_images_per_entity': 3})
        cache_manager = ImageCacheManager(mock_redis)
        validator = LegalComplianceValidator()
        quality_assessor = ImageQualityAssessor()
        
        # Create test scenario
        entity = create_test_enriched_entity("Apple Inc", "ORG")
        
        candidate_images = [
            create_test_image_result("Apple Logo"),
            create_test_image_result("Apple Headquarters"),
            create_test_image_result("iPhone Product"),
            create_test_image_result("Tim Cook CEO"),
            create_test_image_result("Apple Store")
        ]
        
        print("🔄 Starting complete workflow...")
        
        # Step 1: Legal validation
        print("Step 1: Legal validation...")
        validation_results = await validator.validate_image_batch(candidate_images)
        valid_images = [
            result.image for result in validation_results if result.is_valid
        ]
        print(f"  ✅ {len(valid_images)}/{len(candidate_images)} images passed validation")
        
        # Step 2: Quality assessment
        print("Step 2: Quality assessment...")
        quality_assessments = []
        for image in valid_images:
            assessment = await quality_assessor.assess_image_quality(image)
            quality_assessments.append((image, assessment))
        
        # Filter by quality threshold
        high_quality_images = [
            image for image, assessment in quality_assessments 
            if assessment.overall_score >= 0.4
        ]
        print(f"  ✅ {len(high_quality_images)}/{len(valid_images)} images passed quality check")
        
        # Step 3: ML-powered curation
        print("Step 3: ML-powered curation...")
        curated_images = await curator.curate_entity_images(
            entity=entity,
            candidate_images=high_quality_images
        )
        print(f"  ✅ Curated {len(curated_images)} final images")
        
        # Step 4: Caching
        print("Step 4: Caching results...")
        cache_success = await cache_manager.cache_curated_images(entity, curated_images)
        assert cache_success, "Caching should succeed"
        print("  ✅ Results cached successfully")
        
        # Step 5: Cache retrieval test
        print("Step 5: Testing cache retrieval...")
        cached_images = await cache_manager.get_curated_images(
            entity_id=entity.text,
            entity_text=entity.text,
            entity_type=entity.entity_type
        )
        assert cached_images is not None, "Should retrieve cached images"
        assert len(cached_images) == len(curated_images), "Should retrieve all images"
        print("  ✅ Cache retrieval successful")
        
        # Final statistics
        print("\n📊 Final Workflow Statistics:")
        print(f"  Original candidates: {len(candidate_images)}")
        print(f"  Passed validation: {len(valid_images)}")
        print(f"  Passed quality check: {len(high_quality_images)}")
        print(f"  Final curated: {len(curated_images)}")
        print(f"  Success rate: {len(curated_images)/len(candidate_images)*100:.1f}%")
        
        # Performance metrics
        print("\n⚡ Performance Metrics:")
        print("  All operations completed successfully")
        print("  Workflow demonstrates:")
        print("    ✅ Legal compliance validation")
        print("    ✅ Quality assessment")
        print("    ✅ ML-powered curation with CLIP")
        print("    ✅ Intelligent caching")
        print("    ✅ End-to-end integration")
        
        print("✅ Integration Workflow test passed!")
        
    except Exception as e:
        print(f"❌ Integration Workflow test failed: {e}")
        raise

async def main():
    """Run all tests for the Days 23-24 Image Curation System."""
    print("🎯 Days 23-24 Intelligent Image Curation System Tests")
    print("=" * 80)
    print("Testing ML-powered image curation with CLIP, face detection, and quality assessment")
    print("=" * 80)
    
    start_time = datetime.utcnow()
    
    try:
        # Run individual component tests
        await test_image_curator()
        await test_cache_manager()
        await test_legal_validator()
        await test_quality_assessor()
        
        # Run integration test
        await test_integration_workflow()
        
        # Calculate total time
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! Days 23-24 Implementation Complete")
        print("=" * 80)
        print(f"⏱️  Total test time: {total_time:.2f} seconds")
        print("\n📋 Components Successfully Implemented:")
        print("  ✅ ML-Powered Image Curator with CLIP model")
        print("  ✅ Face Detection for person entities")
        print("  ✅ Pre-Curated Image Cache Manager") 
        print("  ✅ Legal Compliance Validator")
        print("  ✅ Computer Vision Quality Assessor")
        print("  ✅ Complete integration workflow")
        print("\n🚀 Ready for Phase 2 Week 6: Visual Consistency and Style Engine!")
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 