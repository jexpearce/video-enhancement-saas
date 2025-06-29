"""
Test suite for Image Storage and CDN Architecture (Days 25-26).
Tests the StorageConfig, ImageStorageManager, and integration workflow.
"""

import asyncio
import io
import pytest
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import os
from datetime import datetime

# Import our storage system
from app.services.images.storage import (
    StorageConfig,
    StoredImage,
    ImageStorageError,
    ImageStorageManager
)
from app.services.images.storage.models import StorageStatus


class TestStorageConfig:
    """Test storage configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StorageConfig()
        
        assert config.original_bucket == "video-enhancement-originals"
        assert config.processed_bucket == "video-enhancement-processed"
        assert config.cdn_domain == "cdn.example.com"
        assert config.webp_quality == 85
        
        # Check image sizes
        assert 'thumbnail' in config.image_sizes
        assert 'preview' in config.image_sizes
        assert config.image_sizes['thumbnail'] == (320, 180)


class TestStoredImageModel:
    """Test StoredImage data model."""
    
    def test_stored_image_creation(self):
        """Test creating a StoredImage instance."""
        stored_image = StoredImage(
            hash="abc123",
            entity_id="person_1",
            s3_keys={'thumbnail': 'entities/person_1/abc123/thumbnail.webp'},
            cdn_urls={'thumbnail': 'https://cdn.example.com/entities/person_1/abc123/thumbnail.webp'},
            metadata={'source': 'unsplash'},
            created_at=datetime.utcnow()
        )
        
        assert stored_image.hash == "abc123"
        assert stored_image.entity_id == "person_1"
        assert stored_image.get_url('thumbnail') == 'https://cdn.example.com/entities/person_1/abc123/thumbnail.webp'


@pytest.fixture
def mock_image_data():
    """Create a test image in memory."""
    img = Image.new('RGB', (100, 100), 'red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture  
def storage_config():
    """Test storage configuration."""
    return StorageConfig(
        original_bucket="test-originals",
        processed_bucket="test-processed", 
        cdn_domain="test-cdn.example.com",
        cloudfront_distribution_id="TEST123456"
    )


class MockS3Client:
    """Mock S3 client for testing."""
    
    def __init__(self):
        self.uploaded_objects = []
    
    def put_object(self, **kwargs):
        """Mock put_object method."""
        self.uploaded_objects.append(kwargs)
        return {'ETag': '"test-etag"'}


class TestImageStorageManager:
    """Test the main ImageStorageManager class."""
    
    @patch('boto3.client')
    def test_init_with_credentials(self, mock_boto_client, storage_config):
        """Test initialization with AWS credentials."""
        mock_s3 = MockS3Client()
        mock_cloudfront = Mock()
        mock_boto_client.side_effect = [mock_s3, mock_cloudfront]
        
        manager = ImageStorageManager(storage_config)
        
        assert manager.config == storage_config
        assert manager.s3_client == mock_s3
    
    def test_generate_hash(self, storage_config, mock_image_data):
        """Test image hash generation."""
        with patch('boto3.client'):
            manager = ImageStorageManager(storage_config)
            
            hash1 = manager._generate_hash(mock_image_data)
            hash2 = manager._generate_hash(mock_image_data)
            
            assert hash1 == hash2  # Same data should produce same hash
            assert len(hash1) == 64  # SHA256 produces 64 char hex string
    
    @patch('boto3.client')
    async def test_process_image_sizes(self, mock_boto_client, storage_config, mock_image_data):
        """Test image processing into multiple sizes."""
        mock_boto_client.return_value = Mock()
        
        manager = ImageStorageManager(storage_config)
        
        processed = await manager._process_image_sizes(mock_image_data)
        
        # Should generate all configured sizes
        assert 'thumbnail' in processed
        assert 'preview' in processed
        assert 'overlay' in processed
        assert 'full' in processed
        
        # All should be bytes (WebP format)
        for size_name, data in processed.items():
            if data is not None:
                assert isinstance(data, bytes)
                assert len(data) > 0


async def run_storage_demo():
    """Demo of the storage system capabilities."""
    print("🚀 Image Storage System Demo")
    print("=" * 50)
    
    config = StorageConfig(
        processed_bucket="my-video-enhancement-bucket",
        cdn_domain="cdn.myapp.com"
    )
    
    print(f"📋 Configuration:")
    print(f"   S3 Bucket: {config.processed_bucket}")
    print(f"   CDN Domain: {config.cdn_domain}")
    print(f"   Image Sizes: {list(config.image_sizes.keys())}")
    print()
    
    print("🎯 Features:")
    print("   ✅ Multi-size image processing")
    print("   ✅ WebP optimization")
    print("   ✅ S3 storage with metadata")
    print("   ✅ CloudFront CDN distribution")
    print("   ✅ Smart cropping")
    print("   ✅ Content deduplication")
    print("   ✅ Error handling")


if __name__ == "__main__":
    print("🧪 Storage System Tests Ready")
    asyncio.run(run_storage_demo()) 