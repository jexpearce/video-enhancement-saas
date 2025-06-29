"""
Storage configuration for AWS S3 and CloudFront integration.
"""

from typing import Dict, Tuple, Optional
from pydantic import BaseModel, Field
import os


class StorageConfig(BaseModel):
    """Configuration for AWS S3 and CloudFront storage system."""
    
    # S3 Buckets
    original_bucket: str = Field(
        default_factory=lambda: os.getenv("AWS_S3_ORIGINAL_BUCKET", "video-enhancement-originals")
    )
    processed_bucket: str = Field(
        default_factory=lambda: os.getenv("AWS_S3_PROCESSED_BUCKET", "video-enhancement-processed")
    )
    
    # CloudFront CDN
    cdn_domain: str = Field(
        default_factory=lambda: os.getenv("AWS_CLOUDFRONT_DOMAIN", "cdn.example.com")
    )
    cloudfront_distribution_id: str = Field(
        default_factory=lambda: os.getenv("AWS_CLOUDFRONT_DISTRIBUTION_ID", "")
    )
    
    # AWS Credentials (use AWS credentials chain by default)
    aws_access_key_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID")
    )
    aws_secret_access_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    aws_region: str = Field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1")
    )
    
    # Image processing settings
    image_sizes: Dict[str, Optional[Tuple[int, int]]] = Field(
        default={
            'thumbnail': (320, 180),    # 16:9 thumbnail
            'preview': (640, 360),      # 16:9 preview
            'overlay': (1280, 720),     # 16:9 HD overlay
            'full': None                # Original size
        }
    )
    
    # Quality settings
    webp_quality: int = Field(default=85, ge=1, le=100)
    webp_method: int = Field(default=6, ge=0, le=6)  # 6 = best compression
    
    # Cache settings
    max_file_size_mb: int = Field(default=50)  # Max individual image size
    cache_ttl_hours: int = Field(default=24 * 7)  # 1 week default TTL
    
    class Config:
        env_prefix = "STORAGE_" 