"""
Data models for image storage system.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json


class ImageStorageError(Exception):
    """Custom exception for image storage operations."""
    pass


class StorageStatus(Enum):
    """Status of stored image."""
    UPLOADING = "uploading"
    COMPLETED = "completed" 
    FAILED = "failed"
    PROCESSING = "processing"


@dataclass
class StoredImage:
    """Represents a stored image with multiple sizes and metadata."""
    
    # Unique identifiers
    hash: str                           # SHA256 hash of original image
    entity_id: str                      # Entity this image represents
    
    # Storage locations
    s3_keys: Dict[str, str]            # Size -> S3 key mapping
    cdn_urls: Dict[str, str]           # Size -> CDN URL mapping
    
    # Metadata
    metadata: Dict[str, Any]           # Original metadata from source
    created_at: datetime               # When stored
    updated_at: Optional[datetime] = None
    
    # Storage info
    status: StorageStatus = StorageStatus.COMPLETED
    total_size_bytes: Optional[int] = None
    original_url: Optional[str] = None
    
    # Processing info  
    sizes_generated: Optional[Dict[str, bool]] = None  # Size -> success mapping
    processing_errors: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.sizes_generated is None:
            self.sizes_generated = {}
        if self.processing_errors is None:
            self.processing_errors = {}
    
    def get_url(self, size: str = 'preview') -> Optional[str]:
        """Get CDN URL for specific size."""
        return self.cdn_urls.get(size)
    
    def get_s3_key(self, size: str = 'preview') -> Optional[str]:
        """Get S3 key for specific size."""
        return self.s3_keys.get(size)
    
    def is_size_available(self, size: str) -> bool:
        """Check if specific size was successfully generated."""
        if self.sizes_generated is None:
            return False
        return self.sizes_generated.get(size, False)
    
    def get_available_sizes(self) -> list[str]:
        """Get list of successfully generated sizes."""
        if self.sizes_generated is None:
            return []
        return [size for size, success in self.sizes_generated.items() if success]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'hash': self.hash,
            'entity_id': self.entity_id,
            's3_keys': self.s3_keys,
            'cdn_urls': self.cdn_urls,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'status': self.status.value,
            'total_size_bytes': self.total_size_bytes,
            'original_url': self.original_url,
            'sizes_generated': self.sizes_generated,
            'processing_errors': self.processing_errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredImage':
        """Create instance from dictionary."""
        # Parse datetime fields
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
            
        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])
            
        return cls(
            hash=data['hash'],
            entity_id=data['entity_id'],
            s3_keys=data['s3_keys'],
            cdn_urls=data['cdn_urls'],
            metadata=data['metadata'],
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at,
            status=StorageStatus(data.get('status', 'completed')),
            total_size_bytes=data.get('total_size_bytes'),
            original_url=data.get('original_url'),
            sizes_generated=data.get('sizes_generated', {}),
            processing_errors=data.get('processing_errors', {})
        )


@dataclass
class ImageUploadRequest:
    """Request to upload and process an image."""
    
    image_url: str
    entity_id: str
    metadata: Dict[str, Any]
    priority: int = 1                   # Higher = more priority
    force_reprocess: bool = False       # Force reprocess even if exists
    sizes_requested: Optional[list[str]] = None  # Specific sizes to generate


@dataclass  
class ImageUploadResult:
    """Result of image upload operation."""
    
    stored_image: Optional[StoredImage]
    success: bool
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    cache_hit: bool = False             # Was this served from cache 