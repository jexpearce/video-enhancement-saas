"""
Unified Image Search Interface - Phase 2

Base classes and interfaces for multi-source image API integration
with legal compliance and quality scoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict
import asyncio
from enum import Enum
from datetime import datetime

class ImageLicense(Enum):
    """Image license types for legal compliance"""
    CREATIVE_COMMONS_ZERO = "cc0"
    CREATIVE_COMMONS_BY = "cc-by"
    CREATIVE_COMMONS_BY_SA = "cc-by-sa"
    PUBLIC_DOMAIN = "public_domain"
    COMMERCIAL_ALLOWED = "commercial"
    EDITORIAL_ONLY = "editorial"
    UNKNOWN = "unknown"

@dataclass
class ImageResult:
    """Image search result with comprehensive metadata"""
    provider: str
    image_url: str
    thumbnail_url: str
    title: str
    description: Optional[str]
    author: Optional[str]
    author_url: Optional[str]
    license: ImageLicense
    license_url: Optional[str]
    width: int
    height: int
    relevance_score: float
    quality_score: float
    metadata: Dict  # Provider-specific data
    
    # Additional Phase 2 fields
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    created_at: Optional[datetime] = None
    popularity_score: float = 0.0
    safety_score: float = 1.0  # 1.0 = safe, 0.0 = unsafe
    
    def __post_init__(self):
        """Validate image result data"""
        if not self.image_url or not self.thumbnail_url:
            raise ValueError("Image and thumbnail URLs are required")
        
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid image dimensions")
            
        if not 0 <= self.relevance_score <= 1:
            raise ValueError("Relevance score must be between 0 and 1")
            
        if not 0 <= self.quality_score <= 1:
            raise ValueError("Quality score must be between 0 and 1")

class ImageProvider(ABC):
    """Abstract base class for image providers"""
    
    @abstractmethod
    async def search(self, query: str, count: int = 10) -> List[ImageResult]:
        """
        Search for images matching the query.
        
        Args:
            query: Search query string
            count: Maximum number of results to return
            
        Returns:
            List of ImageResult objects
        """
        pass
    
    @abstractmethod
    async def validate_license(self, image: ImageResult) -> bool:
        """
        Validate that the image license is correctly identified.
        
        Args:
            image: ImageResult to validate
            
        Returns:
            True if license is valid and correctly identified
        """
        pass
    
    @abstractmethod
    async def check_availability(self, image_url: str) -> bool:
        """
        Check if image URL is accessible.
        
        Args:
            image_url: URL to check
            
        Returns:
            True if image is accessible
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification"""
        pass
    
    @property
    @abstractmethod
    def rate_limit(self) -> Dict[str, int]:
        """Rate limiting information"""
        pass
    
    def calculate_base_quality_score(self, width: int, height: int, metadata: Dict) -> float:
        """
        Calculate base quality score from image properties.
        
        Args:
            width: Image width
            height: Image height  
            metadata: Additional metadata
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Resolution scoring
        total_pixels = width * height
        if total_pixels >= 1920 * 1080:  # Full HD+
            score += 0.4
        elif total_pixels >= 1280 * 720:  # HD
            score += 0.3
        elif total_pixels >= 640 * 480:  # SD
            score += 0.2
        else:
            # Lowest resolution - keep score modest to avoid rounding issues
            score += 0.05
            
        # Aspect ratio scoring (prefer standard ratios)
        aspect_ratio = width / height
        standard_ratios = [16/9, 4/3, 3/2, 1/1]
        
        for ratio in standard_ratios:
            if abs(aspect_ratio - ratio) < 0.1:
                score += 0.2
                break
        else:
            score += 0.1  # Non-standard but still usable
            
        # Metadata-based scoring
        if metadata.get('file_size'):
            # Prefer reasonably sized files (not too small, not too large)
            size_mb = metadata['file_size'] / (1024 * 1024)
            if 0.1 <= size_mb <= 10:
                score += 0.2
            elif size_mb <= 0.1:
                score += 0.05  # Too small, likely low quality
            else:
                score += 0.1  # Too large, but usable
                
        # Engagement metrics (if available)
        if metadata.get('likes', 0) > 100:
            score += 0.1
        if metadata.get('downloads', 0) > 50:
            score += 0.1
            
        return min(1.0, score)
    
    def expand_query(self, query: str, entity_type: Optional[str] = None) -> str:
        """
        Intelligently expand search query for better results.
        
        Args:
            query: Base search query
            entity_type: Type of entity (PERSON, LOCATION, etc.)
            
        Returns:
            Expanded query string
        """
        expansions = {
            'PERSON': ['portrait', 'headshot', 'professional'],
            'LOCATION': ['landmark', 'scenic', 'destination', 'aerial'],
            'ORGANIZATION': ['logo', 'headquarters', 'corporate', 'building'],
            'PRODUCT': ['product', 'isolated', 'clean', 'professional'],
            'EVENT': ['ceremony', 'gathering', 'celebration'],
            'CONCEPT': ['abstract', 'conceptual', 'symbolic']
        }
        
        if entity_type and entity_type in expansions:
            # Add 1-2 relevant expansion terms
            expansion_terms = expansions[entity_type][:2]
            return f"{query} {' '.join(expansion_terms)}"
        
        return query
    
    def filter_by_license(self, images: List[ImageResult], 
                         allowed_licenses: List[ImageLicense]) -> List[ImageResult]:
        """
        Filter images by allowed license types.
        
        Args:
            images: List of images to filter
            allowed_licenses: List of allowed license types
            
        Returns:
            Filtered list of images
        """
        return [img for img in images if img.license in allowed_licenses]
    
    def deduplicate_images(self, images: List[ImageResult]) -> List[ImageResult]:
        """
        Remove duplicate images based on URL and visual similarity.
        
        Args:
            images: List of images to deduplicate
            
        Returns:
            Deduplicated list of images
        """
        seen_urls = set()
        unique_images = []
        
        for image in images:
            if image.image_url not in seen_urls:
                seen_urls.add(image.image_url)
                unique_images.append(image)
                
        return unique_images

@dataclass 
class SearchRequest:
    """Comprehensive search request configuration"""
    query: str
    count: int = 10
    entity_type: Optional[str] = None
    allowed_licenses: Optional[List[ImageLicense]] = None
    min_width: int = 400
    min_height: int = 300
    max_file_size_mb: float = 10.0
    orientation: Optional[str] = None  # 'landscape', 'portrait', 'square'
    quality_threshold: float = 0.3
    safety_threshold: float = 0.8
    include_metadata: bool = True
    
    def __post_init__(self):
        """Set default allowed licenses if none specified"""
        if self.allowed_licenses is None:
            self.allowed_licenses = [
                ImageLicense.CREATIVE_COMMONS_ZERO,
                ImageLicense.CREATIVE_COMMONS_BY,
                ImageLicense.PUBLIC_DOMAIN,
                ImageLicense.COMMERCIAL_ALLOWED
            ]

class ProviderError(Exception):
    """Base exception for provider errors"""
    pass

class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded"""
    pass

class LicenseValidationError(ProviderError):
    """Raised when license validation fails"""
    pass

class ImageUnavailableError(ProviderError):
    """Raised when image is not accessible"""
    pass 