"""
Image Processing Service for Video Enhancement SaaS

Handles image optimization, resizing, and preparation for video overlay:
- Image downloading and caching
- Resize and crop for optimal video overlay
- Quality enhancement and compression
- Format conversion and optimization
- Watermark and branding application
"""

import logging
import asyncio
import aiohttp
import hashlib
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import io
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    """Image processing configuration."""
    overlay_width: int = 400            # Width of overlay on video
    overlay_height: int = 300           # Height of overlay on video
    position: str = "top-right"         # Overlay position
    opacity: float = 0.9               # Overlay opacity
    quality: int = 85                   # JPEG quality
    format: str = "PNG"                 # Output format

@dataclass 
class ProcessedImage:
    """Processed image result."""
    original_url: str                   # Original image URL
    processed_path: str                 # Path to processed image
    width: int                          # Processed width
    height: int                         # Processed height
    file_size: int                      # File size in bytes
    processing_time: float              # Processing time in seconds
    cache_key: str                      # Cache identifier
    created_at: datetime                # Creation timestamp

class ImageProcessor:
    """Image processing service for video overlay optimization."""
    
    def __init__(self, cache_dir: str = "./cache/images"):
        """Initialize the image processor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        
    async def process_image(self, image_url: str, options: Optional[ProcessingOptions] = None) -> Optional[ProcessedImage]:
        """Process an image for video overlay."""
        try:
            start_time = datetime.now()
            
            if options is None:
                options = ProcessingOptions()
            
            # Mock processing for now
            cache_key = f"processed_{hash(image_url)}_{options.overlay_width}_{options.overlay_height}"
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            result = ProcessedImage(
                original_url=image_url,
                processed_path=f"{self.cache_dir}/{cache_key}.png",
                width=options.overlay_width,
                height=options.overlay_height, 
                file_size=50000,  # Mock size
                processing_time=(datetime.now() - start_time).total_seconds(),
                cache_key=cache_key,
                created_at=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return None
    
    async def process_multiple_images(self, image_urls: List[str], options: Optional[ProcessingOptions] = None) -> List[ProcessedImage]:
        """
        Process multiple images in parallel.
        
        Args:
            image_urls: List of image URLs to process
            options: Processing options
            
        Returns:
            List of successfully processed images
        """
        
        # Process images in parallel
        tasks = [self.process_image(url, options) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        processed_images = [
            result for result in results 
            if isinstance(result, ProcessedImage)
        ]
        
        logger.info(f"Successfully processed {len(processed_images)}/{len(image_urls)} images")
        return processed_images
    
    def get_preset_options(self, preset_name: str) -> Optional[ProcessingOptions]:
        """Get predefined processing options."""
        return ProcessingOptions(
            overlay_width=400, overlay_height=300,
            position="top-right", opacity=0.9
        )
    
    async def _download_image(self, image_url: str) -> Optional[bytes]:
        """Download image from URL."""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content_length = response.headers.get('content-length')
                        if content_length and int(content_length) > 5 * 1024 * 1024:  # 5MB
                            logger.warning(f"Image too large: {content_length} bytes")
                            return None
                        
                        return await response.read()
                    else:
                        logger.error(f"Failed to download image: HTTP {response.status}")
                        return None
                    
        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")
            return None
    
    def _validate_image_quality(self, image: Image.Image) -> bool:
        """Validate image meets quality requirements."""
        
        width, height = image.size
        min_width, min_height = (400, 300)
        
        # Check minimum resolution
        if width < min_width or height < min_height:
            logger.warning(f"Image resolution too low: {width}x{height}")
            return False
        
        # Check image mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            logger.warning(f"Unsupported image mode: {image.mode}")
            return False
        
        return True
    
    def _process_image_for_overlay(self, image: Image.Image, options: ProcessingOptions) -> Image.Image:
        """Process image for optimal video overlay."""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for overlay
        processed = self._resize_for_overlay(image, options)
        
        # Apply enhancements
        processed = self._enhance_sharpness(processed)
        processed = self._enhance_contrast(processed)
        
        # Add border radius
        processed = self._add_border_radius(processed, 10)
        
        # Add drop shadow
        processed = self._add_drop_shadow(processed)
        
        # Add watermark if requested
        processed = self._add_watermark(processed)
        
        return processed
    
    def _resize_for_overlay(self, image: Image.Image, options: ProcessingOptions) -> Image.Image:
        """Resize image for overlay with smart cropping."""
        
        target_width = options.overlay_width
        target_height = options.overlay_height
        
        # Calculate aspect ratios
        image_ratio = image.width / image.height
        target_ratio = target_width / target_height
        
        if image_ratio > target_ratio:
            # Image is wider, crop width
            new_width = int(image.height * target_ratio)
            new_height = image.height
            left = (image.width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = new_height
        else:
            # Image is taller, crop height
            new_width = image.width
            new_height = int(image.width / target_ratio)
            left = 0
            top = (image.height - new_height) // 2
            right = new_width
            bottom = top + new_height
        
        # Crop and resize
        cropped = image.crop((left, top, right, bottom))
        resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return resized
    
    def _enhance_sharpness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Apply sharpness enhancement."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def _enhance_contrast(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Apply contrast enhancement."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _add_border_radius(self, image: Image.Image, radius: int) -> Image.Image:
        """Add rounded corners to image."""
        
        # Create mask for rounded corners
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), image.size], radius, fill=255)
        
        # Apply mask
        output = Image.new('RGBA', image.size, (0, 0, 0, 0))
        output.paste(image, (0, 0))
        output.putalpha(mask)
        
        return output
    
    def _add_drop_shadow(self, image: Image.Image, offset: Tuple[int, int] = (3, 3), blur: int = 3) -> Image.Image:
        """Add drop shadow to image."""
        
        # Create shadow
        shadow_size = (image.width + offset[0] + blur*2, image.height + offset[1] + blur*2)
        shadow = Image.new('RGBA', shadow_size, (0, 0, 0, 0))
        
        # Draw shadow
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_bbox = (blur, blur, shadow_size[0] - blur, shadow_size[1] - blur)
        shadow_draw.rectangle(shadow_bbox, fill=(0, 0, 0, 80))
        
        # Blur shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
        
        # Composite image on shadow
        result = Image.new('RGBA', shadow_size, (0, 0, 0, 0))
        result.paste(shadow, (0, 0))
        result.paste(image, (blur - offset[0], blur - offset[1]), image if image.mode == 'RGBA' else None)
        
        return result
    
    def _add_watermark(self, image: Image.Image) -> Image.Image:
        """Add subtle watermark to image."""
        
        # Create watermark (simple text for now)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Add watermark text
        watermark_text = "VideoAI"
        text_width, text_height = draw.textsize(watermark_text, font=font)
        
        # Position in bottom right
        x = image.width - text_width - 10
        y = image.height - text_height - 10
        
        # Draw with transparency
        draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))
        
        return image
    
    def _create_thumbnail(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Create thumbnail image."""
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    
    def _save_processed_image(self, image: Image.Image, cache_key: str, format: str) -> Path:
        """Save processed image to disk."""
        
        filename = f"{cache_key}.{format.lower()}"
        file_path = self.cache_dir / filename
        
        save_options = {'quality': 85, 'optimize': True}
        if format.upper() == 'PNG':
            save_options = {'optimize': True}
        
        image.save(file_path, format=format, **save_options)
        return file_path
    
    def _save_thumbnail(self, thumbnail: Image.Image, cache_key: str) -> Path:
        """Save thumbnail to disk."""
        
        filename = f"{cache_key}_thumb.png"
        file_path = self.cache_dir / filename
        thumbnail.save(file_path, format='PNG', optimize=True)
        return file_path
    
    def _calculate_post_processing_quality(self, image: Image.Image) -> float:
        """Calculate quality score for processed image."""
        
        # Mock quality calculation - in production would use ML model
        width, height = image.size
        
        # Resolution score
        resolution_score = min(1.0, (width * height) / (400 * 300))
        
        # Aspect ratio score (prefer 4:3 or 16:9)
        aspect_ratio = width / height
        preferred_ratios = [4/3, 16/9, 1.0]  # 4:3, 16:9, square
        ratio_score = max(1.0 - abs(aspect_ratio - ratio) for ratio in preferred_ratios)
        
        # Overall quality
        quality_score = (resolution_score * 0.6 + ratio_score * 0.4)
        
        return min(1.0, quality_score)
    
    def _generate_cache_key(self, image_url: str, options: ProcessingOptions) -> str:
        """Generate cache key for image and options."""
        
        # Create hash from URL and processing options
        content = f"{image_url}_{options.overlay_width}_{options.overlay_height}_{options.format}_{options.quality}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_disk_cache(self, cache_key: str) -> Optional[ProcessedImage]:
        """Check if processed image exists in disk cache."""
        
        processed_path = self.cache_dir / f"{cache_key}.png"
        thumbnail_path = self.cache_dir / f"{cache_key}_thumb.png"
        
        if processed_path.exists() and thumbnail_path.exists():
            # Check if cache is still valid (not expired)
            file_age = datetime.now() - datetime.fromtimestamp(processed_path.stat().st_mtime)
            if file_age < timedelta(days=7):  # 7 day cache
                # Reconstruct ProcessedImage object
                try:
                    stat = processed_path.stat()
                    
                    # Load image to get dimensions
                    with Image.open(processed_path) as img:
                        width, height = img.size
                    
                    return ProcessedImage(
                        original_url="cached",
                        processed_path=str(processed_path),
                        width=width,
                        height=height,
                        file_size=stat.st_size,
                        processing_time=0.0,
                        cache_key=cache_key,
                        created_at=datetime.fromtimestamp(stat.st_mtime)
                    )
                except Exception as e:
                    logger.warning(f"Error loading cached image: {e}")
        
        return None
    
    def _add_to_memory_cache(self, cache_key: str, result: ProcessedImage):
        """Add result to memory cache with LRU eviction."""
        
        # Remove oldest entries if cache is full
        if len(self.memory_cache) >= 100:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = result
    
    def clear_cache(self, max_age_days: int = 7):
        """Clear expired cache files."""
        
        removed_count = 0
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Error removing cache file {file_path}: {e}")
        
        # Clear memory cache
        self.memory_cache.clear()
        
        logger.info(f"Cleared {removed_count} expired cache files")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        
        total_files = len(list(self.cache_dir.glob("*")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())
        
        return {
            'cache_directory': str(self.cache_dir),
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'memory_cache_size': len(self.memory_cache),
            'max_memory_cache': 100,
            'supported_presets': ['custom']
        } 