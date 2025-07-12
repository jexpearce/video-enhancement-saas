"""
VideoComposer - Main video composition engine

Integrates with existing sophisticated systems:
- Animation timeline from animation engine
- Style templates from style system  
- Emphasis points from multi-modal emphasis detection
- Curated images from image search and storage
- S3 storage and CDN infrastructure
"""

import asyncio
import tempfile
import logging
import time
import os
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import subprocess

import ffmpeg
import aiohttp
import aiofiles

from .models import (
    CompositionConfig, 
    CompositionResult, 
    CompositionAsset, 
    CompositionTimeline,
    FilterGraph,
    CompositionError,
    FFmpegError,
    AssetError,
    TimelineError
)

# Import existing sophisticated systems  
from ..animation.animation_engine import AnimationEngine
from ..animation.timeline import AnimationTimeline
from ..images.storage.s3_manager import ImageStorageManager
from ..images.styles.style_engine import StyleEngine

logger = logging.getLogger(__name__)

class VideoComposer:
    """
    Production-grade video composition engine using FFmpeg.
    
    Integrates with existing sophisticated systems to produce enhanced videos
    with animated overlays, styled captions, and platform-specific optimizations.
    """
    
    def __init__(self, config: Optional[CompositionConfig] = None):
        """Initialize the video composer with configuration."""
        
        self.config = config or CompositionConfig()
        
        # Initialize connections to existing systems
        self.animation_engine = AnimationEngine()
        self.style_engine = StyleEngine()
        try:
            # Import and initialize StorageConfig for ImageStorageManager
            from ..images.storage.config import StorageConfig
            storage_config = StorageConfig()
            self.storage_manager = ImageStorageManager(storage_config)
        except Exception as e:
            logger.warning(f"Storage manager initialization failed: {e}")
            self.storage_manager = None
        
        # Create temporary directory for composition work
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_composition_"))
        
        # Track assets for cleanup
        self.downloaded_assets: List[str] = []
        
        # Performance tracking
        self.processing_stats = {
            'asset_download_time': 0.0,
            'timeline_processing_time': 0.0,
            'ffmpeg_execution_time': 0.0,
            'total_assets_processed': 0
        }
        
        logger.info(f"VideoComposer initialized with temp dir: {self.temp_dir}")
    
    async def compose_video(
        self, 
        composition_data: Dict[str, Any],
        output_path: str
    ) -> CompositionResult:
        """
        Main composition method that orchestrates the entire rendering pipeline.
        
        Args:
            composition_data: Dictionary containing:
                - input_video_path: Path to original video
                - emphasis_points: Results from emphasis detection
                - entities: Results from entity extraction
                - curated_images: Results from image curation
                - selected_style: Style template from style engine
                - audio_features: Audio analysis results
                - animation_timeline: Pre-built animation timeline (optional)
            output_path: Where to save the final enhanced video
            
        Returns:
            CompositionResult with processing statistics and metadata
        """
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting video composition for output: {output_path}")
            
            # Extract and validate composition data
            input_video = composition_data['input_video_path']
            emphasis_points = composition_data.get('emphasis_points', [])
            curated_images = composition_data.get('curated_images', [])
            selected_style = composition_data.get('selected_style', {})
            audio_features = composition_data.get('audio_features', {})
            
            # Validate input video exists
            if not os.path.exists(input_video):
                raise AssetError(f"Input video not found: {input_video}")
            
            # Get video metadata
            video_metadata = await self._probe_video(input_video)
            logger.info(f"Video metadata: {video_metadata}")
            
            # Stage 1: Process animation timeline
            timeline_start = time.time()
            logger.info(f"🎬 Stage 1: Processing animation timeline with {len(curated_images)} curated images")
            for i, img in enumerate(curated_images[:3]):  # Log first 3 images
                img_type = "ProcessedImage" if hasattr(img, 'original_url') else "Dictionary"
                img_id = getattr(img, 'cache_key', None) if hasattr(img, 'cache_key') else img.get('image_id', img.get('id', 'no_id'))
                logger.info(f"🎬   Image {i}: {img_type}, ID: {img_id}")
            
            composition_timeline = await self._process_animation_timeline(
                composition_data, video_metadata
            )
            self.processing_stats['timeline_processing_time'] = time.time() - timeline_start
            
            logger.info(f"🎬 Animation timeline created with {len(composition_timeline.events)} events")
            for i, event in enumerate(composition_timeline.events[:5]):  # Log first 5 events
                logger.info(f"🎬   Event {i}: {event.get('type')} -> target_id: {event.get('target_id')}")
            
            # Stage 2: Prepare overlay assets
            asset_start = time.time()
            logger.info(f"🎬 Stage 2: Preparing overlay assets from timeline events")
            overlay_assets = await self._prepare_overlay_assets(
                curated_images, composition_timeline
            )
            self.processing_stats['asset_download_time'] = time.time() - asset_start
            self.processing_stats['total_assets_processed'] = len(overlay_assets)
            
            logger.info(f"🎬 Prepared {len(overlay_assets)} overlay assets")
            for i, asset in enumerate(overlay_assets):
                logger.info(f"🎬   Asset {i}: {asset.asset_id} -> {asset.local_path}")
            
            # Stage 3: Generate complex filter graph
            logger.info("🎬 Stage 3: Building FFmpeg filter graph")
            filter_graph = await self._build_filter_graph(
                overlay_assets, composition_timeline, selected_style, video_metadata
            )
            
            logger.info(f"🎬 Filter graph created: {len(filter_graph.filters) if hasattr(filter_graph, 'filters') else 'unknown'} filters")
            
            # Stage 4: Create caption overlays if needed
            caption_filter = None
            if emphasis_points and selected_style.get('show_captions', True):
                caption_filter = await self._create_caption_filter(
                    composition_data.get('transcript', {}),
                    emphasis_points,
                    selected_style
                )
            
            # Stage 5: Execute FFmpeg composition
            ffmpeg_start = time.time()
            ffmpeg_command = await self._execute_composition(
                input_video,
                overlay_assets,
                filter_graph,
                caption_filter,
                output_path
            )
            self.processing_stats['ffmpeg_execution_time'] = time.time() - ffmpeg_start
            
            # Stage 6: Verify and analyze output
            output_stats = await self._analyze_output(output_path)
            
            processing_time = time.time() - start_time
            
            # Create successful result
            result = CompositionResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                composition_config=self.config,
                total_overlays_applied=len(overlay_assets),
                total_effects_applied=len(composition_timeline.effects),
                total_timeline_events=len(composition_timeline.events),
                output_file_size_bytes=output_stats.get('file_size', 0),
                output_duration=output_stats.get('duration', 0),
                output_resolution=output_stats.get('resolution', (0, 0)),
                output_fps=output_stats.get('fps', 0),
                output_bitrate=output_stats.get('bitrate', 0),
                ffmpeg_command=ffmpeg_command,
                assets_used=overlay_assets,
                quality_metrics=await self._calculate_quality_metrics(output_path, video_metadata)
            )
            
            logger.info(f"Video composition completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            error_stage = getattr(e, 'stage', 'unknown')
            
            logger.error(f"Video composition failed: {error_message}")
            
            # Create failed result
            result = CompositionResult(
                success=False,
                output_path=output_path,
                processing_time=processing_time,
                composition_config=self.config,
                error_message=error_message,
                error_stage=error_stage,
                error_details=getattr(e, 'details', {})
            )
            
            return result
            
        finally:
            # Cleanup temporary assets
            await self._cleanup_temp_assets()
    
    async def _process_animation_timeline(
        self,
        composition_data: Dict,
        video_metadata: Dict
    ) -> CompositionTimeline:
        """Process animation timeline using existing animation engine."""
        
        try:
            # Check if we have a pre-built animation timeline
            if 'animation_timeline' in composition_data:
                animation_timeline = composition_data['animation_timeline']
            else:
                # Generate timeline using existing animation engine
                emphasis_points = composition_data.get('emphasis_points', [])
                curated_images = composition_data.get('curated_images', [])
                selected_style = composition_data.get('selected_style', {})
                video_duration = video_metadata.get('duration', 0)
                audio_beats = composition_data.get('audio_features', {}).get('beats', [])
                
                # FIXED: Convert ProcessedImage objects to dictionary format
                ranked_images = []
                for i, img in enumerate(curated_images):
                    try:
                        # Handle ProcessedImage objects vs dictionaries
                        if hasattr(img, 'original_url'):  # ProcessedImage object
                            # Create a consistent image_id that can be matched later
                            img_id = getattr(img, 'cache_key', None) or f"processed_img_{i}"
                            entity_name = getattr(img, 'entity_name', f"entity_{i}")
                            
                            img_dict = {
                                'id': img_id,
                                'image_id': img_id,  # Ensure we have image_id field
                                'entity_id': entity_name,  # Ensure we have entity_id field
                                'url': getattr(img, 'original_url', ''),
                                'local_path': getattr(img, 'processed_path', ''),
                                'entity_name': entity_name,
                                'relevance_score': 0.8,  # Default high score for processed images
                                'quality_score': min(1.0, getattr(img, 'file_size', 0) / 100000),  # File size based quality
                                'width': getattr(img, 'width', 0),
                                'height': getattr(img, 'height', 0),
                                'cache_key': img_id,  # For VideoComposer matching
                                'original_url': getattr(img, 'original_url', ''),  # For VideoComposer matching
                            }
                        else:  # Already a dictionary
                            # Ensure consistent ID fields
                            img_id = img.get('id') or img.get('image_id') or f"dict_img_{i}"
                            entity_name = img.get('entity_name') or img.get('entity_id') or f"entity_{i}"
                            
                            img_dict = {
                                'id': img_id,
                                'image_id': img_id,  # Ensure we have image_id field
                                'entity_id': entity_name,  # Ensure we have entity_id field
                                'url': img.get('url', ''),
                                'local_path': img.get('local_path', ''),
                                'entity_name': entity_name,
                                'relevance_score': img.get('relevance_score', 0.5),
                                'quality_score': img.get('quality_score', 0.5),
                                'width': img.get('width', 0),
                                'height': img.get('height', 0),
                                'cache_key': img_id,  # For VideoComposer matching
                                'original_url': img.get('url', ''),  # For VideoComposer matching
                            }
                        
                        ranked_images.append(img_dict)
                        
                    except Exception as img_error:
                        logger.warning(f"Failed to convert image {i}: {img_error}")
                        # Fallback dictionary with consistent IDs
                        fallback_id = f"fallback_img_{i}"
                        ranked_images.append({
                            'id': fallback_id,
                            'image_id': fallback_id,
                            'entity_id': f"entity_{i}",
                            'url': '',
                            'entity_name': f"entity_{i}",
                            'relevance_score': 0.3,
                            'quality_score': 0.3,
                            'cache_key': fallback_id,
                            'original_url': '',
                        })
                
                logger.info(f"🎬 Converted {len(ranked_images)} images for animation timeline")
                
                # DETAILED DEBUG: Show what image IDs we're passing to AnimationEngine
                for i, img_dict in enumerate(ranked_images[:5]):  # Show first 5
                    logger.info(f"🎬   Ranked Image {i}: image_id='{img_dict.get('image_id')}', entity_id='{img_dict.get('entity_id')}', cache_key='{img_dict.get('cache_key')}'")
                
                # Generate animation timeline
                timeline_result = await self.animation_engine.create_image_animation_timeline(
                    emphasis_points=emphasis_points,
                    ranked_images=ranked_images,
                    style=selected_style,
                    video_duration=video_duration,
                    audio_beats=audio_beats
                )
                
                animation_timeline = timeline_result
            
            # Convert to composition timeline format
            composition_timeline = CompositionTimeline(
                duration=video_metadata.get('duration', 0),
                events=animation_timeline.get('events', [])
            )
            
            # Process timeline events to extract overlays and effects
            for event in composition_timeline.events:
                if event.get('type') == 'image_entry':
                    # Find corresponding image in curated_images
                    image_id = event.get('target_id')
                    curated_images = composition_data.get('curated_images', [])
                    
                    matching_image = None
                    for img in curated_images:
                        # Handle both ProcessedImage objects and dictionaries
                        if hasattr(img, 'cache_key'):  # ProcessedImage object
                            if getattr(img, 'cache_key', '') == image_id:
                                matching_image = img
                                break
                        else:  # Dictionary
                            if img.get('id') == image_id or img.get('entity_name') == image_id:
                                matching_image = img
                                break
                    
                    if matching_image:
                        # Will be processed in _prepare_overlay_assets
                        pass
            
            logger.info(f"Processed animation timeline with {len(composition_timeline.events)} events")
            return composition_timeline
            
        except Exception as e:
            import traceback
            logger.error(f"Timeline processing error: {e}")
            logger.error(f"Timeline traceback: {traceback.format_exc()}")
            raise TimelineError(f"Failed to process animation timeline: {e}")
    
    async def _prepare_overlay_assets(
        self,
        curated_images: List[Dict],
        composition_timeline: CompositionTimeline
    ) -> List[CompositionAsset]:
        """Download and prepare images for overlay based on timeline events.
        FIXED: Now properly matches images to events using entity information.
        """
        
        overlay_assets = []
        used_images = set()  # Track which images have been used
        
        try:
            # CRITICAL FIX: Build entity-to-images mapping first
            entity_image_map = {}
            
            for img in curated_images:
                # Extract entity identifier from various possible fields
                entity_id = None
                
                if hasattr(img, 'entity_name'):  # ProcessedImage object
                    entity_id = getattr(img, 'entity_name', '').lower()
                else:  # Dictionary
                    entity_id = (
                        img.get('entity_name', '') or 
                        img.get('entity_id', '') or
                        img.get('entity', '')
                    ).lower()
                
                if entity_id:
                    if entity_id not in entity_image_map:
                        entity_image_map[entity_id] = []
                    entity_image_map[entity_id].append(img)
                    
                    logger.debug(f"🗂️ Mapped image to entity '{entity_id}'")
            
            logger.info(f"🗂️ Created entity map with {len(entity_image_map)} entities and {len(curated_images)} total images")
            
            # Process timeline events
            for event in composition_timeline.events:
                if event.get('type') != 'image_entry':
                    continue
                
                # Extract entity from target_id
                target_id = event.get('target_id', '')
                entity_id = target_id.lower()  # Normalize for matching
                
                logger.debug(f"🎯 Processing event for entity: '{entity_id}' at {event.get('start_time', 0):.2f}s")
                
                # Find matching images for this entity
                matching_images = entity_image_map.get(entity_id, [])
                
                # If no direct match, try partial matching
                if not matching_images:
                    for mapped_entity, images in entity_image_map.items():
                        if entity_id in mapped_entity or mapped_entity in entity_id:
                            matching_images = images
                            logger.debug(f"🎯 Found partial match: '{entity_id}' ~ '{mapped_entity}'")
                            break
                
                # Select an unused image
                selected_image = None
                for img in matching_images:
                    # Create unique image identifier
                    if hasattr(img, 'cache_key'):
                        img_id = getattr(img, 'cache_key')
                    else:
                        img_id = img.get('image_id') or img.get('id') or img.get('url', '')
                    
                    if img_id not in used_images:
                        selected_image = img
                        used_images.add(img_id)
                        logger.debug(f"✅ Selected unused image for '{entity_id}': {img_id}")
                        break
                
                # If all images for this entity are used, reuse the first one
                if not selected_image and matching_images:
                    selected_image = matching_images[0]
                    logger.debug(f"♻️ Reusing image for '{entity_id}' (all images used)")
                
                # Last resort: use any available image
                if not selected_image and curated_images:
                    for img in curated_images:
                        if hasattr(img, 'cache_key'):
                            img_id = getattr(img, 'cache_key')
                        else:
                            img_id = img.get('image_id') or img.get('id') or img.get('url', '')
                        
                        if img_id not in used_images:
                            selected_image = img
                            used_images.add(img_id)
                            logger.warning(f"⚠️ Using any available image for '{entity_id}'")
                            break
                
                if not selected_image:
                    logger.error(f"❌ No image available for entity '{entity_id}'")
                    continue
                
                # Download image to local temp file
                local_path = await self._download_image_asset(selected_image)
                
                if not local_path:
                    logger.warning(f"Failed to download image for entity '{entity_id}'")
                    continue
                
                # Extract metadata
                if hasattr(selected_image, 'original_url'):  # ProcessedImage object
                    original_url = getattr(selected_image, 'original_url', '')
                    entity_name = getattr(selected_image, 'entity_name', entity_id)
                    relevance_score = 0.8
                    quality_score = min(1.0, getattr(selected_image, 'file_size', 0) / 100000)
                else:  # Dictionary
                    original_url = selected_image.get('url', '')
                    entity_name = selected_image.get('entity_name', entity_id)
                    relevance_score = selected_image.get('relevance_score', 0.7)
                    quality_score = selected_image.get('quality_score', 0.7)
                
                # Create composition asset with correct timing
                asset = CompositionAsset(
                    asset_id=str(target_id),  # Keep original target_id for matching
                    asset_type='image',
                    local_path=local_path,
                    original_url=original_url,
                    start_time=event.get('start_time', 0),
                    duration=event.get('duration', self.config.overlay_duration_default),
                    position=event.get('properties', {}).get('position', 'top-right'),
                    size=event.get('properties', {}).get('size', 'medium'),
                    animation_type=event.get('properties', {}).get('animation', 'fade'),
                    effects=self._extract_effects_for_timerange(
                        composition_timeline,
                        event.get('start_time', 0),
                        event.get('start_time', 0) + event.get('duration', 0)
                    ),
                    metadata={
                        'entity_name': entity_name,
                        'relevance_score': relevance_score,
                        'quality_score': quality_score
                    }
                )
                
                overlay_assets.append(asset)
                logger.info(f"✅ Prepared overlay for '{entity_name}' at {asset.start_time:.2f}s")
            
            logger.info(f"📦 Prepared {len(overlay_assets)} overlay assets")
            
            # Log final asset timeline for debugging
            for i, asset in enumerate(overlay_assets):
                logger.info(f"  Asset {i}: {asset.metadata.get('entity_name')} at {asset.start_time:.2f}s for {asset.duration:.1f}s")
            
            return overlay_assets
            
        except Exception as e:
            logger.error(f"Failed to prepare overlay assets: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise AssetError(f"Failed to prepare overlay assets: {e}")
    
    async def _download_image_asset(self, image_data: Dict) -> Optional[str]:
        """Download image from URL to local temp file."""
        
        try:
            # FIXED: Handle both ProcessedImage objects and dictionaries
            if hasattr(image_data, 'processed_path'):  # ProcessedImage object
                # Check if we already have a local path
                local_path = getattr(image_data, 'processed_path', None)
                if local_path and os.path.exists(local_path):
                    return local_path
                
                # Get image URL from ProcessedImage object
                image_url = getattr(image_data, 'original_url', '')
                
            else:  # Dictionary
                # Check if we already have a local path
                local_path = image_data.get('local_path') or image_data.get('processed_path')
                if local_path and os.path.exists(local_path):
                    logger.debug(f"Using existing local path: {local_path}")
                    return local_path
                
                # FIXED: Get image URL (comprehensive field checking)
                image_url = (
                    image_data.get('original_url') or
                    image_data.get('cdn_url') or 
                    image_data.get('url') or 
                    image_data.get('image_url') or
                    image_data.get('thumbnail_url')
                )
                
                # CRITICAL DEBUG: Log what we're looking for
                logger.debug(f"🔍 Looking for image URL in: {list(image_data.keys())}")
                logger.debug(f"🔍 Found image_url: {image_url}")
            
            if not image_url:
                logger.warning(f"No URL found in image data keys: {list(image_data.keys()) if isinstance(image_data, dict) else 'Not a dict'}")
                return None
            
            # Generate local filename
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
            file_extension = self._get_file_extension(image_url) or '.jpg'
            local_filename = f"overlay_{url_hash}{file_extension}"
            local_path = self.temp_dir / local_filename
            
            # Download image
            logger.debug(f"Downloading image from: {image_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(local_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        
                        # Track for cleanup
                        self.downloaded_assets.append(str(local_path))
                        
                        logger.debug(f"Downloaded image to: {local_path}")
                        return str(local_path)
                    else:
                        logger.warning(f"Failed to download image, status: {response.status}")
                        return None
            
        except Exception as e:
            logger.error(f"Error downloading image asset: {e}")
            return None
    
    def _get_file_extension(self, url: str) -> Optional[str]:
        """Extract file extension from URL."""
        try:
            path = url.split('?')[0]  # Remove query parameters
            if '.' in path:
                return '.' + path.split('.')[-1].lower()
        except:
            pass
        return None

    def _sanitize_label(self, value: str) -> str:
        """Sanitize a string for use as an FFmpeg label."""
        import re
        return re.sub(r"[^A-Za-z0-9]", "", value)
    
    async def _build_filter_graph(
        self,
        overlay_assets: List[CompositionAsset],
        composition_timeline: CompositionTimeline,
        style: Dict,
        video_metadata: Dict
    ) -> FilterGraph:
        """Build complex FFmpeg filter graph for all overlays and effects."""
        
        filter_graph = FilterGraph()
        
        if not overlay_assets:
            # No overlays, return empty filter graph
            logger.info("No overlay assets, returning empty filter graph")
            return filter_graph
        
        # CRITICAL FIX: Build proper filter chain
        all_filter_parts = []
        current_video_label = "[0:v]"
        
        # Add each overlay with its animation
        for i, asset in enumerate(overlay_assets):
            # Calculate overlay position
            x, y = self._calculate_overlay_position(
                asset.position,
                asset.size,
                self.config.output_resolution
            )
            
            # Input label for this overlay image
            overlay_input = f"[{i + 1}:v]"
            
            # Output label for this stage
            if i < len(overlay_assets) - 1:
                next_video_label = f"[stage{i}]"
            else:
                next_video_label = "[outv]"
            
            # Get size in pixels
            size_pixels = self._get_size_pixels(asset.size)
            safe_id = self._sanitize_label(asset.asset_id)
            temp_label = f"scaled_{safe_id}"
            
            # Build filter parts for this overlay
            scale_filter = f"{overlay_input}scale={size_pixels}:-1[{temp_label}]"
            overlay_filter = f"{current_video_label}[{temp_label}]overlay={x}:{y}:enable='between(t,{asset.start_time},{asset.start_time + asset.duration})'{next_video_label}"
            
            # Add to filter parts
            all_filter_parts.append(scale_filter)
            all_filter_parts.append(overlay_filter)
            
            # Update current label for next iteration
            current_video_label = next_video_label
        
        # Join all filter parts and add to filter graph
        if all_filter_parts:
            complete_filter = ";".join(all_filter_parts)
            filter_graph.add_filter(complete_filter)
            logger.info(f"Built filter graph with {len(overlay_assets)} overlays")
            logger.debug(f"Filter graph: {complete_filter}")
        
        return filter_graph
    
    def _create_overlay_filter(
        self,
        base_input: str,
        overlay_input: str,
        asset: CompositionAsset,
        x: int,
        y: int,
        style: Dict
    ) -> str:
        """Create FFmpeg overlay filter with animation for a single asset."""
        
        # Get size in pixels
        size_pixels = self._get_size_pixels(asset.size)
        
        # CRITICAL FIX: Build proper FFmpeg filter chain
        safe_id = self._sanitize_label(asset.asset_id)
        temp_label = f"scaled_{safe_id}"
        
        # Step 1: Scale the overlay image
        scale_filter = f"{overlay_input}scale={size_pixels}:-1[{temp_label}]"
        
        # Step 2: Apply the overlay with timing
        overlay_filter = f"{base_input}[{temp_label}]overlay={x}:{y}:enable='between(t,{asset.start_time},{asset.start_time + asset.duration})'"
        
        # FIXED: Return proper filter chain format
        return f"{scale_filter};{overlay_filter}"
    
    def _calculate_overlay_position(
        self,
        position: str,
        size: str,
        video_resolution: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate pixel position for overlay based on position string and size."""
        
        video_width, video_height = video_resolution
        overlay_size = self._get_size_pixels(size)
        
        # Estimate overlay dimensions (assume 4:3 aspect ratio)
        overlay_width = overlay_size
        overlay_height = int(overlay_size * 0.75)
        
        # Position mappings with padding
        padding = 20
        
        positions = {
            'top-left': (padding, padding),
            'top-center': ((video_width - overlay_width) // 2, padding),
            'top-right': (video_width - overlay_width - padding, padding),
            'center-left': (padding, (video_height - overlay_height) // 2),
            'center': ((video_width - overlay_width) // 2, (video_height - overlay_height) // 2),
            'center-right': (video_width - overlay_width - padding, (video_height - overlay_height) // 2),
            'bottom-left': (padding, video_height - overlay_height - padding),
            'bottom-center': ((video_width - overlay_width) // 2, video_height - overlay_height - padding),
            'bottom-right': (video_width - overlay_width - padding, video_height - overlay_height - padding),
        }
        
        return positions.get(position, positions['top-right'])
    
    def _get_size_pixels(self, size: str) -> int:
        """Convert size string to pixel value."""
        size_map = {
            'small': 200,
            'medium': 350,
            'large': 500
        }
        return size_map.get(size, 350)
    
    def _extract_effects_for_timerange(
        self,
        timeline: CompositionTimeline,
        start_time: float,
        end_time: float
    ) -> List[str]:
        """Extract effects that apply during a specific time range."""
        
        effects = []
        
        for event in timeline.events:
            event_start = event.get('start_time', 0)
            event_duration = event.get('duration', 0)
            event_end = event_start + event_duration
            
            # Check if event overlaps with our time range
            if event_start <= end_time and event_end >= start_time:
                event_type = event.get('type', '')
                
                if event_type == 'ken_burns':
                    effects.append('ken_burns')
                elif event_type == 'pulse_effect':
                    effects.append('pulse')
                elif event_type == 'zoom_burst':
                    effects.append('zoom_burst')
        
        return effects
    
    async def _create_caption_filter(
        self,
        transcript: Dict,
        emphasis_points: List[Dict],
        style: Dict
    ) -> Optional[str]:
        """Create caption filter using ASS subtitles for advanced styling."""
        
        try:
            if not transcript or not emphasis_points:
                return None
            
            # Generate ASS subtitle content
            ass_content = self._generate_ass_subtitles(transcript, emphasis_points, style)
            
            # Save to temp file
            subtitle_path = self.temp_dir / "captions.ass"
            async with aiofiles.open(subtitle_path, 'w', encoding='utf-8') as f:
                await f.write(ass_content)
            
            # Track for cleanup
            self.downloaded_assets.append(str(subtitle_path))
            
            # FIXED: Return just the path, not "ass=" prefix to avoid over-escaping
            return str(subtitle_path)
            
        except Exception as e:
            logger.warning(f"Failed to create caption filter: {e}")
            return None
    
    def _generate_ass_subtitles(
        self,
        transcript: Dict,
        emphasis_points: List[Dict],
        style: Dict
    ) -> str:
        """Generate ASS subtitle content with styling for emphasis."""
        
        # ASS header with style definitions
        font_name = style.get('caption_font', 'Arial')
        font_size = style.get('caption_size', 24)
        
        ass_header = f"""[Script Info]
Title: Video Enhancement Captions
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1
Style: Emphasis,{font_name},{int(font_size * 1.2)},&H00FFFF00,&H00FFFF00,&H00000000,&H80000000,1,0,0,0,110,110,0,0,1,3,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Convert emphasis points to subtitle events
        events = []
        for point in emphasis_points:
            word = point.get('word', '')
            start_time = point.get('start_time', 0)
            end_time = point.get('end_time', start_time + 1)
            
            # Format timestamps for ASS
            start_ass = self._format_ass_time(start_time)
            end_ass = self._format_ass_time(end_time)
            
            # Determine if emphasized
            is_emphasized = point.get('is_emphasized', False) or point.get('emphasis_score', 0) > 0.6
            style_name = "Emphasis" if is_emphasized else "Default"
            
            events.append(f"Dialogue: 0,{start_ass},{end_ass},{style_name},,0,0,0,,{word}")
        
        return ass_header + "\n".join(events)
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format time for ASS subtitles (h:mm:ss.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    
    async def _execute_composition(
        self,
        input_video: str,
        overlay_assets: List[CompositionAsset],
        filter_graph: FilterGraph,
        caption_filter: Optional[str],
        output_path: str
    ) -> str:
        """Execute FFmpeg composition command."""
        
        try:
            cmd = ['ffmpeg', '-y', '-i', input_video]
            for asset in overlay_assets:
                cmd.extend(['-i', asset.local_path])

            filter_complex = filter_graph.build()
            final_video_label = 'outv'

            if overlay_assets and filter_complex:
                logger.info(f"Scenario 1: Using complex filter graph with {len(overlay_assets)} overlays")
                logger.debug(f"Filter complex: {filter_complex}")

                if caption_filter and os.path.exists(caption_filter):
                    escaped = caption_filter.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")
                    filter_complex += f";[outv]subtitles='{escaped}':force_style=1[finalv]"
                    final_video_label = 'finalv'
                    logger.info("Added captions to filter complex")

                cmd += ['-filter_complex', filter_complex, '-map', f'[{final_video_label}]', '-map', '0:a']

            elif caption_filter and os.path.exists(caption_filter):
                logger.info("Scenario 2: No overlays, captions only")
                cmd += [
                    '-vf', f"subtitles={caption_filter}:force_style=1,format=yuv420p",
                    '-map', '0:v', '-map', '0:a'
                ]
            else:
                logger.info("Scenario 3: Simple pass-through")
                cmd += ['-map', '0:v', '-map', '0:a']

            cmd += [
                '-c:v', self.config.output_codec,
                '-c:a', self.config.audio_codec,
                '-b:v', self.config.output_bitrate,
                '-b:a', self.config.audio_bitrate,
                '-preset', self.config.preset,
                '-crf', str(self.config.crf),
                '-threads', str(self.config.threads),
                '-movflags', 'faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            command_str = ' '.join(cmd)
            logger.debug(f"FFmpeg command: {command_str}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"FFmpeg failed with code {process.returncode}")
                logger.error(stderr.decode())
                raise FFmpegError("FFmpeg execution failed", command=command_str, stderr=stderr.decode())

            logger.info("FFmpeg composition completed successfully")
            return command_str
            
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
            logger.error(f"FFmpeg execution failed: {stderr}")
            raise FFmpegError(f"FFmpeg execution failed: {stderr}", stderr=stderr)
        
        except Exception as e:
            logger.error(f"Composition execution failed: {e}")
            raise CompositionError(f"Composition execution failed: {e}", "ffmpeg_execution")
    
    async def _probe_video(self, video_path: str) -> Dict[str, Any]:
        """Probe video file for metadata."""
        
        try:
            probe = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ffmpeg.probe(video_path)
            )
            
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            metadata = {
                'duration': float(probe['format'].get('duration', 0)),
                'size': int(probe['format'].get('size', 0)),
                'bitrate': int(probe['format'].get('bit_rate', 0)),
            }
            
            if video_stream:
                # CRITICAL FIX: Safe frame rate parsing instead of dangerous eval()
                fps_string = video_stream.get('r_frame_rate', '30/1')
                try:
                    if '/' in fps_string:
                        numerator, denominator = fps_string.split('/')
                        fps = float(numerator) / float(denominator)
                    else:
                        fps = float(fps_string)
                except (ValueError, ZeroDivisionError):
                    fps = 30.0
                
                metadata.update({
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': fps,
                    'video_codec': video_stream.get('codec_name', 'unknown')
                })
            
            if audio_stream:
                metadata.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0))
                })
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Video probe failed: {e}")
            return {'duration': 0, 'fps': 30, 'width': 1080, 'height': 1920}
    
    async def _analyze_output(self, output_path: str) -> Dict[str, Any]:
        """Analyze the output video file."""
        
        try:
            if not os.path.exists(output_path):
                return {}
            
            probe = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ffmpeg.probe(output_path)
            )
            
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            
            stats = {
                'file_size': os.path.getsize(output_path),
                'duration': float(probe['format'].get('duration', 0)),
                'bitrate': int(probe['format'].get('bit_rate', 0))
            }
            
            if video_stream:
                fps_string = video_stream.get('r_frame_rate', '30/1')
                try:
                    if '/' in fps_string:
                        numerator, denominator = fps_string.split('/')
                        fps = float(numerator) / float(denominator)
                    else:
                        fps = float(fps_string)
                except (ValueError, ZeroDivisionError):
                    fps = 30.0

                stats.update({
                    'resolution': (int(video_stream.get('width', 0)), int(video_stream.get('height', 0))),
                    'fps': fps
                })
            
            return stats
            
        except Exception as e:
            logger.warning(f"Output analysis failed: {e}")
            return {}
    
    async def _calculate_quality_metrics(
        self,
        output_path: str,
        original_metadata: Dict
    ) -> Dict[str, float]:
        """Calculate quality metrics for the output video."""
        
        try:
            output_stats = await self._analyze_output(output_path)
            
            metrics = {}
            
            # File size efficiency
            if original_metadata.get('size') and output_stats.get('file_size'):
                size_ratio = output_stats['file_size'] / original_metadata['size']
                metrics['size_efficiency'] = min(1.0, 1.0 / size_ratio) if size_ratio > 0 else 0
            
            # Resolution preservation
            orig_width = original_metadata.get('width', 0)
            orig_height = original_metadata.get('height', 0)
            out_width, out_height = output_stats.get('resolution', (0, 0))
            
            if orig_width and orig_height and out_width and out_height:
                resolution_ratio = (out_width * out_height) / (orig_width * orig_height)
                metrics['resolution_preservation'] = min(1.0, resolution_ratio)
            
            # Processing efficiency
            if hasattr(self, 'processing_stats'):
                total_time = self.processing_stats.get('ffmpeg_execution_time', 0)
                duration = output_stats.get('duration', 1)
                if duration > 0:
                    metrics['processing_efficiency'] = duration / max(total_time, 0.1)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            return {}
    
    async def _cleanup_temp_assets(self):
        """Clean up temporary assets and directories."""
        
        try:
            # Remove downloaded assets
            for asset_path in self.downloaded_assets:
                try:
                    if os.path.exists(asset_path):
                        os.remove(asset_path)
                        logger.debug(f"Cleaned up asset: {asset_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup asset {asset_path}: {e}")
            
            # Remove temp directory
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
                
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except:
                pass 