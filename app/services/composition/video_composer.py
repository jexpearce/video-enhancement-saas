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
from ..animation.timeline import Timeline as AnimationTimeline
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
            self.storage_manager = ImageStorageManager()
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
            composition_timeline = await self._process_animation_timeline(
                composition_data, video_metadata
            )
            self.processing_stats['timeline_processing_time'] = time.time() - timeline_start
            
            # Stage 2: Prepare overlay assets
            asset_start = time.time()
            overlay_assets = await self._prepare_overlay_assets(
                curated_images, composition_timeline
            )
            self.processing_stats['asset_download_time'] = time.time() - asset_start
            self.processing_stats['total_assets_processed'] = len(overlay_assets)
            
            # Stage 3: Generate complex filter graph
            logger.info("Building FFmpeg filter graph")
            filter_graph = await self._build_filter_graph(
                overlay_assets, composition_timeline, selected_style, video_metadata
            )
            
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
                
                # Create ranked images format expected by animation engine
                ranked_images = [
                    {
                        'id': img.get('id', f"img_{i}"),
                        'url': img.get('url', ''),
                        'entity_name': img.get('entity_name', ''),
                        'relevance_score': img.get('relevance_score', 0.5),
                        'quality_score': img.get('quality_score', 0.5)
                    }
                    for i, img in enumerate(curated_images)
                ]
                
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
                        if img.get('id') == image_id or img.get('entity_name') == image_id:
                            matching_image = img
                            break
                    
                    if matching_image:
                        # Will be processed in _prepare_overlay_assets
                        pass
            
            logger.info(f"Processed animation timeline with {len(composition_timeline.events)} events")
            return composition_timeline
            
        except Exception as e:
            raise TimelineError(f"Failed to process animation timeline: {e}")
    
    async def _prepare_overlay_assets(
        self,
        curated_images: List[Dict],
        composition_timeline: CompositionTimeline
    ) -> List[CompositionAsset]:
        """Download and prepare images for overlay based on timeline events."""
        
        overlay_assets = []
        
        try:
            for event in composition_timeline.events:
                if event.get('type') != 'image_entry':
                    continue
                
                # Find corresponding curated image
                image_id = event.get('target_id')
                matching_image = None
                
                for img in curated_images:
                    # Try multiple ID matching strategies
                    if (img.get('id') == image_id or 
                        img.get('entity_name') == image_id or
                        img.get('title', '').lower() == str(image_id).lower()):
                        matching_image = img
                        break
                
                if not matching_image:
                    logger.warning(f"No matching image found for ID: {image_id}")
                    continue
                
                # Download image to local temp file
                local_path = await self._download_image_asset(matching_image)
                
                if not local_path:
                    logger.warning(f"Failed to download image: {matching_image}")
                    continue
                
                # Create composition asset
                asset = CompositionAsset(
                    asset_id=str(image_id),
                    asset_type='image',
                    local_path=local_path,
                    original_url=matching_image.get('url', ''),
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
                        'entity_name': matching_image.get('entity_name', ''),
                        'relevance_score': matching_image.get('relevance_score', 0),
                        'quality_score': matching_image.get('quality_score', 0)
                    }
                )
                
                overlay_assets.append(asset)
                logger.debug(f"Prepared overlay asset: {asset.asset_id}")
            
            logger.info(f"Prepared {len(overlay_assets)} overlay assets")
            return overlay_assets
            
        except Exception as e:
            raise AssetError(f"Failed to prepare overlay assets: {e}")
    
    async def _download_image_asset(self, image_data: Dict) -> Optional[str]:
        """Download image from URL to local temp file."""
        
        try:
            # Check if we already have a local path
            if 'local_path' in image_data and os.path.exists(image_data['local_path']):
                return image_data['local_path']
            
            # Get image URL (try multiple possible keys)
            image_url = (
                image_data.get('cdn_url') or 
                image_data.get('url') or 
                image_data.get('image_url') or
                image_data.get('thumbnail_url')
            )
            
            if not image_url:
                logger.warning(f"No URL found in image data: {image_data}")
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
            # No overlays, just pass through
            filter_graph.add_filter("[0:v]format=yuv420p[outv]")
            return filter_graph
        
        # Start with base video
        current_label = "[0:v]"
        
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
            
            # Create overlay filter with animation
            overlay_filter = self._create_overlay_filter(
                current_label,
                overlay_input,
                asset,
                x,
                y,
                style
            )
            
            # Output label for this stage
            stage_output = f"[stage{i}]" if i < len(overlay_assets) - 1 else "[outv]"
            
            filter_graph.add_filter(f"{overlay_filter}{stage_output}")
            current_label = stage_output
        
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
        
        # Base overlay positioning
        overlay_expr = f"overlay=x={x}:y={y}"
        
        # Add timing constraints
        overlay_expr += f":enable='between(t,{asset.start_time},{asset.start_time + asset.duration})'"
        
        # Scale overlay image first
        size_pixels = self._get_size_pixels(asset.size)
        scale_filter = f"{overlay_input}scale={size_pixels}:-1"
        
        # Add animation effects
        if asset.animation_type == 'fade':
            # Fade in/out animation
            transition_duration = style.get('image_transition_duration', 0.5)
            scale_filter += f",fade=t=in:st={asset.start_time}:d={transition_duration}"
            scale_filter += f",fade=t=out:st={asset.start_time + asset.duration - transition_duration}:d={transition_duration}"
        
        elif asset.animation_type == 'slide':
            # Slide animation - modify x position over time
            slide_distance = 100
            start_x = x - slide_distance if 'right' in asset.position else x + slide_distance
            overlay_expr = f"overlay=x='if(between(t,{asset.start_time},{asset.start_time + asset.duration}),{start_x}+({x}-{start_x})*min(1,(t-{asset.start_time})/{style.get('image_transition_duration', 0.5)}),{x})':y={y}"
            overlay_expr += f":enable='between(t,{asset.start_time},{asset.start_time + asset.duration})'"
        
        elif asset.animation_type == 'zoom':
            # Zoom animation
            transition_duration = style.get('image_transition_duration', 0.5)
            scale_filter += f",scale='if(between(t,{asset.start_time},{asset.start_time + asset.duration}),{size_pixels}*(0.5+0.5*min(1,(t-{asset.start_time})/{transition_duration})),{size_pixels})':-1"
        
        # Add Ken Burns effect if specified
        if 'ken_burns' in asset.effects:
            scale_filter += f",zoompan=z='min(zoom+0.001,1.2)':x='iw/2':y='ih/2':d={int(asset.duration * 30)}"
        
        # Combine scale and overlay
        return f"{scale_filter}[overlay{asset.asset_id}];{base_input}[overlay{asset.asset_id}]{overlay_expr}"
    
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
            
            return f"ass={subtitle_path}"
            
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
            # Build input streams
            inputs = [ffmpeg.input(input_video)]
            
            # Add overlay images as inputs
            for asset in overlay_assets:
                inputs.append(ffmpeg.input(asset.local_path))
            
            # Build filter complex
            filter_complex = filter_graph.build()
            
            # Build FFmpeg stream
            if filter_complex:
                stream = ffmpeg.filter(inputs, 'complex', filter_complex)
                video_stream = stream['outv']
                audio_stream = inputs[0]['a']  # Use original audio
            else:
                # No overlays, pass through
                video_stream = inputs[0]['v']
                audio_stream = inputs[0]['a']
            
            # Add captions if available
            if caption_filter:
                video_stream = video_stream.filter('subtitles', caption_filter)
            
            # Build output
            output_stream = ffmpeg.output(
                video_stream,
                audio_stream,
                output_path,
                vcodec=self.config.output_codec,
                acodec=self.config.audio_codec,
                video_bitrate=self.config.output_bitrate,
                audio_bitrate=self.config.audio_bitrate,
                preset=self.config.preset,
                crf=self.config.crf,
                threads=self.config.threads,
                movflags='faststart',  # Web optimization
                pix_fmt='yuv420p'  # Compatibility
            )
            
            # Execute FFmpeg command
            logger.info("Executing FFmpeg composition")
            command_args = ffmpeg.compile(output_stream)
            command_str = ' '.join(command_args)
            
            logger.debug(f"FFmpeg command: {command_str}")
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ffmpeg.run(output_stream, overwrite_output=True, quiet=False)
            )
            
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
                metadata.update({
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': eval(video_stream.get('r_frame_rate', '30/1')),
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
                stats.update({
                    'resolution': (int(video_stream.get('width', 0)), int(video_stream.get('height', 0))),
                    'fps': eval(video_stream.get('r_frame_rate', '30/1'))
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