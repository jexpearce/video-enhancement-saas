"""
Additional methods for VideoComposer class.
Contains FFmpeg operations and utility functions.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import ffmpeg
import aiofiles

from .models import (
    FilterGraph,
    CompositionAsset,
    CompositionTimeline,
    CompositionConfig,
    FFmpegError,
    CompositionError
)

logger = logging.getLogger(__name__)

class VideoComposerMethods:
    """Additional methods for VideoComposer to handle FFmpeg and utilities."""
    
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
            # No overlays, just pass through with format conversion
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
            if i < len(overlay_assets) - 1:
                stage_output = f"[stage{i}]"
            else:
                stage_output = "[outv]"
            
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
        
        # Get size in pixels
        size_pixels = self._get_size_pixels(asset.size)
        
        # Start with scaling the overlay image
        scale_filter = f"{overlay_input}scale={size_pixels}:-1"
        
        # Add animation effects to the scale filter
        if asset.animation_type == 'fade':
            # Fade in/out animation
            transition_duration = style.get('image_transition_duration', 0.5)
            scale_filter += f",fade=t=in:st={asset.start_time}:d={transition_duration}"
            scale_filter += f",fade=t=out:st={asset.start_time + asset.duration - transition_duration}:d={transition_duration}"
        
        elif asset.animation_type == 'zoom':
            # Zoom animation
            transition_duration = style.get('image_transition_duration', 0.5)
            zoom_expr = f"if(between(t,{asset.start_time},{asset.start_time + asset.duration}),{size_pixels}*(0.5+0.5*min(1,(t-{asset.start_time})/{transition_duration})),{size_pixels})"
            scale_filter = f"{overlay_input}scale='{zoom_expr}':-1"
        
        # Add Ken Burns effect if specified
        if 'ken_burns' in asset.effects:
            scale_filter += f",zoompan=z='min(zoom+0.001,1.2)':x='iw/2':y='ih/2':d={int(asset.duration * 30)}"
        
        # Create overlay positioning
        if asset.animation_type == 'slide':
            # Slide animation - modify x position over time
            slide_distance = 100
            start_x = x - slide_distance if 'right' in asset.position else x + slide_distance
            transition_duration = style.get('image_transition_duration', 0.5)
            
            x_expr = f"if(between(t,{asset.start_time},{asset.start_time + asset.duration}),{start_x}+({x}-{start_x})*min(1,(t-{asset.start_time})/{transition_duration}),{x})"
            overlay_expr = f"overlay=x='{x_expr}':y={y}"
        else:
            # Static positioning
            overlay_expr = f"overlay=x={x}:y={y}"
        
        # Add timing constraints
        overlay_expr += f":enable='between(t,{asset.start_time},{asset.start_time + asset.duration})'"
        
        # Combine scale and overlay filters
        overlay_label = f"[overlay{asset.asset_id}]"
        return f"{scale_filter}{overlay_label};{base_input}{overlay_label}{overlay_expr}"
    
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
    
    async def _execute_composition(
        self,
        input_video: str,
        overlay_assets: List[CompositionAsset],
        filter_graph: FilterGraph,
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
            if filter_complex and overlay_assets:
                # Use complex filter for overlays
                stream = ffmpeg.filter(inputs, 'complex', filter_complex)
                video_stream = stream['outv']
                audio_stream = inputs[0]['a']  # Use original audio
            else:
                # No overlays, pass through
                video_stream = inputs[0]['v']
                audio_stream = inputs[0]['a']
            
            # Build output with proper settings
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