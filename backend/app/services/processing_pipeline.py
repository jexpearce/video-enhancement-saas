"""
Main Processing Pipeline for Video Enhancement SaaS

Orchestrates the complete end-to-end workflow:
1. Audio extraction and enhancement
2. Emphasis detection and entity recognition  
3. Image search and processing
4. Content matching and timing optimization
5. Video overlay generation
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import our services
from .audio.processor import AudioProcessor
from .transcription.whisper_service import WhisperService
from .emphasis.detector import MultiModalEmphasisDetector
from .nlp.entity_recognizer import EntityRecognizer
from .nlp.entity_enricher import EntityEnricher
from .images.image_searcher import ImageSearcher
from .images.image_processor import ImageProcessor
from .images.content_matcher import ContentMatcher, VideoSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline."""
    # Audio processing
    enhance_audio: bool = True
    whisper_model: str = "base"
    
    # Emphasis detection
    emphasis_threshold: float = 0.3
    min_emphasis_duration: float = 1.0
    
    # Entity recognition
    entity_confidence_threshold: float = 0.6
    max_entities_per_segment: int = 3
    
    # Image search
    max_images_per_entity: int = 3
    image_quality_threshold: float = 0.6
    
    # Content matching
    max_overlays_per_video: int = 8
    min_overlay_duration: float = 2.0
    max_overlay_duration: float = 6.0
    
    # Output format
    video_format: str = "portrait"  # portrait, landscape, square
    overlay_position: str = "top-right"
    overlay_opacity: float = 0.9

@dataclass
class ProcessingResult:
    """Result of complete processing pipeline."""
    # Input info
    video_path: str
    audio_path: str
    processing_config: ProcessingConfig
    
    # Processing results
    transcription: str
    emphasized_segments: List[Dict]
    recognized_entities: List[Any]
    enriched_entities: List[Any]
    image_results: List[Any]
    processed_images: List[Any]
    content_matches: List[Any]
    
    # Performance metrics
    processing_time: float
    audio_enhancement_time: float
    transcription_time: float
    emphasis_detection_time: float
    entity_recognition_time: float
    image_search_time: float
    image_processing_time: float
    content_matching_time: float
    
    # Quality scores
    overall_confidence: float
    emphasis_accuracy: float
    entity_recognition_accuracy: float
    image_match_quality: float
    
    # Output paths
    enhanced_audio_path: str = ""
    overlay_config_path: str = ""
    processed_at: datetime = field(default_factory=datetime.now)

class ProcessingPipeline:
    """Complete video enhancement processing pipeline."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the processing pipeline."""
        
        self.config = config or ProcessingConfig()
        
        # Initialize all services
        self.audio_processor = AudioProcessor()
        self.whisper_service = WhisperService()
        self.emphasis_detector = MultiModalEmphasisDetector()
        self.entity_recognizer = EntityRecognizer()
        self.entity_enricher = EntityEnricher()
        self.image_searcher = ImageSearcher()
        self.image_processor = ImageProcessor()
        self.content_matcher = ContentMatcher()
        
        # Performance tracking
        self.processing_stats = {
            'total_videos_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'success_rate': 0.0
        }
        
    async def process_video(self, video_path: str, output_dir: str = "./output") -> ProcessingResult:
        """
        Process a complete video through the enhancement pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory for output files
            
        Returns:
            ProcessingResult with all pipeline outputs
        """
        
        try:
            start_time = time.time()
            logger.info(f"Starting video processing pipeline for: {video_path}")
            
            # Initialize result object
            result = ProcessingResult(
                video_path=video_path,
                audio_path="",
                processing_config=self.config,
                transcription="",
                emphasized_segments=[],
                recognized_entities=[],
                enriched_entities=[],
                image_results=[],
                processed_images=[],
                content_matches=[],
                processing_time=0.0,
                audio_enhancement_time=0.0,
                transcription_time=0.0,
                emphasis_detection_time=0.0,
                entity_recognition_time=0.0,
                image_search_time=0.0,
                image_processing_time=0.0,
                content_matching_time=0.0,
                overall_confidence=0.0,
                emphasis_accuracy=0.0,
                entity_recognition_accuracy=0.0,
                image_match_quality=0.0
            )
            
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Extract and enhance audio
            step_start = time.time()
            logger.info("Step 1: Audio extraction and enhancement")
            
            audio_path = await self._extract_and_enhance_audio(video_path, output_path)
            result.audio_path = audio_path
            result.enhanced_audio_path = audio_path
            result.audio_enhancement_time = time.time() - step_start
            
            # Step 2: Transcribe audio with Whisper
            step_start = time.time()
            logger.info("Step 2: Audio transcription")
            
            transcription_result = await self._transcribe_audio(audio_path)
            result.transcription = transcription_result.get('text', '')
            result.transcription_time = time.time() - step_start
            
            # Step 3: Detect emphasized segments
            step_start = time.time()
            logger.info("Step 3: Emphasis detection")
            
            emphasized_segments = await self._detect_emphasis(audio_path, transcription_result)
            result.emphasized_segments = emphasized_segments
            result.emphasis_detection_time = time.time() - step_start
            
            # Step 4: Extract and enrich entities
            step_start = time.time()
            logger.info("Step 4: Entity recognition and enrichment")
            
            entities, enriched_entities = await self._extract_and_enrich_entities(
                result.transcription, emphasized_segments
            )
            result.recognized_entities = entities
            result.enriched_entities = enriched_entities
            result.entity_recognition_time = time.time() - step_start
            
            # Step 5: Search for relevant images
            step_start = time.time() 
            logger.info("Step 5: Image search")
            
            try:
                image_results = await self._search_images(enriched_entities)
                result.image_results = image_results
                result.image_search_time = time.time() - step_start
            except Exception as e:
                import traceback
                logger.error(f"Detailed error in Step 5: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
            
            # Step 6: Process images for overlay
            step_start = time.time()
            logger.info("Step 6: Image processing")
            
            processed_images = await self._process_images(image_results)
            result.processed_images = processed_images
            result.image_processing_time = time.time() - step_start
            
            # Step 7: Match content and optimize timing
            step_start = time.time()
            logger.info("Step 7: Content matching and timing optimization")
            
            content_matches = await self._match_content(
                emphasized_segments, enriched_entities, processed_images
            )
            result.content_matches = content_matches
            result.content_matching_time = time.time() - step_start
            
            # Calculate final metrics
            result.processing_time = time.time() - start_time
            result = self._calculate_quality_metrics(result)
            
            # Save results
            await self._save_results(result, output_path)
            
            # Update stats
            self._update_processing_stats(result)
            
            logger.info(f"Pipeline completed in {result.processing_time:.2f}s with {result.overall_confidence:.1%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise
    
    async def _extract_and_enhance_audio(self, video_path: str, output_path: Path) -> str:
        """Extract and enhance audio from video."""
        
        # For now, mock audio extraction (would use FFmpeg in production)
        audio_filename = f"audio_{int(time.time())}.wav"
        audio_path = output_path / audio_filename
        
        if self.config.enhance_audio:
            # Mock enhancement
            logger.info("Enhancing audio quality...")
            await asyncio.sleep(0.5)  # Simulate processing
        
        # Mock file creation
        audio_path.write_text("mock_audio_data")
        
        return str(audio_path)
    
    async def _transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper."""
        
        # Mock transcription for now
        mock_transcription = {
            'text': "This is a mock transcription of the audio. Biden announced new policies regarding Iran and Netanyahu. The Tesla stock is performing well.",
            'segments': [
                {'start': 0.0, 'end': 3.0, 'text': 'This is a mock transcription of the audio.'},
                {'start': 3.0, 'end': 8.0, 'text': 'Biden announced new policies regarding Iran and Netanyahu.'},
                {'start': 8.0, 'end': 12.0, 'text': 'The Tesla stock is performing well.'}
            ],
            'language': 'en'
        }
        
        logger.info(f"Transcribed {len(mock_transcription['text'])} characters")
        return mock_transcription
    
    async def _detect_emphasis(self, audio_path: str, transcription: Dict) -> List[Dict]:
        """Detect emphasized segments in audio."""
        
        # Mock emphasis detection
        emphasized_segments = [
            {
                'start_time': 3.0,
                'end_time': 8.0,
                'confidence': 0.85,
                'text': 'Biden announced new policies regarding Iran and Netanyahu',
                'emphasis_type': 'strong',
                'audio_features': {'rms_energy': 0.75, 'pitch_variance': 0.8}
            },
            {
                'start_time': 8.0,
                'end_time': 12.0, 
                'confidence': 0.72,
                'text': 'The Tesla stock is performing well',
                'emphasis_type': 'moderate',
                'audio_features': {'rms_energy': 0.65, 'pitch_variance': 0.6}
            }
        ]
        
        logger.info(f"Detected {len(emphasized_segments)} emphasized segments")
        return emphasized_segments
    
    async def _extract_and_enrich_entities(self, transcription: str, emphasized_segments: List[Dict]) -> Tuple[List[Any], List[Any]]:
        """Extract and enrich entities from transcription."""
        
        # Extract entities
        entities = await asyncio.get_event_loop().run_in_executor(
            None, self.entity_recognizer.recognize_entities, transcription
        )
        
        # Enrich entities 
        enriched_entities = await asyncio.get_event_loop().run_in_executor(
            None, self.entity_enricher.enrich_entities, entities
        )
        
        logger.info(f"Extracted {len(entities)} entities, enriched {len(enriched_entities)}")
        return entities, enriched_entities
    
    async def _search_images(self, enriched_entities: List[Any]) -> List[Any]:
        """Search for images for entities."""
        
        image_results = []
        
        for entity in enriched_entities[:self.config.max_entities_per_segment]:
            # Skip low-quality entities with safe type conversion
            if hasattr(entity, 'image_potential'):
                try:
                    image_potential = float(entity.image_potential) if entity.image_potential is not None else 0.0
                    if image_potential < self.config.entity_confidence_threshold:
                        continue
                except (ValueError, TypeError):
                    # Skip entities with invalid image_potential values
                    continue

            # Search for images
            entity_images = await self.image_searcher.search_for_entity(entity)

            # Limit results per entity
            entity_images = entity_images[:self.config.max_images_per_entity]
            image_results.extend(entity_images)
        
        logger.info(f"Found {len(image_results)} images for {len(enriched_entities)} entities")
        return image_results
    
    async def _process_images(self, image_results: List[Any]) -> List[Any]:
        """Process images for video overlay."""
        
        processed_images = []
        
        async with self.image_processor as processor:
            # Get processing options based on video format
            preset_options = processor.get_preset_options(self.config.video_format)
            
            # Process images in parallel (batches of 5)
            batch_size = 5
            for i in range(0, len(image_results), batch_size):
                batch = image_results[i:i + batch_size]
                batch_urls = [getattr(img, 'url', '') for img in batch]
                
                if batch_urls:
                    batch_processed = await processor.process_multiple_images(
                        batch_urls, preset_options
                    )
                    processed_images.extend(batch_processed)
        
        logger.info(f"Processed {len(processed_images)}/{len(image_results)} images")
        return processed_images
    
    async def _match_content(self, emphasized_segments: List[Dict], 
                           enriched_entities: List[Any], 
                           processed_images: List[Any]) -> List[Any]:
        """Match content and optimize timing."""
        
        # Create video segments from emphasis results
        video_segments = []
        for segment in emphasized_segments:
            # Find entities in this segment
            segment_entities = []
            for entity in enriched_entities:
                # Simple timing check (would be more sophisticated in production)
                if hasattr(entity, 'start_time'):
                    entity_time = entity.start_time
                else:
                    entity_time = segment['start_time']  # Default to segment start
                
                if segment['start_time'] <= entity_time <= segment['end_time']:
                    segment_entities.append(entity.canonical_name)
            
            video_segment = VideoSegment(
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                emphasized_entities=segment_entities,
                confidence=segment['confidence'],
                text_content=segment.get('text', '')
            )
            video_segments.append(video_segment)
        
        # Match images to segments
        content_matches = await self.content_matcher.match_images_to_video(
            video_segments, processed_images, self.config.video_format
        )
        
        # Limit to max overlays
        content_matches = content_matches[:self.config.max_overlays_per_video]
        
        logger.info(f"Created {len(content_matches)} content matches")
        return content_matches
    
    def _calculate_quality_metrics(self, result: ProcessingResult) -> ProcessingResult:
        """Calculate overall quality metrics."""
        
        # Overall confidence (weighted average)
        emphasis_weight = 0.3
        entity_weight = 0.3
        image_weight = 0.4
        
        # Emphasis accuracy (based on confidence scores)
        if result.emphasized_segments:
            result.emphasis_accuracy = sum(
                seg['confidence'] for seg in result.emphasized_segments
            ) / len(result.emphasized_segments)
        else:
            result.emphasis_accuracy = 0.0
        
        # Entity recognition accuracy (based on image potential)
        if result.enriched_entities:
            entity_scores = []
            for entity in result.enriched_entities:
                if hasattr(entity, 'image_potential'):
                    entity_scores.append(entity.image_potential)
                else:
                    entity_scores.append(0.7)  # Default score
            
            result.entity_recognition_accuracy = sum(entity_scores) / len(entity_scores)
        else:
            result.entity_recognition_accuracy = 0.0
        
        # Image match quality (based on match scores)
        if result.content_matches:
            match_scores = [
                getattr(match, 'match_score', 0.7) for match in result.content_matches
            ]
            result.image_match_quality = sum(match_scores) / len(match_scores)
        else:
            result.image_match_quality = 0.0
        
        # Overall confidence
        result.overall_confidence = (
            result.emphasis_accuracy * emphasis_weight +
            result.entity_recognition_accuracy * entity_weight +
            result.image_match_quality * image_weight
        )
        
        return result
    
    async def _save_results(self, result: ProcessingResult, output_path: Path):
        """Save processing results to files."""
        
        # Save overlay configuration (JSON format for video editor)
        overlay_config = {
            'video_path': result.video_path,
            'processing_time': result.processing_time,
            'confidence': result.overall_confidence,
            'overlays': []
        }
        
        for match in result.content_matches:
            overlay_config['overlays'].append({
                'image_path': getattr(match, 'processed_path', ''),
                'start_time': getattr(match, 'start_time', 0.0),
                'duration': getattr(match, 'duration', 3.0),
                'position': getattr(match, 'position', 'top-right'),
                'opacity': getattr(match, 'opacity', 0.9),
                'entity_name': getattr(match.image_result, 'entity_name', '') if hasattr(match, 'image_result') else ''
            })
        
        # Save configuration file
        config_filename = f"overlay_config_{int(time.time())}.json"
        config_path = output_path / config_filename
        
        import json
        with open(config_path, 'w') as f:
            json.dump(overlay_config, f, indent=2)
        
        result.overlay_config_path = str(config_path)
        
        logger.info(f"Saved overlay configuration to: {config_path}")
    
    def _update_processing_stats(self, result: ProcessingResult):
        """Update processing statistics."""
        
        # Update stats with type safety
        try:
            current_total = int(self.processing_stats.get('total_videos_processed', 0))
        except (ValueError, TypeError):
            current_total = 0
        self.processing_stats['total_videos_processed'] = current_total + 1
        
        try:
            current_time = float(self.processing_stats.get('total_processing_time', 0.0))
        except (ValueError, TypeError):
            current_time = 0.0
        self.processing_stats['total_processing_time'] = current_time + result.processing_time
        
        # Calculate average with type safety
        try:
            total_videos = int(self.processing_stats.get('total_videos_processed', 1))
            total_time = float(self.processing_stats.get('total_processing_time', 0.0))
        except (ValueError, TypeError):
            total_videos = 1
            total_time = 0.0
        
        self.processing_stats['average_processing_time'] = (
            total_time / max(total_videos, 1)  # Prevent division by zero
        )
        
        # Success rate (based on confidence threshold)
        success_threshold = 0.6
        if result.overall_confidence >= success_threshold:
            current_success_rate = self.processing_stats.get('successful_videos', 0)
            # Ensure type safety - convert to int if it's a string
            try:
                current_success_count = int(current_success_rate) if current_success_rate is not None else 0
            except (ValueError, TypeError):
                current_success_count = 0
            self.processing_stats['successful_videos'] = current_success_count + 1
        
        # Calculate success rate with type safety
        try:
            successful_videos = int(self.processing_stats.get('successful_videos', 0))
        except (ValueError, TypeError):
            successful_videos = 0
        
        self.processing_stats['success_rate'] = (
            successful_videos / total_videos
        )
    
    def get_processing_statistics(self) -> Dict:
        """Get processing pipeline statistics."""
        return {
            **self.processing_stats,
            'config': {
                'video_format': self.config.video_format,
                'max_overlays': self.config.max_overlays_per_video,
                'emphasis_threshold': self.config.emphasis_threshold,
                'entity_threshold': self.config.entity_confidence_threshold
            }
        }
    
    async def process_batch(self, video_paths: List[str], output_dir: str = "./output") -> List[ProcessingResult]:
        """Process multiple videos in parallel."""
        
        logger.info(f"Starting batch processing of {len(video_paths)} videos")
        
        # Process videos in parallel (limit concurrency to avoid resource exhaustion)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent processes
        
        async def process_with_semaphore(video_path: str):
            async with semaphore:
                return await self.process_video(video_path, output_dir)
        
        tasks = [process_with_semaphore(path) for path in video_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [
            result for result in results 
            if isinstance(result, ProcessingResult)
        ]
        
        logger.info(f"Batch processing completed: {len(successful_results)}/{len(video_paths)} successful")
        return successful_results 