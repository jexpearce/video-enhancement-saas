"""
Video Processing Tasks for Celery Background Processing

Handles the complete video enhancement pipeline:
1. Audio analysis and emphasis detection
2. Entity recognition and enrichment  
3. Image search and curation
4. Style selection and video composition
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import tempfile
import shutil

from celery import current_task
from sqlalchemy.orm import Session

from ..celery_app import celery_app, get_db_session
from ..database.models import ProcessingJob, JobStatus
from ..services.audio.emphasis_detector import EmphasisDetector
from ..services.nlp.entity_extractor import EntityExtractor
from ..services.nlp.entity_enricher import EntityEnricher
from ..services.images.image_searcher import ImageSearcher
from ..services.images.curation.curator import ImageCurator
from ..services.style.style_selector import StyleSelector
from ..services.composition.video_composer import VideoComposer
from ..utils.storage.s3_manager import S3Manager

logger = logging.getLogger(__name__)

class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

@celery_app.task(bind=True, name='app.tasks.video_processing.process_video')
async def process_video(self, job_id: str, input_file_path: str, user_preferences: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main video processing task - orchestrates the entire enhancement pipeline.
    
    Args:
        job_id: Unique identifier for the processing job
        input_file_path: Path to the uploaded video file
        user_preferences: Optional user customization preferences
        
    Returns:
        Dict with processing results and metadata
    """
    
    db_session = None
    temp_dir = None
    
    try:
        # Initialize database session
        db_session = get_db_session()
        
        # Get job from database
        job = db_session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            raise VideoProcessingError(f"Job {job_id} not found in database")
        
        # Update job status
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db_session.commit()
        
        logger.info(f"Started processing job {job_id}")
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix=f"video_processing_{job_id}_")
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'stage': 'audio_analysis', 'progress': 10})
        
        # Stage 1: Audio Analysis and Emphasis Detection
        logger.info(f"Job {job_id}: Starting audio analysis")
        audio_results = _process_audio(input_file_path, temp_dir, job_id)
        
        # Update progress and database
        job.audio_enhancement_time = (datetime.now(timezone.utc) - job.started_at).total_seconds()
        # job.emphasis_points_count = len(audio_results.get('emphasis_points', []))
        db_session.commit()
        
        self.update_state(state='PROGRESS', meta={'stage': 'entity_extraction', 'progress': 30})
        
        # Stage 2: Entity Recognition and Enrichment
        logger.info(f"Job {job_id}: Starting entity extraction")
        entity_results = _process_entities(audio_results.get('transcript', ''), job_id)
        
        # Update database
        job.entities_found_count = len(entity_results.get('entities', []))
        db_session.commit()
        
        self.update_state(state='PROGRESS', meta={'stage': 'image_search', 'progress': 50})
        
        # Stage 3: Image Search and Curation
        logger.info(f"Job {job_id}: Starting image search")
        image_results = _process_images(entity_results.get('entities', []), job_id)
        
        # Update database
        job.images_found_count = len(image_results.get('curated_images', []))
        db_session.commit()
        
        self.update_state(state='PROGRESS', meta={'stage': 'style_selection', 'progress': 70})
        
        # Stage 4: Style Selection
        logger.info(f"Job {job_id}: Selecting style")
        style_results = _select_style(user_preferences or {}, entity_results, job_id)
        
        self.update_state(state='PROGRESS', meta={'stage': 'video_composition', 'progress': 80})
        
        # Stage 5: Video Composition
        logger.info(f"Job {job_id}: Composing enhanced video")
        composition_results = await _compose_video(
            input_file_path,
            audio_results,
            entity_results,
            image_results,
            style_results,
            temp_dir,
            job_id
        )
        
        self.update_state(state='PROGRESS', meta={'stage': 'upload', 'progress': 95})
        
        # Stage 6: Upload to S3 and finalize
        logger.info(f"Job {job_id}: Uploading results")
        final_results = _finalize_results(composition_results, job_id)
        
        # Update job with final results
        job.status = JobStatus.COMPLETED
        job.processing_completed_at = datetime.now(timezone.utc)
        job.total_processing_time = (job.processing_completed_at - job.processing_started_at).total_seconds()
        job.output_video_url = final_results.get('output_video_url')
        job.output_metadata = final_results.get('metadata')
        db_session.commit()
        
        logger.info(f"Successfully completed processing job {job_id}")
        
        return {
            'status': 'completed',
            'job_id': job_id,
            'output_video_url': final_results.get('output_video_url'),
            'processing_time': job.total_processing_time,
            'metadata': final_results.get('metadata')
        }
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        
        # Update job status in database
        if db_session and job:
            try:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.processing_completed_at = datetime.now(timezone.utc)
                if job.processing_started_at:
                    job.total_processing_time = (job.processing_completed_at - job.processing_started_at).total_seconds()
                db_session.commit()
            except Exception as db_error:
                logger.error(f"Failed to update job status in database: {db_error}")
        
        # Update task state
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'job_id': job_id
            }
        )
        
        raise
        
    finally:
        # Cleanup
        if db_session:
            db_session.close()
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

def _process_audio(input_file_path: str, temp_dir: str, job_id: str) -> Dict[str, Any]:
    """Process audio for emphasis detection and transcription."""
    try:
        # Initialize emphasis detector
        emphasis_detector = EmphasisDetector()
        
        # Extract audio from video
        audio_file = os.path.join(temp_dir, f"{job_id}_audio.wav")
        emphasis_detector.extract_audio_from_video(input_file_path, audio_file)
        
        # Detect emphasis points
        emphasis_results = emphasis_detector.detect_emphasis(audio_file)
        
        # Get transcript
        transcript = emphasis_detector.get_transcript(audio_file)
        
        return {
            'transcript': transcript,
            'emphasis_points': emphasis_results.emphasis_points,
            'audio_features': emphasis_results.audio_features,
            'processing_metadata': emphasis_results.metadata
        }
        
    except Exception as e:
        logger.error(f"Audio processing failed for job {job_id}: {e}")
        raise VideoProcessingError(f"Audio processing failed: {e}")

def _process_entities(transcript: str, job_id: str) -> Dict[str, Any]:
    """Extract and enrich entities from transcript."""
    try:
        # Extract entities
        entity_extractor = EntityExtractor()
        entities = entity_extractor.extract_entities(transcript)
        
        # Enrich entities
        entity_enricher = EntityEnricher()
        enriched_entities = []
        
        for entity in entities:
            try:
                enriched = entity_enricher.enrich_entity(entity)
                enriched_entities.append(enriched)
            except Exception as e:
                logger.warning(f"Failed to enrich entity {entity.text}: {e}")
                # Add entity without enrichment
                enriched_entities.append(entity)
        
        return {
            'entities': enriched_entities,
            'entity_count': len(enriched_entities),
            'extraction_metadata': {
                'total_entities_found': len(entities),
                'successfully_enriched': len([e for e in enriched_entities if hasattr(e, 'wikidata_id')])
            }
        }
        
    except Exception as e:
        logger.error(f"Entity processing failed for job {job_id}: {e}")
        raise VideoProcessingError(f"Entity processing failed: {e}")

def _process_images(entities: list, job_id: str) -> Dict[str, Any]:
    """Search and curate images for entities."""
    try:
        # Initialize image searcher and curator
        image_searcher = ImageSearcher()
        image_curator = ImageCurator()
        
        all_curated_images = []
        search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'images_found': 0,
            'images_curated': 0
        }
        
        for entity in entities:
            try:
                # Search for images
                candidate_images = image_searcher.search_entity_images(entity)
                search_stats['total_searches'] += 1
                search_stats['images_found'] += len(candidate_images)
                
                if candidate_images:
                    search_stats['successful_searches'] += 1
                    
                    # Curate images
                    curated = image_curator.curate_entity_images(entity, candidate_images)
                    all_curated_images.extend(curated)
                    search_stats['images_curated'] += len(curated)
                    
            except Exception as e:
                logger.warning(f"Image processing failed for entity {entity.text}: {e}")
                continue
        
        return {
            'curated_images': all_curated_images,
            'search_stats': search_stats,
            'curation_metadata': image_curator.get_curation_stats(all_curated_images) if all_curated_images else {}
        }
        
    except Exception as e:
        logger.error(f"Image processing failed for job {job_id}: {e}")
        raise VideoProcessingError(f"Image processing failed: {e}")

def _select_style(user_preferences: Dict, entity_results: Dict, job_id: str) -> Dict[str, Any]:
    """Select appropriate style for video enhancement."""
    try:
        from ..app.services.images.styles.style_engine import StyleEngine
        from ..app.services.images.styles.models import Platform
        
        style_engine = StyleEngine()
        
        # Extract content info for style selection
        entities = entity_results.get('entities', [])
        entity_text = ' '.join([e.get('text', '') for e in entities[:3]])  # Top 3 entities
        
        # Apply style selection
        style_result = await style_engine.apply_style_to_video(
            video_id=job_id,
            audio_transcript=entity_text,
            video_title=user_preferences.get('title', ''),
            target_platform=Platform.TIKTOK,  # Default to TikTok
            creator_age=user_preferences.get('age'),
            creator_gender=user_preferences.get('gender')
        )
        
        return {
            'selected_style': style_result.visual_style,
            'style_metadata': {
                'selection_reason': style_result.selection_method,
                'style_confidence': style_result.confidence_score,
                'processing_time': style_result.processing_time_ms
            }
        }
        
    except Exception as e:
        logger.error(f"Style selection failed for job {job_id}: {e}")
        # Fallback to default style
        return {
            'selected_style': {
                'template_id': 'news_professional',
                'has_ken_burns': True,
                'has_pulse_to_beat': True,
                'animation_type': 'fade'
            },
            'style_metadata': {
                'selection_reason': 'fallback',
                'style_confidence': 0.5
            }
        }

async def _compose_video(
    input_file_path: str,
    audio_results: Dict,
    entity_results: Dict,
    image_results: Dict,
    style_results: Dict,
    temp_dir: str,
    job_id: str
) -> Dict[str, Any]:
    """Compose the final enhanced video using the VideoComposer system."""
    try:
        # Initialize VideoComposer with production settings
        from ..services.composition.models import CompositionConfig
        
        config = CompositionConfig(
            output_resolution=(1080, 1920),  # 9:16 for TikTok/Instagram
            output_fps=30,
            output_bitrate="5M",
            preset="fast",  # Balance quality and speed
            gpu_acceleration=True
        )
        
        video_composer = VideoComposer(config)
        
        # Get animation timeline if available from style results
        animation_timeline = None
        if 'animation_timeline' in style_results:
            animation_timeline = style_results['animation_timeline']
        else:
            # Generate animation timeline using existing animation engine
            try:
                from ..services.animation.animation_engine import AnimationEngine
                animation_engine = AnimationEngine()
                
                # Create ranked images format for animation engine
                curated_images = image_results.get('curated_images', [])
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
                
                # Generate timeline
                video_duration = audio_results.get('audio_features', {}).get('duration', 30.0)
                audio_beats = audio_results.get('audio_features', {}).get('beats', [])
                
                timeline_result = await animation_engine.create_image_animation_timeline(
                    emphasis_points=audio_results.get('emphasis_points', []),
                    ranked_images=ranked_images,
                    style=style_results.get('selected_style', {}),
                    video_duration=video_duration,
                    audio_beats=audio_beats
                )
                
                animation_timeline = timeline_result
                
            except Exception as e:
                logger.warning(f"Failed to generate animation timeline for job {job_id}: {e}")
                # Continue with empty timeline
                animation_timeline = {'events': [], 'duration': 30.0}
        
        # Prepare composition data
        composition_data = {
            'input_video_path': input_file_path,
            'emphasis_points': audio_results.get('emphasis_points', []),
            'entities': entity_results.get('entities', []),
            'curated_images': image_results.get('curated_images', []),
            'selected_style': style_results.get('selected_style', {}),
            'audio_features': audio_results.get('audio_features', {}),
            'animation_timeline': animation_timeline
        }
        
        # Generate output file path
        output_file = os.path.join(temp_dir, f"{job_id}_enhanced.mp4")
        
        logger.info(f"Job {job_id}: Starting video composition with {len(composition_data['curated_images'])} images")
        
        # Compose video (this is the async call)
        composition_result = await video_composer.compose_video(composition_data, output_file)
        
        if not composition_result.success:
            raise VideoProcessingError(f"Video composition failed: {composition_result.error_message}")
        
        logger.info(f"Job {job_id}: Video composition completed successfully in {composition_result.processing_time:.2f}s")
        
        return {
            'output_file_path': output_file,
            'composition_metadata': composition_result.to_dict(),
            'processing_stats': {
                'total_enhancements': len(audio_results.get('emphasis_points', [])),
                'images_used': len(image_results.get('curated_images', [])),
                'style_applied': style_results.get('selected_style', {}).get('template_name', 'default'),
                'processing_time': composition_result.processing_time,
                'overlays_applied': composition_result.total_overlays_applied,
                'effects_applied': composition_result.total_effects_applied
            }
        }
        
    except Exception as e:
        logger.error(f"Video composition failed for job {job_id}: {e}", exc_info=True)
        raise VideoProcessingError(f"Video composition failed: {e}")

def _finalize_results(composition_results: Dict, job_id: str) -> Dict[str, Any]:
    """Upload final video to S3 and prepare results."""
    try:
        s3_manager = S3Manager()
        
        output_file_path = composition_results.get('output_file_path')
        if not output_file_path or not os.path.exists(output_file_path):
            raise VideoProcessingError("Output video file not found")
        
        # Upload to S3
        s3_key = f"enhanced_videos/{job_id}/output.mp4"
        s3_url = s3_manager.upload_file(output_file_path, s3_key)
        
        # Generate CloudFront URL if available
        cloudfront_url = s3_manager.get_cloudfront_url(s3_key)
        
        return {
            'output_video_url': cloudfront_url or s3_url,
            's3_key': s3_key,
            'metadata': {
                'composition_metadata': composition_results.get('composition_metadata'),
                'processing_stats': composition_results.get('processing_stats'),
                'file_size_bytes': os.path.getsize(output_file_path),
                'upload_timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Result finalization failed for job {job_id}: {e}")
        raise VideoProcessingError(f"Result finalization failed: {e}")

@celery_app.task(name='app.tasks.video_processing.cleanup_failed_job')
def cleanup_failed_job(job_id: str) -> Dict[str, Any]:
    """Clean up resources for failed jobs."""
    try:
        db_session = get_db_session()
        
        # Get job from database
        job = db_session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            return {'status': 'job_not_found', 'job_id': job_id}
        
        # Clean up any temporary files
        s3_manager = S3Manager()
        temp_prefix = f"temp/{job_id}/"
        
        try:
            s3_manager.delete_prefix(temp_prefix)
            logger.info(f"Cleaned up S3 temp files for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up S3 temp files for job {job_id}: {e}")
        
        # Mark job as cleaned up
        job.cleanup_completed_at = datetime.now(timezone.utc)
        db_session.commit()
        db_session.close()
        
        return {
            'status': 'cleanup_completed',
            'job_id': job_id,
            'cleanup_timestamp': job.cleanup_completed_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed for job {job_id}: {e}")
        return {
            'status': 'cleanup_failed',
            'job_id': job_id,
            'error': str(e)
        }

@celery_app.task(name='app.tasks.video_processing.health_check')
def health_check() -> Dict[str, Any]:
    """Health check task for monitoring Celery workers."""
    try:
        # Test database connection
        db_session = get_db_session()
        db_session.execute("SELECT 1")
        db_session.close()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'worker_id': current_task.request.hostname
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        } 