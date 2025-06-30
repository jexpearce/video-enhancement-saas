"""
Video processing endpoints for the Video Enhancement SaaS API.

Handles video uploads, processing job creation, and result management.
"""

import os
import hashlib
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# Import database dependencies
from app.database.connection import get_db
from app.database.models import ProcessingJob, StoredImage

# Import processing services
from app.services.processing_pipeline import ProcessingPipeline, ProcessingConfig

# Import schemas
from app.models.schemas import (
    ProcessVideoRequest, 
    ProcessVideoResponse, 
    JobStatusResponse,
    JobStatus
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Allowed video formats - comprehensive list to handle browser variations
ALLOWED_VIDEO_FORMATS = {
    # MP4 variations
    "video/mp4", "video/mpeg4", "video/mp4v-es",
    # MOV/QuickTime variations  
    "video/quicktime", "video/x-quicktime",
    # AVI variations
    "video/x-msvideo", "video/avi", "video/msvideo",
    # WMV variations
    "video/x-ms-wmv", "video/x-ms-asf",
    # WebM
    "video/webm",
    # Additional common formats
    "video/3gpp", "video/x-flv", "video/x-m4v"
}

# File extension to MIME type mapping for fallback validation
EXTENSION_TO_MIME = {
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime', 
    '.avi': 'video/x-msvideo',
    '.wmv': 'video/x-ms-wmv',
    '.webm': 'video/webm',
    '.3gp': 'video/3gpp',
    '.flv': 'video/x-flv',
    '.m4v': 'video/x-m4v'
}

@router.post("/videos/upload", response_model=ProcessVideoResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = "anonymous",
    target_platform: str = "tiktok",
    db: Session = Depends(get_db)
):
    """
    Upload a video file and start processing.
    
    Args:
        file: Video file to process
        user_id: User identifier (optional for MVP)
        target_platform: Target platform (tiktok, instagram, youtube)
        db: Database session
        
    Returns:
        ProcessVideoResponse with job ID and status
    """
    
    logger.info(f"Upload attempt - filename: {file.filename}, content_type: {file.content_type}, user_id: {user_id}, target_platform: {target_platform}")
    
    try:
        # Validate file - check both MIME type and file extension
        is_valid_mime = file.content_type and file.content_type in ALLOWED_VIDEO_FORMATS
        is_valid_extension = False
        
        if file.filename:
            file_ext = os.path.splitext(file.filename.lower())[1]
            is_valid_extension = file_ext in EXTENSION_TO_MIME
        
        if not is_valid_mime and not is_valid_extension:
            logger.warning(f"Invalid file - content_type: {file.content_type}, filename: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Please upload MP4, MOV, AVI, WMV, or WebM videos."
            )
        
        # If MIME type is missing or incorrect but extension is valid, use extension-based MIME type
        if not is_valid_mime and is_valid_extension and file.filename:
            file_ext = os.path.splitext(file.filename.lower())[1]
            logger.info(f"Using extension-based MIME type for {file.filename}: {EXTENSION_TO_MIME[file_ext]}")
            # Note: We'll use the corrected MIME type for storage later
        
        # Read file content for validation and hashing
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check for existing processing job with same hash
        existing_job = db.query(ProcessingJob).filter(
            ProcessingJob.original_video_hash == file_hash,
            ProcessingJob.status.in_(["pending", "processing", "completed"])
        ).first()
        
        if existing_job:
            logger.info(f"Found existing job for file hash: {file_hash}")
            return ProcessVideoResponse(
                job_id=str(existing_job.id),
                status=JobStatus(existing_job.status),
                message="Video already processed or processing"
            )
        
        # Save file temporarily (in production, upload to S3)
        temp_dir = "/tmp/video_enhancement"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Handle filename extension safely
        if file.filename and '.' in file.filename:
            file_extension = file.filename.split('.')[-1]
        else:
            file_extension = 'mp4'  # Default extension
        
        temp_file_path = f"{temp_dir}/{file_hash}.{file_extension}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Create processing job record
        job = ProcessingJob(
            user_id=user_id,
            status="pending",
            original_video_url=temp_file_path,  # In production: S3 URL
            original_video_hash=file_hash,
            file_size_bytes=len(file_content),
            target_platform=target_platform,
            video_format=file.content_type
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        logger.info(f"Created processing job: {job.id}")
        
        # Start background processing
        background_tasks.add_task(process_video_background, str(job.id))
        
        return ProcessVideoResponse(
            job_id=str(job.id),
            status=JobStatus.PENDING,
            message="Video uploaded successfully. Processing started."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload video"
        )

@router.get("/videos/{job_id}/download")
async def download_processed_video(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Download the processed video result.
    
    Args:
        job_id: Processing job ID
        db: Database session
        
    Returns:
        Video file or redirect to CDN URL
    """
    
    try:
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Video not ready. Current status: {job.status}"
            )
        
        if not job.final_video_url:
            raise HTTPException(
                status_code=404, 
                detail="Processed video not found"
            )
        
        # In production: return redirect to S3/CloudFront URL
        # For now, return the URL directly
        return JSONResponse({
            "download_url": job.final_video_url,
            "expires_at": "2024-12-31T23:59:59Z"  # Set appropriate expiration
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate download URL"
        )

@router.get("/videos/{job_id}/images")
async def get_processed_images(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all processed images for a job.
    
    Args:
        job_id: Processing job ID
        db: Database session
        
    Returns:
        List of processed images with URLs and metadata
    """
    
    try:
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get stored images for this job
        images = db.query(StoredImage).filter(StoredImage.job_id == job_id).all()
        
        image_data = []
        for image in images:
            image_data.append({
                "id": str(image.id),
                "entity_name": image.entity_name,
                "entity_type": image.entity_type,
                "cdn_urls": image.cdn_urls,
                "original_url": image.original_url,
                "provider": image.original_provider,
                "photographer": image.photographer,
                "quality_scores": {
                    "overall": image.quality_score,
                    "relevance": image.relevance_score,
                    "aesthetic": image.aesthetic_score,
                    "ranking": image.ranking_score
                },
                "timing": {
                    "suggested_start": image.suggested_start_time,
                    "suggested_duration": image.suggested_duration,
                    "actual_start": image.actual_start_time,
                    "actual_duration": image.actual_duration
                },
                "created_at": image.created_at.isoformat()
            })
        
        return {
            "job_id": job_id,
            "total_images": len(image_data),
            "images": image_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get images failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve images"
        )

async def process_video_background(job_id: str):
    """
    Background task to process video.
    
    Args:
        job_id: Processing job ID
    """
    
    # Import here to avoid circular imports
    from app.database.connection import SessionLocal
    
    db = SessionLocal()
    
    try:
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        # Update job status
        job.status = "processing"
        db.commit()
        
        logger.info(f"Starting video processing for job: {job_id}")
        
        # Initialize processing pipeline
        config = ProcessingConfig(
            enhance_audio=True,
            emphasis_threshold=0.3,
            entity_confidence_threshold=0.6,
            max_images_per_entity=3,
            image_quality_threshold=0.6
        )
        
        pipeline = ProcessingPipeline(config)
        
        # Process the video
        result = await pipeline.process_video(
            video_path=job.original_video_url,
            output_dir=f"/tmp/video_enhancement/jobs/{job_id}"
        )
        
        # Update job with results
        job.transcript = result.transcription
        job.emphasis_points = [
            {
                "word": point.get("word", {}),
                "features": point.get("features", {}),
                "timestamp": point.get("timestamp", 0),
                "duration": point.get("duration", 0)
            }
            for point in result.emphasized_segments
        ]
        job.detected_entities = [
            {
                "text": entity.text if hasattr(entity, 'text') else str(entity),
                "type": entity.type if hasattr(entity, 'type') else "UNKNOWN",
                "confidence": entity.confidence if hasattr(entity, 'confidence') else 0.0
            }
            for entity in result.enriched_entities
        ]
        
        # Performance metrics with type safety
        try:
            job.processing_time_seconds = float(result.processing_time) if result.processing_time is not None else 0.0
        except (ValueError, TypeError):
            job.processing_time_seconds = 0.0
            
        try:
            job.audio_enhancement_time = float(result.audio_enhancement_time) if result.audio_enhancement_time is not None else 0.0
        except (ValueError, TypeError):
            job.audio_enhancement_time = 0.0
            
        try:
            job.transcription_time = float(result.transcription_time) if result.transcription_time is not None else 0.0
        except (ValueError, TypeError):
            job.transcription_time = 0.0
            
        try:
            job.emphasis_detection_time = float(result.emphasis_detection_time) if result.emphasis_detection_time is not None else 0.0
        except (ValueError, TypeError):
            job.emphasis_detection_time = 0.0
            
        try:
            job.entity_extraction_time = float(result.entity_recognition_time) if result.entity_recognition_time is not None else 0.0
        except (ValueError, TypeError):
            job.entity_extraction_time = 0.0
            
        try:
            job.image_search_time = float(result.image_search_time) if result.image_search_time is not None else 0.0
        except (ValueError, TypeError):
            job.image_search_time = 0.0
            
        try:
            job.image_processing_time = float(result.image_processing_time) if result.image_processing_time is not None else 0.0
        except (ValueError, TypeError):
            job.image_processing_time = 0.0
        
        # Mark as completed
        from datetime import datetime
        job.status = "completed"
        job.completed_at = datetime.now()
        job.final_video_url = result.enhanced_audio_path  # Placeholder
        
        db.commit()
        
        logger.info(f"Video processing completed for job: {job_id}")
        
    except Exception as e:
        logger.error(f"Video processing failed for job {job_id}: {e}")
        
        # Update job status to failed
        try:
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                # Ensure type safety for retry_count increment
                try:
                    current_retry_count = int(job.retry_count) if job.retry_count is not None else 0
                except (ValueError, TypeError):
                    current_retry_count = 0
                job.retry_count = current_retry_count + 1
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")
    
    finally:
        db.close() 