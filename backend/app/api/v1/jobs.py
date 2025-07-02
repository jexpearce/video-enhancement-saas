"""
Job management endpoints for the Video Enhancement SaaS API.

Handles job status checking, progress tracking, and job management.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

# Import database dependencies
from app.database.connection import get_db
from app.database.models import ProcessingJob, StoredImage, EnrichedEntity

# Import schemas
from app.models.schemas import JobStatusResponse, JobStatus, JobResult

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the current status of a processing job.
    
    Args:
        job_id: Processing job ID
        db: Database session
        
    Returns:
        JobStatusResponse with current status and progress
    """
    
    try:
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Calculate progress based on status
        progress_mapping = {
            "pending": 0,
            "processing": 50,
            "completed": 100,
            "failed": 0
        }
        
        progress = progress_mapping.get(job.status, 0)
        
        # Get additional info for completed jobs
        result = None
        if job.status == "completed":
            # Get processed images count
            image_count = db.query(StoredImage).filter(StoredImage.job_id == job_id).count()
            
            # Get entities count
            entity_count = db.query(EnrichedEntity).filter(EnrichedEntity.job_id == job_id).count()
            
            # Convert local file paths to HTTP URLs for backward compatibility
            final_video_url = job.final_video_url
            if final_video_url and final_video_url.startswith("/tmp/video_enhancement/"):
                relative_path = final_video_url.replace("/tmp/video_enhancement/", "")
                final_video_url = f"http://localhost:8000/api/v1/files/{relative_path}"
            
            # Create JobResult object
            result = JobResult(
                transcript=job.transcript,
                emphasis_points_count=len(job.emphasis_points) if job.emphasis_points else 0,
                entities_count=entity_count,
                images_count=image_count,
                final_video_url=final_video_url,
                processing_time=job.processing_time_seconds
            )
        
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus(job.status),
            progress=progress,
            message=job.error_message if job.status == "failed" else None,
            result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get job status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get job status"
        )

@router.get("/jobs")
async def list_jobs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    db: Session = Depends(get_db)
):
    """
    List processing jobs with optional filtering.
    
    Args:
        user_id: Filter by user ID (optional)
        status: Filter by status (optional)
        limit: Number of jobs to return (1-100)
        offset: Number of jobs to skip
        db: Database session
        
    Returns:
        List of jobs with basic information
    """
    
    try:
        # Build query
        query = db.query(ProcessingJob)
        
        if user_id:
            query = query.filter(ProcessingJob.user_id == user_id)
        
        if status:
            query = query.filter(ProcessingJob.status == status)
        
        # Order by creation date (newest first)
        query = query.order_by(ProcessingJob.created_at.desc())
        
        # Apply pagination
        total_count = query.count()
        jobs = query.offset(offset).limit(limit).all()
        
        # Format response
        job_list = []
        for job in jobs:
            job_list.append({
                "job_id": str(job.id),
                "user_id": job.user_id,
                "status": job.status,
                "target_platform": job.target_platform,
                "file_size_bytes": job.file_size_bytes,
                "video_duration_seconds": job.video_duration_seconds,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "processing_time_seconds": job.processing_time_seconds,
                "retry_count": job.retry_count
            })
        
        return {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "jobs": job_list
        }
        
    except Exception as e:
        logger.error(f"List jobs failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list jobs"
        )

@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a processing job and all associated data.
    
    Args:
        job_id: Processing job ID
        db: Database session
        
    Returns:
        Confirmation message
    """
    
    try:
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Don't allow deletion of processing jobs
        if job.status == "processing":
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete job that is currently processing"
            )
        
        # Delete the job (cascading deletes will handle related records)
        db.delete(job)
        db.commit()
        
        logger.info(f"Deleted job: {job_id}")
        
        return {"message": "Job deleted successfully", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete job failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete job"
        )

@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Retry a failed processing job.
    
    Args:
        job_id: Processing job ID
        db: Database session
        
    Returns:
        Updated job status
    """
    
    try:
        # Get job from database
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != "failed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not in failed status. Current status: {job.status}"
            )
        
        # Check retry limit
        if job.retry_count >= 3:
            raise HTTPException(
                status_code=400,
                detail="Maximum retry attempts reached"
            )
        
        # Reset job for retry
        job.status = "pending"
        job.error_message = None
        job.started_at = None
        job.completed_at = None
        
        db.commit()
        
        # TODO: Trigger background processing again
        
        logger.info(f"Job {job_id} queued for retry")
        
        return {
            "message": "Job queued for retry",
            "job_id": job_id,
            "retry_count": job.retry_count + 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retry job failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retry job"
        ) 