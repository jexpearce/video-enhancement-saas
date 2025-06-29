"""
Celery Application Configuration for Video Enhancement SaaS

Handles background processing of video enhancement tasks including:
- Audio analysis and emphasis detection
- Entity recognition and enrichment
- Image search and curation
- Video composition and rendering
"""

import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
import logging

from .database.connection import SessionLocal, check_database_connection

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'video_enhancement',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['app.tasks.video_processing']
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'app.tasks.video_processing.process_video': {'queue': 'video_processing'},
        'app.tasks.video_processing.cleanup_failed_job': {'queue': 'cleanup'},
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time
    task_acks_late=True,          # Acknowledge after task completion
    worker_disable_rate_limits=False,
    
    # Task time limits
    task_soft_time_limit=1800,    # 30 minutes soft limit
    task_time_limit=2400,         # 40 minutes hard limit
    
    # Retry settings
    task_default_retry_delay=60,   # 1 minute retry delay
    task_max_retries=3,
    
    # Result settings
    result_expires=3600,          # Results expire after 1 hour
    
    # Task compression
    task_compression='gzip',
    result_compression='gzip',
    
    # Memory management
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
)

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Initialize database connection when worker is ready."""
    try:
        if check_database_connection():
            logger.info("Celery worker database connection verified")
        else:
            logger.error("Celery worker cannot connect to database")
    except Exception as e:
        logger.error(f"Failed to verify database connection in worker: {e}")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Log worker shutdown."""
    logger.info("Celery worker shutting down")

def get_db_session():
    """Get database session for tasks."""
    return SessionLocal() 