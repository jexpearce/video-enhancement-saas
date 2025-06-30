"""
Database models for Video Enhancement SaaS

Core tables for job tracking, image storage, entity management, and processing results.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .connection import Base

def generate_uuid():
    """Generate a string UUID for cross-database compatibility."""
    return str(uuid.uuid4())

class ProcessingJob(Base):
    """
    Main job tracking table for video processing requests.
    """
    __tablename__ = "processing_jobs"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # User information
    user_id = Column(String(255), nullable=False, index=True)
    
    # Job status and tracking
    status = Column(String(50), default="pending", index=True)  # pending, processing, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    
    # Input video information
    original_video_url = Column(String(500), nullable=False)
    original_video_hash = Column(String(64), nullable=True, index=True)  # For deduplication
    file_size_bytes = Column(Integer, nullable=True)
    video_duration_seconds = Column(Float, nullable=True)
    video_format = Column(String(20), nullable=True)  # mp4, mov, etc.
    
    # Processing configuration
    target_platform = Column(String(20), default="tiktok")  # tiktok, instagram, youtube
    processing_options = Column(JSON, nullable=True)  # Custom processing settings
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Results and metadata
    transcript = Column(Text, nullable=True)
    emphasis_points = Column(JSON, nullable=True)
    detected_entities = Column(JSON, nullable=True)
    applied_style = Column(JSON, nullable=True)
    
    # Output information
    final_video_url = Column(String(500), nullable=True)
    manifest_url = Column(String(500), nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Performance metrics
    processing_time_seconds = Column(Float, nullable=True)
    audio_enhancement_time = Column(Float, nullable=True)
    transcription_time = Column(Float, nullable=True)
    emphasis_detection_time = Column(Float, nullable=True)
    entity_extraction_time = Column(Float, nullable=True)
    image_search_time = Column(Float, nullable=True)
    image_processing_time = Column(Float, nullable=True)
    
    # Relationships
    stored_images = relationship("StoredImage", back_populates="job", cascade="all, delete-orphan")
    enriched_entities = relationship("EnrichedEntity", back_populates="job", cascade="all, delete-orphan")
    emphasis_points_rel = relationship("EmphasisPoint", back_populates="job", cascade="all, delete-orphan")

class StoredImage(Base):
    """
    Images downloaded and processed for video enhancement.
    """
    __tablename__ = "stored_images"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign key to processing job
    job_id = Column(String(36), ForeignKey("processing_jobs.id"), nullable=False, index=True)
    
    # Original image information
    original_url = Column(String(500), nullable=False)
    original_provider = Column(String(50), nullable=False)  # unsplash, pexels, etc.
    
    # Storage information
    s3_key = Column(String(500), nullable=False)
    s3_bucket = Column(String(100), nullable=False)
    cdn_urls = Column(JSON, nullable=False)  # Different sizes/formats
    
    # Image metadata
    entity_name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)  # PERSON, ORG, GPE, etc.
    
    # Original image properties
    original_width = Column(Integer, nullable=True)
    original_height = Column(Integer, nullable=True)
    original_file_size = Column(Integer, nullable=True)
    photographer = Column(String(255), nullable=True)
    license_type = Column(String(50), nullable=True)
    
    # Quality and ranking scores
    clip_similarity = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    aesthetic_score = Column(Float, nullable=True)
    ranking_score = Column(Float, nullable=True)
    
    # Video timing information
    suggested_start_time = Column(Float, nullable=True)
    suggested_duration = Column(Float, default=3.0)
    actual_start_time = Column(Float, nullable=True)
    actual_duration = Column(Float, nullable=True)
    
    # Processing metadata
    processed_variants = Column(JSON, nullable=True)  # Different sizes/crops
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="stored_images")

class EnrichedEntity(Base):
    """
    Entities extracted from video transcripts with enrichment data.
    """
    __tablename__ = "enriched_entities"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign key to processing job
    job_id = Column(String(36), ForeignKey("processing_jobs.id"), nullable=False, index=True)
    
    # Basic entity information
    text = Column(String(255), nullable=False)
    normalized_text = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)  # PERSON, ORG, GPE, etc.
    confidence = Column(Float, nullable=False)
    
    # Position in transcript
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    
    # Enrichment data from external sources
    wikidata_id = Column(String(50), nullable=True, index=True)
    wikipedia_url = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    aliases = Column(JSON, nullable=True)  # Alternative names
    related_entities = Column(JSON, nullable=True)
    
    # Image information
    reference_image_urls = Column(JSON, nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=1)
    emphasis_strength = Column(Float, nullable=True)  # How emphasized this entity was
    
    # Search optimization
    search_queries = Column(JSON, nullable=True)  # Optimized search terms for this entity
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="enriched_entities")

class EmphasisPoint(Base):
    """
    Detected emphasis points in audio with timing and feature data.
    """
    __tablename__ = "emphasis_points"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign key to processing job
    job_id = Column(String(36), ForeignKey("processing_jobs.id"), nullable=False, index=True)
    
    # Word/phrase information
    word_text = Column(String(255), nullable=False)
    word_start_time = Column(Float, nullable=False)
    word_end_time = Column(Float, nullable=False)
    word_confidence = Column(Float, nullable=False)
    
    # Emphasis features
    acoustic_score = Column(Float, nullable=False)
    prosodic_score = Column(Float, nullable=False)
    linguistic_score = Column(Float, nullable=False)
    visual_score = Column(Float, nullable=True)  # Future: visual emphasis detection
    combined_score = Column(Float, nullable=False, index=True)
    
    # Context information
    context_before = Column(String(500), nullable=True)
    context_after = Column(String(500), nullable=True)
    sentence_position = Column(Integer, nullable=True)  # Position within sentence
    
    # Associated entity (if any)
    entity_id = Column(String(36), ForeignKey("enriched_entities.id"), nullable=True)
    
    # Processing metadata
    detection_confidence = Column(Float, nullable=False)
    feature_vector = Column(JSON, nullable=True)  # Raw feature data for analysis
    
    # Image selection
    selected_for_image = Column(Boolean, default=False)
    image_selection_reason = Column(String(100), nullable=True)  # Why this point was selected
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="emphasis_points_rel")
    entity = relationship("EnrichedEntity")

class UserPreferences(Base):
    """
    User preferences for style, processing options, and customizations.
    """
    __tablename__ = "user_preferences"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # User identification
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Style preferences
    preferred_style_templates = Column(JSON, nullable=True)  # Favorite template IDs
    custom_color_schemes = Column(JSON, nullable=True)
    default_animation_intensity = Column(Float, default=1.0)
    
    # Processing preferences
    default_target_platform = Column(String(20), default="tiktok")
    emphasis_sensitivity = Column(Float, default=0.3)
    max_images_per_video = Column(Integer, default=8)
    
    # Quality preferences
    min_image_quality = Column(Float, default=0.6)
    prefer_high_resolution = Column(Boolean, default=True)
    avoid_text_in_images = Column(Boolean, default=False)
    
    # Usage statistics
    total_videos_processed = Column(Integer, default=0)
    total_processing_time = Column(Float, default=0.0)
    favorite_entities = Column(JSON, nullable=True)  # Most used entities
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class APIUsage(Base):
    """
    Track API usage for billing and rate limiting.
    """
    __tablename__ = "api_usage"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # User and API information
    user_id = Column(String(255), nullable=False, index=True)
    api_endpoint = Column(String(100), nullable=False)
    request_method = Column(String(10), nullable=False)
    
    # Request details
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Status and billing
    status_code = Column(Integer, nullable=False)
    billable = Column(Boolean, default=True)
    cost_cents = Column(Integer, nullable=True)  # Cost in cents
    
    # Rate limiting
    rate_limit_key = Column(String(100), nullable=True)  # For grouping related requests
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True) 