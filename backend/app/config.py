from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/video_enhancement"
    redis_url: str = "redis://localhost:6379"
    
    # API Keys
    openai_api_key: Optional[str] = None
    
    # AWS (for later phases)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    
    # App Config
    secret_key: str = "dev-secret-key-change-in-production"
    environment: str = "development"
    debug: bool = True
    
    # Processing Config
    whisper_model: str = "base"  # Start with base, upgrade to large-v3 later
    max_video_duration: int = 300  # 5 minutes in seconds
    max_file_size: int = 500 * 1024 * 1024  # 500MB in bytes
    
    # Audio Processing Config
    target_sample_rate: int = 16000  # Optimal for Whisper
    audio_chunk_duration: int = 30  # seconds
    
    # Emphasis Detection Config
    emphasis_threshold: float = 0.7
    acoustic_weight: float = 0.6
    prosodic_weight: float = 0.2
    linguistic_weight: float = 0.15
    visual_weight: float = 0.05
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings() 