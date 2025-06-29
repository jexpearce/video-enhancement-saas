"""
Configuration settings for the services module.
"""

import os
from pathlib import Path

class Settings:
    """Application settings."""
    
    # Audio processing settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    
    # File paths
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp"))
    
    # Processing settings
    MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", "300"))  # 5 minutes
    
    # Enhancement settings
    ENHANCEMENT_LEVEL = os.getenv("ENHANCEMENT_LEVEL", "adaptive")

# Global settings instance
settings = Settings() 