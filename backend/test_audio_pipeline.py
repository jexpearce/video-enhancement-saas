#!/usr/bin/env python3
"""
Simple test script for the audio processing pipeline

This script tests the core audio processing functionality
without requiring all ML dependencies to be installed.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test if we can import our core modules"""
    print("Testing imports...")
    
    try:
        from app.config import settings
        print("✅ Config imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from app.models.schemas import ProcessedAudio, AudioSegment, JobStatus, EntityType
        print("✅ Schemas imported successfully")
    except ImportError as e:
        print(f"❌ Schema import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration settings"""
    print("\nTesting configuration...")
    
    try:
        from app.config import settings
        
        print(f"Target sample rate: {settings.target_sample_rate}")
        print(f"Whisper model: {settings.whisper_model}")
        print(f"Max video duration: {settings.max_video_duration}")
        print(f"Audio chunk duration: {settings.audio_chunk_duration}")
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_schemas():
    """Test Pydantic schemas"""
    print("\nTesting schemas...")
    
    try:
        from app.models.schemas import (
            ProcessedAudio, AudioSegment, TranscriptionResult, 
            WordInfo, EmphasisPoint, EnrichedEntity, ProcessingJob
        )
        
        # Test creating a basic schema
        word = WordInfo(
            text="test",
            start=1.0,
            end=2.0,
            confidence=0.95
        )
        
        segment = AudioSegment(
            path="/tmp/test.wav",
            start_time=0.0,
            end_time=30.0,
            duration=30.0
        )
        
        print(f"Created word: {word.text} ({word.start}s - {word.end}s)")
        print(f"Created segment: {segment.duration}s duration")
        print("✅ Schema test passed")
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "app",
        "app/services",
        "app/services/audio",
        "app/services/transcription", 
        "app/services/emphasis",
        "app/services/nlp",
        "app/services/processing",
        "app/models",
        "app/api",
        "app/database"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required directories exist")
    return True

def test_file_structure():
    """Test that key files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "app/__init__.py",
        "app/config.py",
        "app/models/__init__.py",
        "app/models/schemas.py",
        "app/services/__init__.py",
        "app/services/audio/__init__.py",
        "app/services/audio/processor.py",
        "app/services/audio/quality_analyzer.py",
        "app/services/transcription/__init__.py",
        "app/services/transcription/whisper_service.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def main():
    """Run all tests"""
    print("🧪 Testing Video Enhancement SaaS - Phase 1 Setup")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_file_structure,
        test_imports,
        test_config,
        test_schemas
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Phase 1 foundation is ready!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up database: Create PostgreSQL database")
        print("3. Start Redis server")
        print("4. Run the application")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 