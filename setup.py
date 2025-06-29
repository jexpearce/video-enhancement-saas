#!/usr/bin/env python3
"""
Setup script for Video Enhancement SaaS - Phase 1

This script helps users set up the development environment
and install necessary dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🎬 Video Enhancement SaaS - Phase 1 Setup")
    print("=" * 50)
    print("AI-powered video enhancement for content creators")
    print("=" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required. Current: {version.major}.{version.minor}")
        print("Please install Python 3.11 or newer")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
    return True

def check_system_dependencies():
    """Check if system dependencies are available"""
    print("\nChecking system dependencies...")
    
    dependencies = {
        'ffmpeg': 'FFmpeg (for audio processing)',
        'git': 'Git (for version control)'
    }
    
    missing = []
    for cmd, description in dependencies.items():
        try:
            subprocess.run([cmd, '--version'], 
                         capture_output=True, check=True)
            print(f"✅ {description}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {description}")
            missing.append(cmd)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("\nInstall instructions:")
        
        system = platform.system().lower()
        if system == 'darwin':  # macOS
            print("  brew install ffmpeg")
        elif system == 'linux':
            print("  sudo apt-get install ffmpeg  # Ubuntu/Debian")
            print("  sudo yum install ffmpeg      # CentOS/RHEL")
        else:
            print("  Please install FFmpeg for your system")
        
        return False
    
    return True

def create_virtual_environment():
    """Create Python virtual environment"""
    print("\nSetting up Python virtual environment...")
    
    venv_path = Path("backend/venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([
            sys.executable, '-m', 'venv', str(venv_path)
        ], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Determine pip path
    system = platform.system().lower()
    if system == 'windows':
        pip_path = "backend/venv/Scripts/pip"
    else:
        pip_path = "backend/venv/bin/pip"
    
    try:
        # Upgrade pip first
        subprocess.run([
            pip_path, 'install', '--upgrade', 'pip'
        ], check=True)
        
        # Install requirements
        subprocess.run([
            pip_path, 'install', '-r', 'backend/requirements.txt'
        ], check=True)
        
        print("✅ Python dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Try installing manually:")
        print("  cd backend")
        print("  source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
        print("  pip install -r requirements.txt")
        return False

def download_ml_models():
    """Download required ML models"""
    print("\nDownloading ML models...")
    
    # Determine python path
    system = platform.system().lower()
    if system == 'windows':
        python_path = "backend/venv/Scripts/python"
    else:
        python_path = "backend/venv/bin/python"
    
    try:
        # Download Whisper model
        print("Downloading Whisper base model (this may take a few minutes)...")
        subprocess.run([
            python_path, '-c', 
            'import whisper; whisper.load_model("base")'
        ], check=True)
        
        # Download spaCy model
        print("Downloading spaCy English model...")
        subprocess.run([
            python_path, '-m', 'spacy', 'download', 'en_core_web_sm'
        ], check=True)
        
        print("✅ ML models downloaded")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download models: {e}")
        print("You can download manually later:")
        print("  python -c 'import whisper; whisper.load_model(\"base\")'")
        print("  python -m spacy download en_core_web_sm")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    print("\nSetting up environment configuration...")
    
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    try:
        import secrets
        secret_key = secrets.token_hex(32)
        
        env_content = f"""# Video Enhancement SaaS Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/video_enhancement
REDIS_URL=redis://localhost:6379
SECRET_KEY={secret_key}
ENVIRONMENT=development
DEBUG=True
WHISPER_MODEL=base
MAX_VIDEO_DURATION=300
MAX_FILE_SIZE=524288000
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def run_tests():
    """Run basic tests to verify setup"""
    print("\nRunning setup tests...")
    
    # Determine python path
    system = platform.system().lower()
    if system == 'windows':
        python_path = "backend/venv/Scripts/python"
    else:
        python_path = "backend/venv/bin/python"
    
    try:
        result = subprocess.run([
            python_path, 'backend/test_audio_pipeline.py'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🚀 Setup Complete! Next Steps:")
    print("=" * 40)
    
    system = platform.system().lower()
    if system == 'windows':
        activate_cmd = "backend\\venv\\Scripts\\activate"
    else:
        activate_cmd = "source backend/venv/bin/activate"
    
    print(f"1. Activate virtual environment:")
    print(f"   {activate_cmd}")
    print()
    print("2. Start required services:")
    print("   - PostgreSQL: Create database 'video_enhancement'")
    print("   - Redis: Start Redis server")
    print()
    print("3. Start the application:")
    print("   cd backend")
    print("   docker-compose up  # OR")
    print("   uvicorn app.main:app --reload")
    print()
    print("4. Test the API:")
    print("   Visit: http://localhost:8000/docs")
    print()
    print("📚 Documentation: See README.md for detailed instructions")

def main():
    """Main setup function"""
    print_banner()
    
    steps = [
        ("Python Version", check_python_version),
        ("System Dependencies", check_system_dependencies),
        ("Virtual Environment", create_virtual_environment),
        ("Python Dependencies", install_python_dependencies),
        ("ML Models", download_ml_models),
        ("Environment Config", create_env_file),
        ("Setup Tests", run_tests)
    ]
    
    passed = 0
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if step_func():
            passed += 1
        else:
            print(f"❌ {step_name} failed")
            break
    
    print(f"\n{'='*50}")
    print(f"Setup Results: {passed}/{len(steps)} steps completed")
    
    if passed == len(steps):
        print("🎉 Setup completed successfully!")
        print_next_steps()
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 