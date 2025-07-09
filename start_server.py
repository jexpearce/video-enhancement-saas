#!/usr/bin/env python3
import subprocess
import sys
import os

# Set up environment
os.chdir('/Users/jexpearce/video-enhancement-saas/backend')

# Start uvicorn with proper logging
cmd = [
    sys.executable, "-m", "uvicorn", 
    "app.main:app", 
    "--host", "0.0.0.0", 
    "--port", "8000", 
    "--reload",
    "--log-level", "info"
]

print("🚀 Starting Video Enhancement SaaS Backend...")
print("📋 Command:", " ".join(cmd))
print("🔗 Server will be available at: http://localhost:8000")
print("=" * 50)

try:
    # Run uvicorn with real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            sys.stdout.flush()
        
except KeyboardInterrupt:
    print("\n🛑 Shutting down server...")
    process.terminate()
except Exception as e:
    print(f"❌ Error starting server: {e}") 