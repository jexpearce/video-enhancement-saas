"""
Main FastAPI application for Video Enhancement SaaS

Provides RESTful API endpoints for video processing, job management, and results retrieval.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import database setup
from app.database.connection import init_database, check_database_connection

# Import API routes
from app.api.v1 import videos, jobs, health

# Import authentication
from app.auth.api_key import get_demo_environment_setup, get_optional_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup/shutdown tasks.
    """
    # Startup
    logger.info("Starting Video Enhancement SaaS API...")
    
    # Check database connection
    if not check_database_connection():
        logger.error("Database connection failed!")
        raise RuntimeError("Cannot connect to database")
    
    # Initialize database tables
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Log demo API key setup
    demo_setup = get_demo_environment_setup()
    if "ADMIN_API_KEY is already configured" not in demo_setup:
        logger.info("=== API Key Setup ===")
        for line in demo_setup.split('\n'):
            if line.strip():
                logger.info(line)
        logger.info("======================")
    
    logger.info("API startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Video Enhancement SaaS API...")

# Create FastAPI application
app = FastAPI(
    title="Video Enhancement SaaS",
    description="""
    AI-powered video enhancement tool that automatically detects emphasized words 
    and adds relevant visual context to talking-head videos.
    
    Perfect for content creators making TikToks, Reels, or educational videos 
    about news, politics, or any topic.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Exception handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

# General exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Include API routers
app.include_router(videos.router, prefix="/api/v1", tags=["videos"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

# Root endpoint
@app.get("/")
async def root(current_user: dict = Depends(get_optional_user)):
    """Root endpoint with API information."""
    
    base_response = {
        "message": "Video Enhancement SaaS API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "upload_video": "/api/v1/videos/upload",
            "check_job": "/api/v1/jobs/{job_id}",
            "download_video": "/api/v1/videos/{job_id}/download"
        }
    }
    
    # Add authentication info
    if current_user:
        base_response["authenticated"] = True
        base_response["user_id"] = current_user["user_id"]
        base_response["is_admin"] = current_user.get("is_admin", False)
    else:
        base_response["authenticated"] = False
        base_response["auth_info"] = {
            "required": "API key required for most endpoints",
            "format": "Bearer vea_[64-character-hex-string]",
            "header": "Authorization: Bearer YOUR_API_KEY",
            "demo_setup": "/api/demo-setup"
        }
    
    return base_response

@app.get("/api/demo-setup")
async def get_api_demo_setup():
    """Get demo API key setup information."""
    return {
        "demo_setup": get_demo_environment_setup(),
        "instructions": [
            "1. Copy the generated ADMIN_API_KEY to your .env file",
            "2. Use the demo API keys for testing different users",
            "3. Include API key in Authorization header: 'Bearer YOUR_API_KEY'",
            "4. Test authenticated endpoint: curl -H 'Authorization: Bearer YOUR_KEY' http://localhost:8000/api/v1/health/detailed",
            "5. Visit /docs for interactive API documentation"
        ],
        "example_usage": {
            "curl": "curl -H 'Authorization: Bearer vea_...' http://localhost:8000/api/v1/jobs",
            "python": {
                "headers": {"Authorization": "Bearer vea_..."},
                "url": "http://localhost:8000/api/v1/videos/upload"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    ) 