"""
Health check endpoints for the Video Enhancement SaaS API.

Provides system health monitoring and diagnostics.
"""

import os
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# Import database check and settings
from app.database.connection import check_database_connection
from app.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        System health status
    """
    
    try:
        # Check database connection
        db_healthy = check_database_connection()
        
        # Check required settings (now properly loaded from .env)
        settings_healthy = bool(settings.database_url)
        
        # Determine overall health
        healthy = db_healthy and settings_healthy
        
        status_code = 200 if healthy else 503
        
        response = {
            "status": "healthy" if healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "checks": {
                "database": "healthy" if db_healthy else "unhealthy",
                "settings": "healthy" if settings_healthy else "unhealthy"
            }
        }
        
        return JSONResponse(content=response, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check failed"
            },
            status_code=503
        )

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with component status.
    
    Returns:
        Detailed system health information
    """
    
    try:
        # Database check
        db_healthy = check_database_connection()
        
        # Settings check (loaded from .env file)
        env_vars = {
            "DATABASE_URL": bool(settings.database_url),
            "UNSPLASH_API_KEY": bool(settings.unsplash_api_key),
            "REDIS_URL": bool(settings.redis_url),
            "AWS_ACCESS_KEY_ID": bool(settings.aws_access_key_id),
            "AWS_SECRET_ACCESS_KEY": bool(settings.aws_secret_access_key)
        }
        
        # API providers status
        providers = {
            "unsplash": bool(settings.unsplash_api_key),
            "database": db_healthy
        }
        
        # System resources (basic check)
        disk_space_available = True  # Simplified for MVP
        memory_available = True      # Simplified for MVP
        
        # Calculate overall health
        critical_services = ["database"]
        critical_healthy = all(providers.get(service, False) for service in critical_services)
        
        overall_healthy = critical_healthy and disk_space_available and memory_available
        
        response = {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "critical": True
                },
                "image_providers": {
                    "status": "healthy" if providers["unsplash"] else "degraded",
                    "critical": False,
                    "details": {
                        "unsplash": "configured" if providers["unsplash"] else "not_configured"
                    }
                },
                "storage": {
                    "status": "healthy" if env_vars.get("AWS_ACCESS_KEY_ID") else "degraded",
                    "critical": False,
                    "details": {
                        "aws_s3": "configured" if env_vars.get("AWS_ACCESS_KEY_ID") else "not_configured"
                    }
                },
                "system_resources": {
                    "status": "healthy",
                    "critical": True,
                    "details": {
                        "disk_space": "available" if disk_space_available else "limited",
                        "memory": "available" if memory_available else "limited"
                    }
                }
            },
            "environment_variables": env_vars
        }
        
        status_code = 200 if overall_healthy else 200  # Return 200 even for degraded
        
        return JSONResponse(content=response, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check failed"
            },
            status_code=503
        )

@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes-style readiness check.
    
    Returns:
        200 if ready to serve traffic, 503 if not
    """
    
    try:
        # Check critical dependencies
        db_healthy = check_database_connection()
        
        if not db_healthy:
            return JSONResponse(
                content={"ready": False, "reason": "Database not available"},
                status_code=503
            )
        
        return JSONResponse(
            content={"ready": True, "timestamp": datetime.utcnow().isoformat()},
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={"ready": False, "reason": "Service not ready"},
            status_code=503
        )

@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes-style liveness check.
    
    Returns:
        200 if application is alive, 503 if it should be restarted
    """
    
    return JSONResponse(
        content={
            "alive": True,
            "timestamp": datetime.utcnow().isoformat()
        },
        status_code=200
    ) 