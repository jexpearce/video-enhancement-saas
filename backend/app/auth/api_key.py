"""
API Key Authentication System

Simple and secure API key authentication for the Video Enhancement SaaS.
Supports rate limiting, usage tracking, and user identification.
"""

import os
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import logging

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..database.models import APIUsage

logger = logging.getLogger(__name__)

# Security configuration
API_KEY_LENGTH = 32
API_KEY_PREFIX = "vea_"  # Video Enhancement API prefix
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")  # For admin operations

# Rate limiting configuration
DEFAULT_RATE_LIMIT_PER_HOUR = 100
DEFAULT_RATE_LIMIT_PER_DAY = 1000

security = HTTPBearer(auto_error=False)

class APIKeyError(Exception):
    """API Key related errors"""
    pass

class RateLimitError(Exception):
    """Rate limit exceeded errors"""
    pass

def generate_api_key() -> str:
    """Generate a new API key with proper format and security."""
    random_bytes = secrets.token_bytes(API_KEY_LENGTH)
    key_suffix = random_bytes.hex()
    return f"{API_KEY_PREFIX}{key_suffix}"

def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format without checking if it exists."""
    if not api_key:
        return False
    
    if not api_key.startswith(API_KEY_PREFIX):
        return False
    
    expected_length = len(API_KEY_PREFIX) + (API_KEY_LENGTH * 2)
    return len(api_key) == expected_length

def get_user_from_api_key(api_key: str) -> Optional[str]:
    """
    Get user ID from API key.
    
    In production, this would query a users/api_keys table.
    For now, we'll use a simple mapping with environment variables.
    """
    
    # Check if it's the admin key
    if ADMIN_API_KEY and api_key == ADMIN_API_KEY:
        return "admin"
    
    # For demo purposes, accept keys that match the format
    # In production, you'd query: SELECT user_id FROM api_keys WHERE key_hash = hash_api_key(api_key) AND active = true
    if validate_api_key_format(api_key):
        # Extract a user ID from the key (demo purposes)
        key_hash = hash_api_key(api_key)
        return f"user_{key_hash[:8]}"
    
    return None

def check_rate_limit(user_id: str, endpoint: str, db: Session) -> bool:
    """
    Check if user has exceeded rate limits.
    
    Args:
        user_id: User identifier
        endpoint: API endpoint being accessed
        db: Database session
        
    Returns:
        True if within limits, False if exceeded
    """
    
    try:
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Count requests in the last hour
        hourly_count = db.query(APIUsage).filter(
            APIUsage.user_id == user_id,
            APIUsage.created_at >= hour_ago,
            APIUsage.billable == True
        ).count()
        
        # Count requests in the last day
        daily_count = db.query(APIUsage).filter(
            APIUsage.user_id == user_id,
            APIUsage.created_at >= day_ago,
            APIUsage.billable == True
        ).count()
        
        # Check limits (in production, these would be per-user settings)
        hourly_limit = DEFAULT_RATE_LIMIT_PER_HOUR
        daily_limit = DEFAULT_RATE_LIMIT_PER_DAY
        
        # Admin has higher limits
        if user_id == "admin":
            hourly_limit *= 10
            daily_limit *= 10
        
        if hourly_count >= hourly_limit:
            logger.warning(f"Hourly rate limit exceeded for user {user_id}: {hourly_count}/{hourly_limit}")
            return False
            
        if daily_count >= daily_limit:
            logger.warning(f"Daily rate limit exceeded for user {user_id}: {daily_count}/{daily_limit}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Rate limit check failed for user {user_id}: {e}")
        # Fail open for now - allow request if rate limit check fails
        return True

def log_api_usage(
    user_id: str,
    endpoint: str,
    method: str,
    status_code: int,
    processing_time_ms: int,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None,
    billable: bool = True,
    db: Optional[Session] = None
):
    """Log API usage for billing and monitoring."""
    
    if not db:
        return
        
    try:
        usage_record = APIUsage(
            user_id=user_id,
            api_endpoint=endpoint,
            request_method=method,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            processing_time_ms=processing_time_ms,
            status_code=status_code,
            billable=billable,
            rate_limit_key=f"{user_id}:{endpoint}"
        )
        
        db.add(usage_record)
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to log API usage: {e}")

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_endpoint(current_user: dict = Depends(get_current_user)):
            return {"user_id": current_user["user_id"]}
    """
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    
    # Validate API key format
    if not validate_api_key_format(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from API key
    user_id = get_user_from_api_key(api_key)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check rate limits
    if not check_rate_limit(user_id, "general", db):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": "3600",  # 1 hour
                "X-RateLimit-Limit": str(DEFAULT_RATE_LIMIT_PER_HOUR),
                "X-RateLimit-Remaining": "0"
            }
        )
    
    return {
        "user_id": user_id,
        "api_key": api_key,
        "is_admin": user_id == "admin"
    }

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication - doesn't raise error if no credentials provided.
    Useful for endpoints that have different behavior for authenticated vs anonymous users.
    """
    
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None

async def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency that requires admin privileges.
    
    Usage:
        @app.delete("/admin/cleanup")
        async def admin_cleanup(admin_user: dict = Depends(require_admin)):
            # Only admins can access this
    """
    
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user

class APIKeyMiddleware:
    """
    Middleware to track API usage across all endpoints.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = datetime.now(timezone.utc)
        
        # Track response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Could extract user info here if needed
                pass
            elif message["type"] == "http.response.body":
                end_time = datetime.now(timezone.utc)
                processing_time = int((end_time - start_time).total_seconds() * 1000)
                
                # Log usage if this was an authenticated request
                # This could be enhanced to extract user info from the request
                
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

# Demo API key generation utilities
def create_demo_keys() -> Dict[str, str]:
    """Create demo API keys for testing."""
    
    demo_keys = {}
    
    # Generate a few demo keys
    for i in range(3):
        key = generate_api_key()
        user_id = f"demo_user_{i+1}"
        demo_keys[user_id] = key
    
    return demo_keys

def get_demo_environment_setup() -> str:
    """Get environment variable setup for demo keys."""
    
    if not ADMIN_API_KEY:
        admin_key = generate_api_key()
        setup_text = f"""
# Add these to your .env file for testing:

ADMIN_API_KEY={admin_key}

# Demo API keys (for testing):
"""
        
        demo_keys = create_demo_keys()
        for user_id, key in demo_keys.items():
            setup_text += f"# {user_id}: {key}\n"
        
        setup_text += f"""
# Usage examples:
# curl -H "Authorization: Bearer {admin_key}" http://localhost:8000/api/v1/health/detailed
# curl -H "Authorization: Bearer {list(demo_keys.values())[0]}" http://localhost:8000/api/v1/jobs
"""
        
        return setup_text
    
    return "ADMIN_API_KEY is already configured" 