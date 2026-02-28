"""
CARELINK AI Microservice - API Dependencies

Dependency injection functions for FastAPI routes.
Provides reusable dependencies for authentication, rate limiting, and services.
"""

import uuid
import time
from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from app.core.security import verify_api_key
from app.core.settings import get_settings
from app.core.logging import logger

settings = get_settings()

# Simple in-memory rate limiter (use Redis in production)
_rate_limit_cache = {}


def get_request_id(request: Request) -> str:
    """
    Generate or extract request ID for tracking.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Unique request ID
    """
    # Try to get from header first
    request_id = request.headers.get("X-Request-ID")
    
    if not request_id:
        # Generate new request ID
        request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    return request_id


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Client IP address
    """
    # Check for forwarded IP (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check for real IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client
    return request.client.host if request.client else "unknown"


async def check_rate_limit(
    request: Request,
    request_id: str = Depends(get_request_id)
) -> None:
    """
    Simple rate limiting dependency.
    
    In production, use Redis-based rate limiting with sliding window.
    
    Args:
        request: FastAPI request object
        request_id: Request ID from dependency
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = get_client_ip(request)
    current_time = time.time()
    
    # Clean old entries (older than 60 seconds)
    _rate_limit_cache[client_ip] = [
        ts for ts in _rate_limit_cache.get(client_ip, [])
        if current_time - ts < 60
    ]
    
    # Check rate limit
    request_count = len(_rate_limit_cache.get(client_ip, []))
    if request_count >= settings.RATE_LIMIT_PER_MINUTE:
        logger.warning(
            f"Rate limit exceeded for IP: {client_ip}",
            extra={"request_id": request_id, "client_ip": client_ip}
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )
    
    # Add current request
    if client_ip not in _rate_limit_cache:
        _rate_limit_cache[client_ip] = []
    _rate_limit_cache[client_ip].append(current_time)


async def validate_request(
    request: Request,
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key)
) -> dict:
    """
    Combined validation dependency for all protected endpoints.
    
    Args:
        request: FastAPI request object
        request_id: Request ID from dependency
        api_key: Validated API key
        
    Returns:
        dict: Request context with metadata
    """
    client_ip = get_client_ip(request)
    
    context = {
        "request_id": request_id,
        "client_ip": client_ip,
        "api_key_valid": api_key != "disabled",
        "timestamp": time.time()
    }
    
    logger.debug(
        "Request validated",
        extra=context
    )
    
    return context


class RequestTimer:
    """Dependency for measuring request duration."""
    
    def __init__(self):
        self.start_time = None
    
    async def __call__(self, request: Request):
        """Start timer when request begins."""
        self.start_time = time.time()
        return self
    
    def get_duration_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000


def get_timer() -> RequestTimer:
    """Get request timer dependency."""
    return RequestTimer()
