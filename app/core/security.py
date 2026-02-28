"""
CARELINK AI Microservice - Security Module

Production-grade security features including API key authentication,
request validation, and security headers.
"""

import secrets
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.settings import get_settings
from app.core.logging import logger

settings = get_settings()

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from request header.
    
    Args:
        api_key: API key from request header
        
    Returns:
        str: Validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not settings.API_KEY_ENABLED:
        # API key authentication disabled
        return "disabled"
    
    if api_key is None:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "X-API-Key"}
        )
    
    if not secrets.compare_digest(api_key, settings.API_KEY):
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "X-API-Key"}
        )
    
    return api_key


def sanitize_input(text: str, max_length: int = 5000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
        
    Raises:
        ValueError: If input exceeds maximum length
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Check length
    if len(text) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_age(age: int) -> None:
    """
    Validate age input.
    
    Args:
        age: Patient age
        
    Raises:
        ValueError: If age is invalid
    """
    if not isinstance(age, int):
        raise ValueError("Age must be an integer")
    
    if age < 0 or age > 150:
        raise ValueError("Age must be between 0 and 150")


def validate_duration(duration_days: int) -> None:
    """
    Validate symptom duration.
    
    Args:
        duration_days: Duration in days
        
    Raises:
        ValueError: If duration is invalid
    """
    if not isinstance(duration_days, int):
        raise ValueError("Duration must be an integer")
    
    if duration_days < 0 or duration_days > 365:
        raise ValueError("Duration must be between 0 and 365 days")


class SecurityHeaders:
    """
    Security headers middleware for production deployment.
    """
    
    @staticmethod
    def get_headers() -> dict:
        """Get recommended security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }
