"""
CARELINK AI Microservice - Helper Utilities

Utility functions for common operations.
"""

import hashlib
import secrets
from typing import Any, Dict
from datetime import datetime, timedelta


def generate_request_id() -> str:
    """
    Generate unique request ID.
    
    Returns:
        str: Unique request ID
    """
    return f"req_{secrets.token_hex(8)}"


def hash_string(text: str) -> str:
    """
    Generate SHA-256 hash of string.
    
    Args:
        text: Input string
        
    Returns:
        str: Hex digest of hash
    """
    return hashlib.sha256(text.encode()).hexdigest()


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        float: Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def sanitize_dict(data: Dict[str, Any], remove_keys: list = None) -> Dict[str, Any]:
    """
    Sanitize dictionary by removing sensitive keys.
    
    Args:
        data: Input dictionary
        remove_keys: Keys to remove
        
    Returns:
        Dict: Sanitized dictionary
    """
    if remove_keys is None:
        remove_keys = ["password", "api_key", "secret", "token"]
    
    sanitized = {}
    for key, value in data.items():
        if key.lower() not in remove_keys:
            if isinstance(value, dict):
                sanitized[key] = sanitize_dict(value, remove_keys)
            else:
                sanitized[key] = value
    
    return sanitized


def parse_iso_datetime(datetime_str: str) -> datetime:
    """
    Parse ISO format datetime string.
    
    Args:
        datetime_str: ISO format datetime string
        
    Returns:
        datetime: Parsed datetime object
    """
    # Remove 'Z' suffix and parse
    datetime_str = datetime_str.rstrip('Z')
    return datetime.fromisoformat(datetime_str)


def get_age_category(age: int) -> str:
    """
    Get age category from age.
    
    Args:
        age: Age in years
        
    Returns:
        str: Age category
    """
    if age < 2:
        return "infant"
    elif age < 12:
        return "child"
    elif age < 18:
        return "adolescent"
    elif age < 65:
        return "adult"
    else:
        return "elderly"
