"""
CARELINK AI Microservice - Constants

Centralized constants and enumerations for the application.
"""

from enum import Enum


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SymptomCategory(str, Enum):
    """Medical symptom categories."""
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    GASTROINTESTINAL = "gastrointestinal"
    RESPIRATORY = "respiratory"
    MUSCULOSKELETAL = "musculoskeletal"
    GENERAL = "general"
    OTHER = "other"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"


# Error messages
ERROR_MESSAGES = {
    "INVALID_INPUT": "Invalid input data provided",
    "MODEL_NOT_LOADED": "ML model not loaded",
    "PROCESSING_ERROR": "Error processing request",
    "RATE_LIMIT_EXCEEDED": "Rate limit exceeded",
    "AUTHENTICATION_FAILED": "Authentication failed"
}

# HTTP status codes
HTTP_STATUS = {
    "SUCCESS": 200,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "NOT_FOUND": 404,
    "TOO_MANY_REQUESTS": 429,
    "INTERNAL_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503
}

# Model metadata
MODEL_METADATA_KEYS = [
    "version",
    "training_date",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "feature_count",
    "training_samples"
]

# Audit event types
AUDIT_EVENT_TYPES = [
    "prediction",
    "model_load",
    "error",
    "user_action",
    "system_event"
]
