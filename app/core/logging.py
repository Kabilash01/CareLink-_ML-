"""
CARELINK AI Microservice - Logging Configuration

Production-grade structured logging with JSON formatting,
correlation IDs, and audit trail support.
"""

import sys
import logging
from datetime import datetime
from typing import Any, Dict
from pythonjsonlogger import jsonlogger

from app.core.settings import get_settings

# Get settings
settings = get_settings()

# Configure logger
logger = logging.getLogger("carelink_ai")


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with additional context fields.
    """
    
    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict):
        """Add custom fields to log records."""
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service metadata
        log_record['service'] = 'carelink-ai-service'
        log_record['version'] = settings.APP_VERSION
        log_record['environment'] = settings.ENVIRONMENT
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add source location
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno


def setup_logging():
    """
    Initialize logging configuration for the application.
    Sets up structured JSON logging for production or human-readable logging for development.
    """
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Set formatter based on environment
    if settings.LOG_FORMAT == "json" or settings.is_production():
        # JSON formatter for production
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        # Human-readable formatter for development
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(
        "Logging initialized",
        extra={
            "log_level": settings.LOG_LEVEL,
            "log_format": settings.LOG_FORMAT,
            "environment": settings.ENVIRONMENT
        }
    )


def log_request(endpoint: str, method: str, request_id: str, **kwargs):
    """Log incoming API request."""
    logger.info(
        f"Incoming request: {method} {endpoint}",
        extra={
            "event_type": "api_request",
            "endpoint": endpoint,
            "method": method,
            "request_id": request_id,
            **kwargs
        }
    )


def log_response(endpoint: str, method: str, request_id: str, status_code: int, duration_ms: float):
    """Log API response."""
    logger.info(
        f"Response: {method} {endpoint} - {status_code}",
        extra={
            "event_type": "api_response",
            "endpoint": endpoint,
            "method": method,
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": duration_ms
        }
    )


def log_prediction(
    request_id: str,
    prediction: str,
    confidence: float,
    model_version: str,
    rules_triggered: list,
    **kwargs
):
    """Log ML prediction for audit trail."""
    logger.info(
        f"Prediction made: {prediction}",
        extra={
            "event_type": "ml_prediction",
            "request_id": request_id,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
            "rules_triggered": rules_triggered,
            **kwargs
        }
    )


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context."""
    logger.error(
        f"Error occurred: {str(error)}",
        extra={
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        },
        exc_info=True
    )


def log_model_load(model_version: str, success: bool, **kwargs):
    """Log model loading event."""
    level = logging.INFO if success else logging.ERROR
    logger.log(
        level,
        f"Model load {'successful' if success else 'failed'}: {model_version}",
        extra={
            "event_type": "model_load",
            "model_version": model_version,
            "success": success,
            **kwargs
        }
    )
