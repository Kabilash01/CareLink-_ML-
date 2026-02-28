"""
CARELINK AI Microservice - Application Settings

Production-grade settings management using Pydantic Settings.
Validates environment variables and provides type-safe configuration.
"""

import os
from typing import List, Optional, Union
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All sensitive data should be passed through environment variables.
    """
    
    # Application Metadata
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    ENVIRONMENT: str = Field(default="development", description="Environment name")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format: json or text")
    
    # CORS
    CORS_ORIGINS: Union[str, List[str]] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default="app/models/risk_model.pkl",
        description="Path to ML model file"
    )
    MODEL_VERSION: str = Field(default="v1.0.0", description="Model version")
    MODEL_METADATA_PATH: str = Field(
        default="app/models/model_metadata.json",
        description="Path to model metadata"
    )
    
    # NLP Configuration
    SPACY_MODEL: str = Field(
        default="en_core_web_sm",
        description="spaCy model name"
    )
    SUPPORTED_LANGUAGES: Union[str, List[str]] = Field(
        default=["en", "es", "fr"],
        description="Supported languages"
    )
    
    # LLM Configuration — HuggingFace Transformers + BitsAndBytes NF4
    LLM_HF_MODEL_ID: str = Field(
        default="google/medgemma-4b-it",
        description="HuggingFace model ID to load with BitsAndBytes NF4"
    )
    LLM_MAX_NEW_TOKENS: int = Field(default=200, description="Max tokens to generate (max_new_tokens only — never max_length)")
    LLM_MAX_LENGTH: int = Field(default=512, description="Max input token budget for tokenizer truncation")
    LLM_TEMPERATURE: float = Field(default=0.25, description="Sampling temperature — lower is more deterministic")
    LLM_TOP_P: float = Field(default=0.9, description="Nucleus sampling ceiling")
    LLM_REPEAT_PENALTY: float = Field(default=1.1, description="Repetition penalty (1.0 = off)")
    HUGGINGFACE_TOKEN: Optional[str] = Field(
        default=None,
        description="HuggingFace token for accessing gated models (e.g. MedGemma)"
    )

    # Security
    API_KEY_ENABLED: bool = Field(default=False, description="Enable API key authentication")
    API_KEY: Optional[str] = Field(default=None, description="API key for authentication")
    
    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./carelink.db",
        description="SQLite database connection string"
    )
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    ENABLE_AUDIT_LOGGING: bool = Field(default=True, description="Enable audit logging")
    
    # Feature Flags
    SAFETY_RULES_ENABLED: bool = Field(default=True, description="Enable safety rules")
    CONFIDENCE_CONTROLLER_ENABLED: bool = Field(
        default=True,
        description="Enable confidence controller"
    )
    EXPLANATION_SERVICE_ENABLED: bool = Field(
        default=True,
        description="Enable LLM explanations"
    )
    
    # Thresholds
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    ESCALATION_THRESHOLD: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence escalation threshold"
    )
    EMERGENCY_FLAG_THRESHOLD: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Emergency flag threshold"
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="API rate limit per minute"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("SUPPORTED_LANGUAGES", mode="before")
    @classmethod
    def parse_supported_languages(cls, v):
        """Parse supported languages from comma-separated string or list."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This ensures settings are loaded once and reused across the application.
    """
    return Settings()
