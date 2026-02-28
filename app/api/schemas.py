"""
CARELINK AI Microservice - API Schemas

Pydantic models for request/response validation and documentation.
All API contracts are defined here for type safety and OpenAPI generation.
"""

from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Language(str, Enum):
    """Supported languages."""
    EN = "en"
    ES = "es"
    FR = "fr"


class TriageRequest(BaseModel):
    """
    Request schema for symptom triage prediction.
    
    Example:
        {
            "symptoms_text": "I have chest pain and sweating",
            "age": 52,
            "duration_days": 1,
            "chronic_conditions": ["diabetes"],
            "language": "en"
        }
    """
    symptoms_text: str = Field(
        ...,
        description="Patient's symptom description in natural language",
        min_length=1,
        max_length=5000,
        example="I have chest pain and sweating"
    )
    age: Optional[int] = Field(
        default=30,
        description="Patient's age in years (extracted from text if omitted)",
        ge=0,
        le=150,
        example=52
    )
    duration_days: Optional[int] = Field(
        default=1,
        description="Duration of symptoms in days (extracted from text if omitted)",
        ge=0,
        le=365,
        example=1
    )
    chronic_conditions: Optional[List[str]] = Field(
        default=[],
        description="List of chronic medical conditions",
        example=["diabetes", "hypertension"]
    )
    language: Language = Field(
        default=Language.EN,
        description="Language code for symptom text and explanation",
        example="en"
    )
    
    @field_validator('symptoms_text')
    @classmethod
    def validate_symptoms_text(cls, v):
        """Validate symptoms text."""
        v = v.strip()
        if not v:
            raise ValueError("Symptoms text cannot be empty")
        return v
    
    @field_validator('chronic_conditions')
    @classmethod
    def validate_chronic_conditions(cls, v):
        """Validate chronic conditions list."""
        if v is None:
            return []
        # Convert to lowercase and remove duplicates
        return list(set([condition.lower().strip() for condition in v if condition]))
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms_text": "I have chest pain and sweating",
                "age": 52,
                "duration_days": 1,
                "chronic_conditions": ["diabetes"],
                "language": "en"
            }
        }


class ProbabilityDistribution(BaseModel):
    """Probability distribution across risk levels."""
    low: float = Field(..., ge=0.0, le=1.0, description="Probability of LOW risk")
    medium: float = Field(..., ge=0.0, le=1.0, description="Probability of MEDIUM risk")
    high: float = Field(..., ge=0.0, le=1.0, description="Probability of HIGH risk")
    
    @field_validator('high')
    @classmethod
    def validate_probabilities_sum(cls, v, info):
        """Ensure probabilities sum to approximately 1.0."""
        total = info.data.get('low', 0) + info.data.get('medium', 0) + v
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        return v


class TriageResponse(BaseModel):
    """
    Response schema for symptom triage prediction.
    
    Example:
        {
            "prediction": "HIGH",
            "probabilities": {
                "low": 0.03,
                "medium": 0.15,
                "high": 0.82
            },
            "confidence": 0.91,
            "rules_triggered": ["CHEST_PAIN_OVERRIDE"],
            "explanation": "Based on chest pain and sweating...",
            "model_version": "v1.0.0",
            "request_id": "abc123",
            "timestamp": "2026-02-27T10:30:00Z"
        }
    """
    prediction: RiskLevel = Field(
        ...,
        description="Final risk classification",
        example="HIGH"
    )
    probabilities: ProbabilityDistribution = Field(
        ...,
        description="Probability distribution across risk levels"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score",
        example=0.91
    )
    rules_triggered: List[str] = Field(
        default=[],
        description="List of clinical safety rules that were triggered",
        example=["CHEST_PAIN_OVERRIDE"]
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the prediction",
        example="Based on your chest pain and sweating symptoms..."
    )
    model_version: str = Field(
        ...,
        description="ML model version used for prediction",
        example="v1.0.0"
    )
    escalated: bool = Field(
        default=False,
        description="Whether prediction was escalated due to low confidence"
    )
    emergency_flag: bool = Field(
        default=False,
        description="Whether emergency care is recommended"
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracking",
        example="req_abc123xyz"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )

    model_config = ConfigDict(protected_namespaces=())


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[Dict] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status", example="healthy")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_version: Optional[str] = Field(default=None, description="ML model version")

    model_config = ConfigDict(protected_namespaces=())


class MetricsResponse(BaseModel):
    """Metrics endpoint response schema."""
    total_predictions: int = Field(..., description="Total predictions made")
    predictions_by_risk: Dict[str, int] = Field(..., description="Predictions grouped by risk level")
    average_confidence: float = Field(..., description="Average confidence score")
    rules_triggered_count: int = Field(..., description="Total rules triggered")
    escalations_count: int = Field(..., description="Total escalations")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ExtractedSymptom(BaseModel):
    """Extracted symptom entity."""
    text: str = Field(..., description="Symptom text")
    normalized: str = Field(..., description="Normalized symptom name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")


class DebugResponse(BaseModel):
    """Debug information response (development only)."""
    extracted_symptoms: List[ExtractedSymptom] = Field(..., description="Extracted symptoms")
    feature_vector_size: int = Field(..., description="Feature vector dimensions")
    raw_probabilities: Dict[str, float] = Field(..., description="Raw model probabilities")
    rules_evaluated: List[str] = Field(..., description="All rules evaluated")
    processing_time_ms: float = Field(..., description="Total processing time")
