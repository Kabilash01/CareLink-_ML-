"""
CARELINK AI Microservice - Application Configuration

Centralized configuration management using environment variables
and secure defaults for production deployment.
"""

from typing import List, Dict
from functools import lru_cache


# Model Configuration
MODEL_PATH = "app/models/risk_model.pkl"
MODEL_METADATA_PATH = "app/models/model_metadata.json"
MODEL_VERSION = "v1.0.0"

# NLP Configuration
SPACY_MODEL = "en_core_web_sm"
SUPPORTED_LANGUAGES = ["en", "es", "fr"]

# LLM Configuration
LLM_MODEL_NAME = "facebook/bart-large-cnn"
LLM_MAX_LENGTH = 512
LLM_TEMPERATURE = 0.7

# Risk Classification Thresholds
RISK_THRESHOLDS = {
    "low": 0.33,
    "medium": 0.66,
    "high": 1.0
}

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.6
ESCALATION_THRESHOLD = 0.75
EMERGENCY_FLAG_THRESHOLD = 0.75

# Clinical Safety Rules
SAFETY_RULES_ENABLED = True

# Feature Engineering
FEATURE_VECTOR_SIZE = 100
MAX_SYMPTOMS = 20

# API Rate Limiting
RATE_LIMIT_PER_MINUTE = 60

# Monitoring
ENABLE_METRICS = True
ENABLE_AUDIT_LOGGING = True

# Prediction Classes
RISK_CLASSES = ["LOW", "MEDIUM", "HIGH"]
CLASS_INDICES = {"low": 0, "medium": 1, "high": 2}

# Critical Symptoms for Rule Engine
CRITICAL_SYMPTOMS = [
    "chest pain",
    "difficulty breathing",
    "loss of consciousness",
    "severe bleeding",
    "stroke symptoms",
    "severe allergic reaction"
]

# High-Risk Combinations
HIGH_RISK_COMBINATIONS = [
    ["chest pain", "sweating"],
    ["chest pain", "shortness of breath"],
    ["pregnancy", "bleeding"],
    ["severe headache", "vision changes"]
]

# Minimum Risk Overrides
MINIMUM_RISK_OVERRIDES = {
    "infant_fever": "MEDIUM",
    "pregnancy_complications": "MEDIUM",
    "elderly_fall": "MEDIUM"
}
