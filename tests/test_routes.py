"""
CARELINK AI Microservice - API Route Tests

Comprehensive tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestTriageEndpoint:
    """Test triage prediction endpoint."""
    
    def test_valid_triage_request(self):
        """Test valid triage request."""
        payload = {
            "symptoms_text": "I have chest pain and sweating",
            "age": 52,
            "duration_days": 1,
            "chronic_conditions": ["diabetes"],
            "language": "en"
        }
        
        response = client.post("/api/v1/triage/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["LOW", "MEDIUM", "HIGH"]
        assert "probabilities" in data
        assert "confidence" in data
        assert "explanation" in data
        assert "model_version" in data
        assert "request_id" in data
    
    def test_invalid_age(self):
        """Test invalid age input."""
        payload = {
            "symptoms_text": "I have a headache",
            "age": 200,  # Invalid age
            "duration_days": 1,
            "chronic_conditions": [],
            "language": "en"
        }
        
        response = client.post("/api/v1/triage/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_empty_symptoms(self):
        """Test empty symptoms text."""
        payload = {
            "symptoms_text": "",
            "age": 30,
            "duration_days": 1,
            "chronic_conditions": [],
            "language": "en"
        }
        
        response = client.post("/api/v1/triage/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_multilingual_request(self):
        """Test multilingual support."""
        payload = {
            "symptoms_text": "Tengo dolor de pecho",
            "age": 45,
            "duration_days": 1,
            "chronic_conditions": [],
            "language": "es"
        }
        
        response = client.post("/api/v1/triage/predict", json=payload)
        assert response.status_code == 200


class TestMetricsEndpoints:
    """Test metrics endpoints."""
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "predictions_by_risk" in data
        assert "average_confidence" in data
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        # May return 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]


class TestRulesEndpoint:
    """Test rules endpoint."""
    
    def test_get_rules(self):
        """Test get rules endpoint."""
        response = client.get("/api/v1/rules")
        assert response.status_code == 200
        
        data = response.json()
        assert "rules" in data
        assert "rules_enabled" in data
        assert isinstance(data["rules"], list)
