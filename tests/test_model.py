"""
CARELINK AI Microservice - Model Service Tests

Tests for ML model loading and inference.
"""

import pytest
import numpy as np
from app.services.model_service import ModelService


class TestModelService:
    """Test model service functionality."""
    
    @pytest.fixture
    def model_service(self):
        """Create model service instance."""
        return ModelService()
    
    def test_model_initialization(self, model_service):
        """Test model service initialization."""
        assert model_service is not None
        assert model_service.model_version is not None
    
    def test_fallback_prediction(self, model_service):
        """Test fallback prediction when model not loaded."""
        # Create dummy feature vector
        features = np.random.random(100)
        
        prediction, probabilities = model_service.predict(features)
        
        assert prediction in ["LOW", "MEDIUM", "HIGH"]
        assert "low" in probabilities
        assert "medium" in probabilities
        assert "high" in probabilities
        
        # Check probabilities sum to ~1.0
        prob_sum = sum(probabilities.values())
        assert 0.99 <= prob_sum <= 1.01
    
    def test_get_model_info(self, model_service):
        """Test get model info."""
        info = model_service.get_model_info()
        
        assert "version" in info
        assert "is_loaded" in info
        assert "model_type" in info
