"""
CARELINK AI Microservice - Model Service

ML model loading, inference, and lifecycle management.
Handles scikit-learn model loading with graceful fallback.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple
from pathlib import Path

from app.core.logging import logger, log_model_load
from app.core.settings import get_settings

settings = get_settings()


class ModelService:
    """
    ML model service for risk classification.
    
    Responsibilities:
    - Load trained model from disk
    - Perform inference
    - Model version tracking
    - Graceful fallback handling
    """
    
    def __init__(self):
        """Initialize model service."""
        self.model = None
        self.model_version = settings.MODEL_VERSION
        self.metadata = {}
        self.is_loaded = False
        self.loaded_at: Optional[datetime] = None
        
        logger.info("ModelService initialized")
    
    async def load_model(self) -> bool:
        """
        Load ML model from disk.
        
        Returns:
            bool: True if model loaded successfully
        """
        model_path = Path(settings.MODEL_PATH)
        metadata_path = Path(settings.MODEL_METADATA_PATH)
        
        try:
            # Check if model file exists
            if not model_path.exists():
                logger.warning(
                    f"Model file not found: {model_path}. "
                    f"Service will use fallback predictions."
                )
                self.is_loaded = False
                return False
            
            # Load model
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Load metadata if available
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.model_version = self.metadata.get("version", self.model_version)
            
            self.is_loaded = True
            self.loaded_at = datetime.utcnow()
            
            log_model_load(
                model_version=self.model_version,
                success=True,
                model_type=type(self.model).__name__
            )
            
            logger.info(
                f"Model loaded successfully: {self.model_version} "
                f"(type: {type(self.model).__name__})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            log_model_load(
                model_version=self.model_version,
                success=False,
                error=str(e)
            )
            self.is_loaded = False
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Perform risk prediction using loaded model.
        
        Args:
            features: Feature vector (numpy array)
            
        Returns:
            Tuple[str, Dict[str, float]]: (prediction, probabilities)
        """
        if not self.is_loaded or self.model is None:
            # Fallback prediction
            logger.warning("Model not loaded, using fallback prediction")
            return self._fallback_prediction(features)
        
        try:
            # Reshape if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Get prediction
            prediction_idx = self.model.predict(features)[0]
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
            else:
                # If model doesn't support probabilities, create one-hot
                probabilities = np.zeros(3)
                probabilities[prediction_idx] = 1.0
            
            # Map to risk levels
            risk_levels = ["LOW", "MEDIUM", "HIGH"]
            prediction = risk_levels[prediction_idx]
            
            probability_dict = {
                "low": float(probabilities[0]),
                "medium": float(probabilities[1]),
                "high": float(probabilities[2])
            }
            
            logger.debug(
                f"Model prediction: {prediction} "
                f"(probabilities: {probability_dict})"
            )
            
            return prediction, probability_dict
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Fallback prediction when model is not available.
        
        Uses heuristic-based approach based on feature analysis.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple[str, Dict[str, float]]: (prediction, probabilities)
        """
        logger.warning("Using fallback prediction logic")
        
        # Simple heuristic based on feature analysis
        # Feature 59 is symptom severity (index 59)
        # Feature 58 is symptom count (index 58)
        
        try:
            severity = features[59] if len(features) > 59 else 0.3
            symptom_count = features[58] if len(features) > 58 else 0.2
            age_normalized = features[50] if len(features > 50) else 0.4
            
            # Simple scoring
            risk_score = 0.4 * severity + 0.3 * symptom_count + 0.3 * age_normalized
            
            # Map to risk levels
            if risk_score < 0.33:
                prediction = "LOW"
                probabilities = {"low": 0.7, "medium": 0.2, "high": 0.1}
            elif risk_score < 0.66:
                prediction = "MEDIUM"
                probabilities = {"low": 0.2, "medium": 0.6, "high": 0.2}
            else:
                prediction = "HIGH"
                probabilities = {"low": 0.1, "medium": 0.2, "high": 0.7}
            
            logger.info(
                f"Fallback prediction: {prediction} "
                f"(risk_score={risk_score:.2f})"
            )
            
            return prediction, probabilities
            
        except Exception as e:
            logger.error(f"Fallback prediction error: {str(e)}")
            # Ultra-conservative fallback
            return "MEDIUM", {"low": 0.1, "medium": 0.5, "high": 0.4}
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            dict: Model metadata
        """
        return {
            "version": self.model_version,
            "is_loaded": self.is_loaded,
            "model_type": type(self.model).__name__ if self.model else None,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "metadata": self.metadata
        }
