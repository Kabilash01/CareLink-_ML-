"""
CARELINK AI Microservice - Feature Engineering

Feature extraction and engineering for ML model input.
Converts raw symptoms and patient data into structured feature vectors.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from app.nlp.symptom_extractor import ExtractedSymptom
from app.core.logging import logger
from app.config import FEATURE_VECTOR_SIZE


@dataclass
class FeatureVector:
    """Structured feature vector for ML model."""
    features: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


class FeatureEngineer:
    """
    Feature engineering for symptom triage model.
    
    Converts:
    - Extracted symptoms
    - Patient demographics
    - Symptom duration
    - Chronic conditions
    
    Into fixed-size feature vector for ML model.
    """
    
    def __init__(self):
        """Initialize feature engineer with symptom vocabulary."""
        self.symptom_vocabulary = self._build_symptom_vocabulary()
        self.feature_size = FEATURE_VECTOR_SIZE
        
        logger.info(f"FeatureEngineer initialized with {len(self.symptom_vocabulary)} symptoms")
    
    def _build_symptom_vocabulary(self) -> Dict[str, int]:
        """
        Build symptom vocabulary for one-hot encoding.
        
        In production, this should be loaded from the trained model's vocabulary
        to ensure consistency between training and inference.
        
        Returns:
            Dict[str, int]: Symptom to index mapping
        """
        # Core symptom vocabulary (should match training data)
        symptoms = [
            # Cardiovascular
            "chest pain", "heart palpitations", "shortness of breath",
            "irregular heartbeat", "rapid heartbeat",
            
            # Neurological
            "headache", "dizziness", "confusion", "seizure", "numbness",
            "weakness", "paralysis", "vision changes", "blurred vision",
            "loss of consciousness", "memory loss",
            
            # Gastrointestinal
            "nausea", "vomiting", "diarrhea", "constipation",
            "abdominal pain", "bloating", "loss of appetite",
            
            # Respiratory
            "cough", "wheezing", "sore throat", "difficulty breathing",
            
            # Musculoskeletal
            "joint pain", "muscle pain", "back pain", "neck pain",
            "stiffness", "swelling",
            
            # General
            "fever", "chills", "sweating", "fatigue",
            "weight loss", "bleeding", "rash", "itching",
            
            # Pain descriptors
            "severe pain", "sharp pain", "dull pain", "burning pain",
        ]
        
        return {symptom: idx for idx, symptom in enumerate(symptoms)}
    
    def extract_features(
        self,
        symptoms: List[ExtractedSymptom],
        age: int,
        duration_days: int,
        chronic_conditions: List[str]
    ) -> FeatureVector:
        """
        Extract features from patient data.
        
        Args:
            symptoms: List of extracted symptoms
            age: Patient age
            duration_days: Symptom duration
            chronic_conditions: List of chronic conditions
            
        Returns:
            FeatureVector: Structured feature vector
        """
        # Initialize feature array
        features = np.zeros(self.feature_size)
        feature_names = []
        
        # Feature 1-50: Symptom presence (one-hot encoded)
        symptom_features = self._encode_symptoms(symptoms)
        features[:len(symptom_features)] = symptom_features
        feature_names.extend([f"symptom_{i}" for i in range(len(symptom_features))])
        
        # Feature 51: Age (normalized)
        age_normalized = min(age / 100.0, 1.0)  # Normalize to [0, 1]
        features[50] = age_normalized
        feature_names.append("age_normalized")
        
        # Feature 52: Age group (categorical)
        age_group = self._get_age_group(age)
        features[51] = age_group
        feature_names.append("age_group")
        
        # Feature 53: Duration (normalized)
        duration_normalized = min(duration_days / 30.0, 1.0)  # Normalize to [0, 1]
        features[52] = duration_normalized
        feature_names.append("duration_normalized")
        
        # Feature 54-58: Chronic condition flags
        chronic_features = self._encode_chronic_conditions(chronic_conditions)
        features[53:53+len(chronic_features)] = chronic_features
        feature_names.extend([f"chronic_{i}" for i in range(len(chronic_features))])
        
        # Feature 59: Number of symptoms
        features[58] = min(len(symptoms) / 10.0, 1.0)  # Normalized
        feature_names.append("symptom_count")
        
        # Feature 60: Symptom severity indicator
        features[59] = self._calculate_symptom_severity(symptoms)
        feature_names.append("symptom_severity")
        
        # Additional engineered features (60-100)
        # Feature 61-70: Symptom category flags
        category_features = self._encode_symptom_categories(symptoms)
        features[60:60+len(category_features)] = category_features
        feature_names.extend([f"category_{i}" for i in range(len(category_features))])
        
        metadata = {
            "num_symptoms": len(symptoms),
            "symptom_names": [s.normalized for s in symptoms],
            "age_group": age_group,
            "has_chronic_conditions": len(chronic_conditions) > 0
        }
        
        logger.debug(f"Extracted {len(features)} features")
        
        return FeatureVector(
            features=features,
            feature_names=feature_names,
            metadata=metadata
        )
    
    def _encode_symptoms(self, symptoms: List[ExtractedSymptom]) -> np.ndarray:
        """
        Encode symptoms as one-hot vector.
        
        Args:
            symptoms: List of extracted symptoms
            
        Returns:
            np.ndarray: One-hot encoded symptom vector
        """
        vector = np.zeros(len(self.symptom_vocabulary))
        
        for symptom in symptoms:
            if symptom.normalized in self.symptom_vocabulary:
                idx = self.symptom_vocabulary[symptom.normalized]
                vector[idx] = symptom.confidence
        
        return vector
    
    def _get_age_group(self, age: int) -> int:
        """
        Get age group category.
        
        Args:
            age: Patient age
            
        Returns:
            int: Age group (0=infant, 1=child, 2=adult, 3=elderly)
        """
        if age < 2:
            return 0  # Infant
        elif age < 18:
            return 1  # Child
        elif age < 65:
            return 2  # Adult
        else:
            return 3  # Elderly
    
    def _encode_chronic_conditions(self, conditions: List[str]) -> np.ndarray:
        """
        Encode chronic conditions as binary flags.
        
        Args:
            conditions: List of chronic conditions
            
        Returns:
            np.ndarray: Binary flags for common conditions
        """
        # Common chronic conditions
        condition_flags = {
            "diabetes": 0,
            "hypertension": 1,
            "heart disease": 2,
            "asthma": 3,
            "copd": 4
        }
        
        vector = np.zeros(5)
        
        for condition in conditions:
            condition_lower = condition.lower()
            for key, idx in condition_flags.items():
                if key in condition_lower:
                    vector[idx] = 1.0
        
        return vector
    
    def _calculate_symptom_severity(self, symptoms: List[ExtractedSymptom]) -> float:
        """
        Calculate overall symptom severity score.
        
        Args:
            symptoms: List of extracted symptoms
            
        Returns:
            float: Severity score [0, 1]
        """
        # Critical symptoms get higher weights
        critical_symptoms = {
            "chest pain": 0.9,
            "shortness of breath": 0.8,
            "loss of consciousness": 1.0,
            "severe pain": 0.8,
            "bleeding": 0.7,
            "seizure": 0.9,
        }
        
        if not symptoms:
            return 0.0
        
        severity_scores = []
        for symptom in symptoms:
            base_score = critical_symptoms.get(symptom.normalized, 0.3)
            severity_scores.append(base_score * symptom.confidence)
        
        # Average severity
        return min(np.mean(severity_scores), 1.0)
    
    def _encode_symptom_categories(self, symptoms: List[ExtractedSymptom]) -> np.ndarray:
        """
        Encode symptom by medical categories.
        
        Args:
            symptoms: List of extracted symptoms
            
        Returns:
            np.ndarray: Category presence flags
        """
        categories = {
            "cardiovascular": ["chest pain", "heart palpitations", "shortness of breath"],
            "neurological": ["headache", "dizziness", "confusion", "seizure"],
            "gastrointestinal": ["nausea", "vomiting", "abdominal pain"],
            "respiratory": ["cough", "wheezing", "difficulty breathing"],
            "general": ["fever", "fatigue", "sweating"]
        }
        
        vector = np.zeros(len(categories))
        
        for idx, (category, category_symptoms) in enumerate(categories.items()):
            for symptom in symptoms:
                if any(cs in symptom.normalized for cs in category_symptoms):
                    vector[idx] = 1.0
                    break
        
        return vector
