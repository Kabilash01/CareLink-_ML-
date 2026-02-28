"""
CARELINK AI Microservice - Confidence Controller

Confidence-based escalation and quality control for ML predictions.
Implements logic for adjusting risk levels based on model confidence.
"""

from typing import Tuple
from dataclasses import dataclass

from app.core.logging import logger
from app.core.settings import get_settings

settings = get_settings()


@dataclass
class ConfidenceDecision:
    """Decision from confidence controller."""
    escalated: bool
    emergency_flag: bool
    adjusted_risk: str
    adjusted_confidence: float
    reason: str


class ConfidenceController:
    """
    Confidence-based quality control for predictions.
    
    Implements:
    - Confidence thresholding
    - Automatic escalation for low confidence
    - Emergency flagging for high-risk predictions
    - Conservative fallback strategies
    """
    
    def __init__(self):
        """Initialize confidence controller with thresholds."""
        self.enabled = settings.CONFIDENCE_CONTROLLER_ENABLED
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self.escalation_threshold = settings.ESCALATION_THRESHOLD
        self.emergency_threshold = settings.EMERGENCY_FLAG_THRESHOLD
        
        logger.info(
            f"ConfidenceController initialized "
            f"(threshold={self.confidence_threshold}, "
            f"escalation={self.escalation_threshold})"
        )
    
    def evaluate(
        self,
        predicted_risk: str,
        confidence: float,
        probabilities: dict
    ) -> ConfidenceDecision:
        """
        Evaluate prediction confidence and adjust if needed.
        
        Args:
            predicted_risk: Predicted risk level (LOW/MEDIUM/HIGH)
            confidence: Model confidence score
            probabilities: Full probability distribution
            
        Returns:
            ConfidenceDecision: Confidence evaluation result
        """
        if not self.enabled:
            return ConfidenceDecision(
                escalated=False,
                emergency_flag=False,
                adjusted_risk=predicted_risk,
                adjusted_confidence=confidence,
                reason="Confidence controller disabled"
            )
        
        escalated = False
        emergency_flag = False
        adjusted_risk = predicted_risk
        adjusted_confidence = confidence
        reasons = []
        
        # Rule 1: Low confidence escalation
        if confidence < self.confidence_threshold:
            # Escalate one level due to uncertainty
            if predicted_risk == "LOW":
                adjusted_risk = "MEDIUM"
                escalated = True
                reasons.append(
                    f"Escalated LOW → MEDIUM due to low confidence ({confidence:.2f} < {self.confidence_threshold})"
                )
            elif predicted_risk == "MEDIUM":
                adjusted_risk = "HIGH"
                escalated = True
                reasons.append(
                    f"Escalated MEDIUM → HIGH due to low confidence ({confidence:.2f} < {self.confidence_threshold})"
                )
            else:
                reasons.append(
                    f"Already HIGH risk with low confidence ({confidence:.2f})"
                )
        
        # Rule 2: Emergency flag for high-risk predictions
        high_prob = probabilities.get("high", 0.0)
        if high_prob > self.emergency_threshold:
            emergency_flag = True
            reasons.append(
                f"Emergency flag set (HIGH probability {high_prob:.2f} > {self.emergency_threshold})"
            )
        
        # Rule 3: Uncertain predictions near decision boundary
        # If probabilities are close, escalate to be conservative
        prob_values = list(probabilities.values())
        if len(prob_values) >= 2:
            sorted_probs = sorted(prob_values, reverse=True)
            if sorted_probs[0] - sorted_probs[1] < 0.1:  # Within 10% margin
                if predicted_risk != "HIGH":
                    # Close call - be conservative
                    if predicted_risk == "LOW":
                        adjusted_risk = "MEDIUM"
                    else:
                        adjusted_risk = "HIGH"
                    escalated = True
                    reasons.append(
                        "Escalated due to uncertain prediction (close probabilities)"
                    )
        
        # Rule 4: Very low confidence - default to MEDIUM
        if confidence < 0.3:
            adjusted_risk = "MEDIUM"
            escalated = True
            reasons.append(
                f"Defaulted to MEDIUM due to very low confidence ({confidence:.2f} < 0.3)"
            )
        
        # Rule 5: Emergency flag for any HIGH risk (covers rule-escalated HIGH,
        # where ML fallback probabilities may be low but clinical rules override)
        if adjusted_risk == "HIGH" and not emergency_flag:
            emergency_flag = True
            reasons.append(
                "Emergency flag set: final risk level is HIGH (rule-escalated or ML-predicted)"
            )
        
        # Boost confidence if we escalated for safety
        if escalated:
            adjusted_confidence = min(confidence + 0.1, 0.95)
        
        reason = "; ".join(reasons) if reasons else "No adjustments needed"
        
        logger.info(
            f"Confidence evaluation: {predicted_risk} → {adjusted_risk} "
            f"(confidence={confidence:.2f}, escalated={escalated}, emergency={emergency_flag})"
        )
        
        return ConfidenceDecision(
            escalated=escalated,
            emergency_flag=emergency_flag,
            adjusted_risk=adjusted_risk,
            adjusted_confidence=adjusted_confidence,
            reason=reason
        )
    
    def calculate_confidence(self, probabilities: dict) -> float:
        """
        Calculate confidence score from probability distribution.
        
        Confidence is based on:
        - Maximum probability (primary)
        - Entropy of distribution (secondary)
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            float: Confidence score [0, 1]
        """
        import numpy as np
        
        # Primary: max probability
        max_prob = max(probabilities.values())
        
        # Secondary: entropy (lower entropy = higher confidence)
        probs = np.array(list(probabilities.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probabilities))
        normalized_entropy = entropy / max_entropy
        
        # Combine: 80% max prob, 20% inverted entropy
        confidence = 0.8 * max_prob + 0.2 * (1 - normalized_entropy)
        
        return confidence
    
    def requires_human_review(self, confidence: float, risk: str) -> bool:
        """
        Determine if prediction requires human review.
        
        Args:
            confidence: Prediction confidence
            risk: Risk level
            
        Returns:
            bool: True if human review recommended
        """
        # Low confidence HIGH risk always needs review
        if risk == "HIGH" and confidence < 0.7:
            return True
        
        # Very low confidence always needs review
        if confidence < 0.4:
            return True
        
        return False
