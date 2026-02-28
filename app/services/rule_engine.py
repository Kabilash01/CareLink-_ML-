"""
CARELINK AI Microservice - Clinical Safety Rule Engine

Production-grade rule engine for clinical safety overrides.
Implements deterministic rules that override ML predictions when critical conditions are detected.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from app.nlp.symptom_extractor import ExtractedSymptom
from app.core.logging import logger
from app.core.settings import get_settings
from app.config import (
    CRITICAL_SYMPTOMS,
    HIGH_RISK_COMBINATIONS,
    MINIMUM_RISK_OVERRIDES
)

settings = get_settings()


@dataclass
class RuleResult:
    """Result of a triggered safety rule."""
    rule_name: str
    risk_override: Optional[str]
    confidence_boost: float
    reason: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class RuleEngine:
    """
    Clinical safety rule engine for risk assessment overrides.
    
    Rules are:
    - Deterministic and transparent
    - Versioned and auditable
    - Configurable via settings
    - Prioritized by severity
    """
    
    def __init__(self):
        """Initialize rule engine with clinical safety rules."""
        self.version = "1.0.0"
        self.enabled = settings.SAFETY_RULES_ENABLED
        
        # Rule registry
        self.rules = self._register_rules()
        
        logger.info(f"RuleEngine initialized with {len(self.rules)} rules (version {self.version})")
    
    def _register_rules(self) -> dict:
        """
        Register all clinical safety rules.
        
        Returns:
            dict: Rule name to function mapping
        """
        return {
            "CRITICAL_SYMPTOM_CHECK": self._check_critical_symptoms,
            "HIGH_RISK_COMBINATION": self._check_high_risk_combinations,
            "CHEST_PAIN_CARDIOVASCULAR": self._check_chest_pain_cardiovascular,
            "PREGNANCY_COMPLICATIONS": self._check_pregnancy_complications,
            "INFANT_FEVER": self._check_infant_fever,
            "ELDERLY_FALL": self._check_elderly_fall,
            "SEVERE_ALLERGIC_REACTION": self._check_severe_allergic_reaction,
            "NEUROLOGICAL_EMERGENCY": self._check_neurological_emergency,
        }
    
    def evaluate_rules(
        self,
        symptoms: List[ExtractedSymptom],
        age: int,
        duration_days: int,
        chronic_conditions: List[str],
        predicted_risk: str,
        confidence: float
    ) -> Tuple[str, float, List[RuleResult]]:
        """
        Evaluate all safety rules and determine final risk level.
        
        Args:
            symptoms: Extracted symptoms
            age: Patient age
            duration_days: Symptom duration
            chronic_conditions: List of chronic conditions
            predicted_risk: ML model prediction
            confidence: ML model confidence
            
        Returns:
            Tuple[str, float, List[RuleResult]]: (final_risk, final_confidence, triggered_rules)
        """
        if not self.enabled:
            logger.debug("Rule engine disabled")
            return predicted_risk, confidence, []
        
        triggered_rules = []
        final_risk = predicted_risk
        final_confidence = confidence
        
        # Context for rule evaluation
        context = {
            "symptoms": symptoms,
            "age": age,
            "duration_days": duration_days,
            "chronic_conditions": chronic_conditions,
            "predicted_risk": predicted_risk,
            "confidence": confidence
        }
        
        # Evaluate each rule
        for rule_name, rule_func in self.rules.items():
            try:
                result = rule_func(context)
                
                if result:
                    triggered_rules.append(result)
                    logger.info(
                        f"Rule triggered: {rule_name}",
                        extra={
                            "rule": rule_name,
                            "override": result.risk_override,
                            "reason": result.reason
                        }
                    )
                    
                    # Apply risk override if specified
                    if result.risk_override:
                        # Only escalate, never de-escalate
                        risk_priority = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
                        if risk_priority.get(result.risk_override, 0) > risk_priority.get(final_risk, 0):
                            final_risk = result.risk_override
                            final_confidence = min(final_confidence + result.confidence_boost, 1.0)
            
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {str(e)}")
        
        return final_risk, final_confidence, triggered_rules
    
    def _check_critical_symptoms(self, context: dict) -> Optional[RuleResult]:
        """Check for presence of critical symptoms."""
        symptoms = context["symptoms"]
        
        critical_found = []
        for symptom in symptoms:
            if any(critical in symptom.normalized for critical in CRITICAL_SYMPTOMS):
                critical_found.append(symptom.normalized)
        
        if critical_found:
            return RuleResult(
                rule_name="CRITICAL_SYMPTOM_CHECK",
                risk_override="HIGH",
                confidence_boost=0.2,
                reason=f"Critical symptoms detected: {', '.join(critical_found)}"
            )
        
        return None
    
    def _check_high_risk_combinations(self, context: dict) -> Optional[RuleResult]:
        """Check for high-risk symptom combinations."""
        symptoms = context["symptoms"]
        symptom_set = {s.normalized for s in symptoms}
        
        for combination in HIGH_RISK_COMBINATIONS:
            if all(symptom in ' '.join(symptom_set) for symptom in combination):
                return RuleResult(
                    rule_name="HIGH_RISK_COMBINATION",
                    risk_override="HIGH",
                    confidence_boost=0.25,
                    reason=f"High-risk combination detected: {' + '.join(combination)}"
                )
        
        return None
    
    def _check_chest_pain_cardiovascular(self, context: dict) -> Optional[RuleResult]:
        """Check for chest pain with cardiovascular risk factors."""
        symptoms = context["symptoms"]
        chronic_conditions = context["chronic_conditions"]
        age = context["age"]
        
        has_chest_pain = any("chest pain" in s.normalized for s in symptoms)
        
        if not has_chest_pain:
            return None
        
        # Check cardiovascular risk factors
        risk_factors = []
        
        if age > 45:
            risk_factors.append("age > 45")
        
        cardiovascular_conditions = ["diabetes", "hypertension", "heart disease", "high cholesterol"]
        for condition in chronic_conditions:
            if any(cv in condition.lower() for cv in cardiovascular_conditions):
                risk_factors.append(condition)
        
        # Check for associated symptoms
        associated_symptoms = ["sweating", "nausea", "shortness of breath", "dizziness"]
        for symptom in symptoms:
            if any(assoc in symptom.normalized for assoc in associated_symptoms):
                risk_factors.append(symptom.normalized)
        
        if risk_factors:
            return RuleResult(
                rule_name="CHEST_PAIN_CARDIOVASCULAR",
                risk_override="HIGH",
                confidence_boost=0.3,
                reason=f"Chest pain with cardiovascular risk factors: {', '.join(risk_factors)}"
            )
        
        return None
    
    def _check_pregnancy_complications(self, context: dict) -> Optional[RuleResult]:
        """Check for pregnancy-related complications."""
        symptoms = context["symptoms"]
        chronic_conditions = context["chronic_conditions"]
        
        is_pregnancy_related = any(
            "pregnancy" in condition.lower() or "pregnant" in condition.lower()
            for condition in chronic_conditions
        )
        
        if not is_pregnancy_related:
            return None
        
        # Check for pregnancy warning signs
        warning_signs = ["bleeding", "severe pain", "vision changes", "severe headache", "swelling"]
        found_warnings = []
        
        for symptom in symptoms:
            if any(warning in symptom.normalized for warning in warning_signs):
                found_warnings.append(symptom.normalized)
        
        if found_warnings:
            return RuleResult(
                rule_name="PREGNANCY_COMPLICATIONS",
                risk_override="HIGH",
                confidence_boost=0.25,
                reason=f"Pregnancy with warning signs: {', '.join(found_warnings)}"
            )
        
        return None
    
    def _check_infant_fever(self, context: dict) -> Optional[RuleResult]:
        """Check for fever in infants (high risk)."""
        age = context["age"]
        symptoms = context["symptoms"]
        
        if age >= 2:  # Not an infant
            return None
        
        has_fever = any("fever" in s.normalized for s in symptoms)
        
        if has_fever:
            return RuleResult(
                rule_name="INFANT_FEVER",
                risk_override="MEDIUM",  # Minimum medium, could be escalated to HIGH
                confidence_boost=0.2,
                reason="Fever in infant (< 2 years old) requires medical evaluation"
            )
        
        return None
    
    def _check_elderly_fall(self, context: dict) -> Optional[RuleResult]:
        """Check for fall in elderly patient."""
        age = context["age"]
        symptoms = context["symptoms"]
        
        if age < 65:  # Not elderly
            return None
        
        fall_related = ["fall", "fell", "injury", "head injury", "confusion", "dizziness"]
        has_fall = any(
            any(keyword in s.text.lower() for keyword in fall_related)
            for s in symptoms
        )
        
        if has_fall:
            return RuleResult(
                rule_name="ELDERLY_FALL",
                risk_override="MEDIUM",
                confidence_boost=0.15,
                reason="Fall in elderly patient (≥ 65 years) requires assessment"
            )
        
        return None
    
    def _check_severe_allergic_reaction(self, context: dict) -> Optional[RuleResult]:
        """Check for signs of severe allergic reaction (anaphylaxis)."""
        symptoms = context["symptoms"]
        
        allergic_signs = {
            "difficulty breathing": 0.9,
            "shortness of breath": 0.9,
            "swelling": 0.7,
            "rash": 0.5,
            "itching": 0.5,
            "dizziness": 0.6
        }
        
        matched_signs = []
        severity_score = 0
        
        for symptom in symptoms:
            for sign, weight in allergic_signs.items():
                if sign in symptom.normalized:
                    matched_signs.append(sign)
                    severity_score += weight
        
        # If multiple allergic signs with high severity
        if len(matched_signs) >= 2 and severity_score >= 1.0:
            return RuleResult(
                rule_name="SEVERE_ALLERGIC_REACTION",
                risk_override="HIGH",
                confidence_boost=0.2,
                reason=f"Possible severe allergic reaction: {', '.join(matched_signs)}"
            )
        
        return None
    
    def _check_neurological_emergency(self, context: dict) -> Optional[RuleResult]:
        """Check for neurological emergency signs."""
        symptoms = context["symptoms"]
        
        neuro_emergency_signs = [
            "loss of consciousness",
            "seizure",
            "paralysis",
            "severe headache",
            "confusion",
            "vision changes",
            "slurred speech"
        ]
        
        found_signs = []
        for symptom in symptoms:
            if any(sign in symptom.normalized for sign in neuro_emergency_signs):
                found_signs.append(symptom.normalized)
        
        if found_signs:
            return RuleResult(
                rule_name="NEUROLOGICAL_EMERGENCY",
                risk_override="HIGH",
                confidence_boost=0.25,
                reason=f"Neurological emergency signs: {', '.join(found_signs)}"
            )
        
        return None
    
    def get_all_rules(self) -> List[Dict[str, str]]:
        """
        Get list of all registered rules with descriptions.
        
        Returns:
            List[Dict]: Rule metadata
        """
        rule_descriptions = {
            "CRITICAL_SYMPTOM_CHECK": "Checks for presence of critical symptoms",
            "HIGH_RISK_COMBINATION": "Detects high-risk symptom combinations",
            "CHEST_PAIN_CARDIOVASCULAR": "Evaluates chest pain with cardiovascular risk factors",
            "PREGNANCY_COMPLICATIONS": "Identifies pregnancy-related warning signs",
            "INFANT_FEVER": "Flags fever in infants under 2 years",
            "ELDERLY_FALL": "Detects falls in elderly patients (≥ 65 years)",
            "SEVERE_ALLERGIC_REACTION": "Identifies signs of anaphylaxis",
            "NEUROLOGICAL_EMERGENCY": "Detects neurological emergency signs"
        }
        
        return [
            {
                "name": rule_name,
                "description": rule_descriptions.get(rule_name, "No description available"),
                "enabled": self.enabled
            }
            for rule_name in self.rules.keys()
        ]
