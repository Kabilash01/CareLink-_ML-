"""
CARELINK AI Microservice - Rule Engine Tests

Tests for clinical safety rule engine.
"""

import pytest
from app.services.rule_engine import RuleEngine
from app.nlp.symptom_extractor import ExtractedSymptom


class TestRuleEngine:
    """Test rule engine functionality."""
    
    @pytest.fixture
    def rule_engine(self):
        """Create rule engine instance."""
        return RuleEngine()
    
    @pytest.fixture
    def sample_symptoms(self):
        """Create sample symptoms."""
        return [
            ExtractedSymptom(
                text="chest pain",
                normalized="chest pain",
                confidence=0.9,
                span=(0, 10)
            ),
            ExtractedSymptom(
                text="sweating",
                normalized="sweating",
                confidence=0.8,
                span=(15, 23)
            )
        ]
    
    def test_rule_engine_initialization(self, rule_engine):
        """Test rule engine initialization."""
        assert rule_engine is not None
        assert len(rule_engine.rules) > 0
        assert rule_engine.version is not None
    
    def test_critical_symptom_detection(self, rule_engine, sample_symptoms):
        """Test critical symptom detection."""
        context = {
            "symptoms": sample_symptoms,
            "age": 50,
            "duration_days": 1,
            "chronic_conditions": [],
            "predicted_risk": "MEDIUM",
            "confidence": 0.7
        }
        
        final_risk, final_confidence, triggered = rule_engine.evaluate_rules(
            symptoms=context["symptoms"],
            age=context["age"],
            duration_days=context["duration_days"],
            chronic_conditions=context["chronic_conditions"],
            predicted_risk=context["predicted_risk"],
            confidence=context["confidence"]
        )
        
        # Chest pain should trigger rules
        assert len(triggered) > 0
        assert final_risk == "HIGH"  # Should be escalated
    
    def test_infant_fever_rule(self, rule_engine):
        """Test infant fever rule."""
        fever_symptom = [
            ExtractedSymptom(
                text="fever",
                normalized="fever",
                confidence=0.9,
                span=(0, 5)
            )
        ]
        
        context = {
            "symptoms": fever_symptom,
            "age": 1,  # Infant
            "duration_days": 1,
            "chronic_conditions": [],
            "predicted_risk": "LOW",
            "confidence": 0.8
        }
        
        final_risk, final_confidence, triggered = rule_engine.evaluate_rules(
            symptoms=context["symptoms"],
            age=context["age"],
            duration_days=context["duration_days"],
            chronic_conditions=context["chronic_conditions"],
            predicted_risk=context["predicted_risk"],
            confidence=context["confidence"]
        )
        
        # Should escalate to at least MEDIUM
        assert final_risk in ["MEDIUM", "HIGH"]
        assert any("INFANT_FEVER" in rule.rule_name for rule in triggered)
    
    def test_get_all_rules(self, rule_engine):
        """Test get all rules."""
        rules = rule_engine.get_all_rules()
        
        assert isinstance(rules, list)
        assert len(rules) > 0
        
        for rule in rules:
            assert "name" in rule
            assert "description" in rule
            assert "enabled" in rule
