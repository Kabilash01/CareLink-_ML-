"""
CARELINK AI Microservice - Triage Service

Main orchestration service for symptom triage pipeline.
Coordinates all components: NLP, feature engineering, ML model, rules, and explanations.
"""

from typing import List
from datetime import datetime
import time

from app.api.schemas import TriageResponse, ProbabilityDistribution, RiskLevel
from app.nlp.text_preprocessor import TextPreprocessor
from app.nlp.symptom_extractor import SymptomExtractor
from app.nlp.language_handler import LanguageHandler
from app.services.feature_engineering import FeatureEngineer
from app.services.model_service import ModelService
from app.services.rule_engine import RuleEngine
from app.services.confidence_controller import ConfidenceController
from app.services.explanation_service import ExplanationService
from app.core.logging import logger, log_prediction
from app.core.security import sanitize_input, validate_age, validate_duration
from app.db.base import SessionLocal
from app.db.models import TriagePrediction


class TriageService:
    """
    Main triage service orchestrating the complete prediction pipeline.
    
    Pipeline steps:
    1. Input validation and sanitization
    2. Text preprocessing
    3. Symptom extraction (NLP)
    4. Feature engineering
    5. ML model prediction
    6. Clinical safety rule evaluation
    7. Confidence-based adjustment
    8. Explanation generation
    9. Response assembly
    """
    
    def __init__(self):
        """Initialize triage service with all components."""
        # Initialize pipeline components
        self.text_preprocessor = TextPreprocessor()
        self.symptom_extractor = SymptomExtractor()
        self.language_handler = LanguageHandler()
        self.feature_engineer = FeatureEngineer()
        self.model_service = ModelService()
        self.rule_engine = RuleEngine()
        self.confidence_controller = ConfidenceController()
        self.explanation_service = ExplanationService()
        
        logger.info("TriageService initialized")
    
    async def predict(
        self,
        symptoms_text: str,
        age: int,
        duration_days: int,
        chronic_conditions: List[str],
        language: str,
        request_id: str
    ) -> TriageResponse:
        """
        Perform complete triage prediction pipeline.
        
        Args:
            symptoms_text: Patient symptom description
            age: Patient age
            duration_days: Symptom duration in days
            chronic_conditions: List of chronic conditions
            language: Language code
            request_id: Unique request ID
            
        Returns:
            TriageResponse: Complete triage assessment
        """
        logger.info(
            f"Starting triage prediction",
            extra={"request_id": request_id}
        )
        start_time = time.monotonic()
        
        # Step 1: Validate and sanitize input
        symptoms_text = sanitize_input(symptoms_text)
        validate_age(age)
        validate_duration(duration_days)
        
        # Step 2: Preprocess text
        preprocessed_text = self.text_preprocessor.preprocess(symptoms_text)
        logger.debug(f"Preprocessed text: {preprocessed_text}")
        
        # Step 3: Extract symptoms
        extracted_symptoms = self.symptom_extractor.extract_symptoms(symptoms_text)
        logger.info(
            f"Extracted {len(extracted_symptoms)} symptoms",
            extra={"request_id": request_id, "symptoms": [s.normalized for s in extracted_symptoms]}
        )
        
        # Step 4: Feature engineering
        feature_vector = self.feature_engineer.extract_features(
            symptoms=extracted_symptoms,
            age=age,
            duration_days=duration_days,
            chronic_conditions=chronic_conditions
        )
        logger.debug(f"Feature vector shape: {feature_vector.features.shape}")
        
        # Step 5: ML model prediction (always run as baseline)
        ml_risk, probabilities = self.model_service.predict(feature_vector.features)
        ml_confidence = self.confidence_controller.calculate_confidence(probabilities)

        logger.info(
            f"ML model prediction: {ml_risk} (confidence: {ml_confidence:.2f})",
            extra={"request_id": request_id}
        )

        # Step 5b: MedGemma triage (primary predictor when available)
        medgemma_result = self.explanation_service.predict_triage(
            symptoms_text=symptoms_text,
            age=age,
            duration_days=duration_days,
            chronic_conditions=chronic_conditions,
            language=language,
        )

        if medgemma_result is not None:
            # MedGemma available — use it as primary, blend confidence with ML
            predicted_risk = medgemma_result["risk_level"]
            mg_confidence  = medgemma_result["confidence"]

            # If ML and MedGemma agree, boost confidence; if they disagree, be conservative
            if predicted_risk == ml_risk:
                confidence = min(1.0, (mg_confidence + ml_confidence) / 2 + 0.05)
            else:
                # MedGemma wins clinically — but reduce confidence to reflect disagreement
                confidence = mg_confidence * 0.85
                logger.info(
                    f"ML/MedGemma disagree: ML={ml_risk} MedGemma={predicted_risk} "
                    f"— using MedGemma | request_id={request_id}"
                )

            # Sync probabilities to MedGemma's decision (keep ML probs as base, adjust peak)
            risk_idx = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}[predicted_risk]
            other_total = 1.0 - mg_confidence
            probabilities = {k: other_total / 2 for k in ("low", "medium", "high")}
            probabilities[risk_idx] = mg_confidence

            logger.info(
                f"MedGemma triage: {predicted_risk} ({confidence:.2f}) | "
                f"reasoning: {medgemma_result['reasoning'][:80]}",
                extra={"request_id": request_id}
            )
        else:
            # MedGemma not ready — fall back to ML model
            predicted_risk = ml_risk
            confidence     = ml_confidence
            logger.info(
                "MedGemma not available — using ML model prediction",
                extra={"request_id": request_id}
            )
        
        # Step 6: Apply clinical safety rules
        final_risk, final_confidence, triggered_rules = self.rule_engine.evaluate_rules(
            symptoms=extracted_symptoms,
            age=age,
            duration_days=duration_days,
            chronic_conditions=chronic_conditions,
            predicted_risk=predicted_risk,
            confidence=confidence
        )
        
        rule_names = [rule.rule_name for rule in triggered_rules]
        logger.info(
            f"Rules evaluation: {predicted_risk} → {final_risk}, "
            f"triggered={len(rule_names)} rules",
            extra={"request_id": request_id, "rules": rule_names}
        )
        
        # Step 7: Confidence-based adjustment
        confidence_decision = self.confidence_controller.evaluate(
            predicted_risk=final_risk,
            confidence=final_confidence,
            probabilities=probabilities
        )
        
        final_risk = confidence_decision.adjusted_risk
        final_confidence = confidence_decision.adjusted_confidence
        escalated = confidence_decision.escalated
        emergency_flag = confidence_decision.emergency_flag
        
        logger.info(
            f"Confidence adjustment: escalated={escalated}, emergency={emergency_flag}",
            extra={"request_id": request_id}
        )
        
        # Step 8: Generate explanation
        symptom_names = [s.normalized for s in extracted_symptoms]
        explanation = self.explanation_service.generate_explanation(
            risk_level=final_risk,
            symptoms=symptom_names,
            confidence=final_confidence,
            rules_triggered=rule_names,
            emergency_flag=emergency_flag,
            language=language
        )
        
        # Step 9: Assemble response
        response = TriageResponse(
            prediction=RiskLevel(final_risk),
            probabilities=ProbabilityDistribution(**probabilities),
            confidence=final_confidence,
            rules_triggered=rule_names,
            explanation=explanation,
            model_version=self.model_service.model_version,
            escalated=escalated,
            emergency_flag=emergency_flag,
            request_id=request_id,
            timestamp=datetime.utcnow()
        )
        
        # Log prediction for audit trail
        log_prediction(
            request_id=request_id,
            prediction=final_risk,
            confidence=final_confidence,
            model_version=self.model_service.model_version,
            rules_triggered=rule_names,
            num_symptoms=len(extracted_symptoms),
            age=age,
            escalated=escalated,
            emergency_flag=emergency_flag
        )

        # ---- Persist to SQLite ----------------------------------------
        processing_ms = (time.monotonic() - start_time) * 1000
        self._save_prediction_to_db(
            request_id=request_id,
            symptoms_text=symptoms_text,
            age=age,
            duration_days=duration_days,
            chronic_conditions=chronic_conditions,
            language=language,
            prediction=final_risk,
            confidence=final_confidence,
            probabilities=probabilities,
            rules_triggered=rule_names,
            escalated=escalated,
            emergency_flag=emergency_flag,
            explanation=explanation,
            model_version=self.model_service.model_version,
            processing_ms=processing_ms,
        )
        
        logger.info(
            f"Triage prediction completed: {final_risk}",
            extra={"request_id": request_id}
        )
        
        return response

    # ------------------------------------------------------------------
    # Private DB helper
    # ------------------------------------------------------------------

    def _save_prediction_to_db(
        self,
        request_id: str,
        symptoms_text: str,
        age: int,
        duration_days: int,
        chronic_conditions: List[str],
        language: str,
        prediction: str,
        confidence: float,
        probabilities: dict,
        rules_triggered: List[str],
        escalated: bool,
        emergency_flag: bool,
        explanation: str,
        model_version: str,
        processing_ms: float,
    ):
        """Persist each triage inference to SQLite for audit and analytics."""
        db = SessionLocal()
        try:
            record = TriagePrediction(
                request_id=request_id,
                symptoms_text=symptoms_text,
                patient_age=age,
                duration_days=duration_days,
                chronic_conds=chronic_conditions,
                language=language,
                prediction=prediction,
                confidence=confidence,
                prob_low=probabilities.get("low"),
                prob_medium=probabilities.get("medium"),
                prob_high=probabilities.get("high"),
                rules_triggered=rules_triggered,
                escalated=escalated,
                emergency_flag=emergency_flag,
                explanation=explanation,
                model_version=model_version,
                processing_ms=processing_ms,
            )
            db.add(record)
            db.commit()
            logger.debug(f"Prediction saved to SQLite | request_id={request_id}")
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to persist prediction to SQLite: {exc}")
        finally:
            db.close()
