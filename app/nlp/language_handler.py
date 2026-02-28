"""
CARELINK AI Microservice - Language Handler

Multilingual support for symptom processing and explanation generation.
Handles language detection and translation for supported languages.
"""

from typing import Optional
from app.core.logging import logger
from app.core.settings import get_settings

settings = get_settings()


class LanguageHandler:
    """
    Handle multilingual symptom processing and explanations.
    
    Supports:
    - English (en)
    - Spanish (es)
    - French (fr)
    """
    
    def __init__(self):
        """Initialize language handler with translation mappings."""
        
        # Basic medical term translations (extend with proper translation service)
        self.translations = {
            "es": {  # Spanish
                "chest pain": "dolor de pecho",
                "headache": "dolor de cabeza",
                "fever": "fiebre",
                "nausea": "náusea",
                "dizziness": "mareo",
                "shortness of breath": "falta de aire",
                "cough": "tos",
                "fatigue": "fatiga",
            },
            "fr": {  # French
                "chest pain": "douleur thoracique",
                "headache": "mal de tête",
                "fever": "fièvre",
                "nausea": "nausée",
                "dizziness": "vertiges",
                "shortness of breath": "essoufflement",
                "cough": "toux",
                "fatigue": "fatigue",
            }
        }
        
        # Risk level translations
        self.risk_translations = {
            "es": {
                "LOW": "BAJO",
                "MEDIUM": "MEDIO",
                "HIGH": "ALTO"
            },
            "fr": {
                "LOW": "BAS",
                "MEDIUM": "MOYEN",
                "HIGH": "ÉLEVÉ"
            }
        }
        
        # Explanation templates by language
        self.explanation_templates = {
            "en": {
                "high": "Based on your symptoms of {symptoms}, immediate medical attention is recommended. Please seek emergency care.",
                "medium": "Based on your symptoms of {symptoms}, you should consult a healthcare provider soon. Monitor your condition closely.",
                "low": "Based on your symptoms of {symptoms}, your condition appears manageable. Consider consulting a healthcare provider if symptoms worsen."
            },
            "es": {
                "high": "Según sus síntomas de {symptoms}, se recomienda atención médica inmediata. Busque atención de emergencia.",
                "medium": "Según sus síntomas de {symptoms}, debe consultar a un proveedor de atención médica pronto. Monitoree su condición de cerca.",
                "low": "Según sus síntomas de {symptoms}, su condición parece manejable. Considere consultar a un proveedor de atención médica si los síntomas empeoran."
            },
            "fr": {
                "high": "En fonction de vos symptômes de {symptoms}, une attention médicale immédiate est recommandée. Veuillez consulter un médecin d'urgence.",
                "medium": "En fonction de vos symptômes de {symptoms}, vous devriez consulter un professionnel de santé bientôt. Surveillez votre état de près.",
                "low": "En fonction de vos symptômes de {symptoms}, votre état semble gérable. Envisagez de consulter un professionnel de santé si les symptômes s'aggravent."
            }
        }
        
        logger.info("LanguageHandler initialized")
    
    def is_supported(self, language: str) -> bool:
        """
        Check if language is supported.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            bool: True if language is supported
        """
        return language.lower() in settings.SUPPORTED_LANGUAGES
    
    def translate_symptom(self, symptom: str, target_language: str) -> str:
        """
        Translate symptom to target language.
        
        Args:
            symptom: Symptom in English
            target_language: Target language code
            
        Returns:
            str: Translated symptom or original if translation not available
        """
        if target_language == "en" or target_language not in self.translations:
            return symptom
        
        return self.translations[target_language].get(symptom, symptom)
    
    def translate_risk_level(self, risk_level: str, target_language: str) -> str:
        """
        Translate risk level to target language.
        
        Args:
            risk_level: Risk level in English (LOW/MEDIUM/HIGH)
            target_language: Target language code
            
        Returns:
            str: Translated risk level
        """
        if target_language == "en" or target_language not in self.risk_translations:
            return risk_level
        
        return self.risk_translations[target_language].get(risk_level, risk_level)
    
    def get_explanation_template(self, risk_level: str, language: str) -> str:
        """
        Get explanation template for given risk level and language.
        
        Args:
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            language: Language code
            
        Returns:
            str: Explanation template
        """
        risk_key = risk_level.lower()
        
        if language not in self.explanation_templates:
            language = "en"  # Fallback to English
        
        return self.explanation_templates[language].get(
            risk_key,
            self.explanation_templates["en"][risk_key]
        )
    
    def format_explanation(
        self,
        risk_level: str,
        symptoms: list,
        language: str
    ) -> str:
        """
        Format explanation in target language.
        
        Args:
            risk_level: Risk level
            symptoms: List of symptom strings
            language: Target language
            
        Returns:
            str: Formatted explanation
        """
        template = self.get_explanation_template(risk_level, language)
        
        # Translate symptoms if needed
        if language != "en":
            symptoms = [
                self.translate_symptom(s, language) for s in symptoms
            ]
        
        # Format symptoms list
        if len(symptoms) == 0:
            symptoms_text = "unknown symptoms"
        elif len(symptoms) == 1:
            symptoms_text = symptoms[0]
        elif len(symptoms) == 2:
            symptoms_text = f"{symptoms[0]} and {symptoms[1]}"
        else:
            symptoms_text = ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"
        
        # Fill template
        explanation = template.format(symptoms=symptoms_text)
        
        return explanation
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from text.
        
        This is a simple heuristic-based detector. In production,
        use a proper language detection library like langdetect or fasttext.
        
        Args:
            text: Input text
            
        Returns:
            str: Detected language code
        """
        # Simple keyword-based detection (placeholder)
        text_lower = text.lower()
        
        spanish_indicators = ["dolor", "tengo", "siento", "estoy", "síntomas"]
        french_indicators = ["douleur", "j'ai", "je", "suis", "symptômes"]
        
        if any(word in text_lower for word in spanish_indicators):
            return "es"
        elif any(word in text_lower for word in french_indicators):
            return "fr"
        
        # Default to English
        return "en"
