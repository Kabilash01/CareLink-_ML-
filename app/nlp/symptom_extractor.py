"""
CARELINK AI Microservice - Symptom Extractor

Medical entity extraction for symptom identification using spaCy.
Extracts and normalizes symptom mentions from natural language text.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc

from app.core.logging import logger
from app.core.settings import get_settings

settings = get_settings()


@dataclass
class ExtractedSymptom:
    """Extracted symptom entity with confidence."""
    text: str
    normalized: str
    confidence: float
    span: Tuple[int, int]  # Character span in original text


class SymptomExtractor:
    """
    Extract medical symptoms from natural language text.
    
    Uses spaCy for NER and custom medical vocabulary for symptom recognition.
    """
    
    def __init__(self):
        """Initialize symptom extractor with spaCy model and medical vocabulary."""
        self.nlp = None
        self.medical_symptoms = self._load_medical_symptoms()
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for NLP processing."""
        try:
            # Try to load the specified model
            self.nlp = spacy.load(settings.SPACY_MODEL)
            logger.info(f"Loaded spaCy model: {settings.SPACY_MODEL}")
        except OSError:
            logger.warning(
                f"spaCy model {settings.SPACY_MODEL} not found. "
                f"Using blank English model. "
                f"Run: python -m spacy download {settings.SPACY_MODEL}"
            )
            # Fallback to blank model
            self.nlp = spacy.blank("en")
    
    def _load_medical_symptoms(self) -> set:
        """
        Load medical symptom vocabulary.
        
        In production, this should be loaded from a comprehensive medical ontology
        (e.g., SNOMED CT, ICD-10, or custom medical knowledge base).
        
        Returns:
            set: Set of known medical symptoms
        """
        # Core symptoms vocabulary (expand this in production)
        symptoms = {
            # Cardiovascular
            "chest pain", "heart palpitations", "irregular heartbeat",
            "shortness of breath", "breathing difficulty",
            
            # Neurological
            "headache", "dizziness", "lightheadedness", "vertigo",
            "confusion", "memory loss", "seizure", "numbness",
            "weakness", "paralysis", "vision changes", "blurred vision",
            
            # Gastrointestinal
            "nausea", "vomiting", "diarrhea", "constipation",
            "abdominal pain", "stomach pain", "bloating",
            "loss of appetite", "indigestion",
            
            # Respiratory
            "cough", "wheezing", "sore throat", "runny nose",
            "congestion", "difficulty breathing",
            
            # Musculoskeletal
            "joint pain", "muscle pain", "back pain", "neck pain",
            "stiffness", "swelling",
            
            # General
            "fever", "chills", "sweating", "fatigue", "weakness",
            "weight loss", "weight gain", "loss of consciousness",
            "bleeding", "rash", "itching",
            
            # Pain descriptors
            "sharp pain", "dull pain", "burning pain", "throbbing pain",
            "severe pain", "mild pain", "moderate pain",
        }
        
        return symptoms
    
    def extract_symptoms(self, text: str) -> List[ExtractedSymptom]:
        """
        Extract symptoms from text.
        
        Args:
            text: Patient symptom description
            
        Returns:
            List[ExtractedSymptom]: List of extracted symptoms with metadata
        """
        if not text:
            return []
        
        symptoms = []
        text_lower = text.lower()
        
        # Method 1: Pattern matching with medical vocabulary
        for symptom in self.medical_symptoms:
            if symptom in text_lower:
                start = text_lower.find(symptom)
                end = start + len(symptom)
                
                symptoms.append(ExtractedSymptom(
                    text=symptom,
                    normalized=self._normalize_symptom(symptom),
                    confidence=0.9,  # High confidence for exact matches
                    span=(start, end)
                ))
        
        # Method 2: spaCy NER extraction (if model supports medical entities)
        if self.nlp and hasattr(self.nlp, 'pipe_names'):
            doc = self.nlp(text)
            
            # Extract entities that might be symptoms
            for ent in doc.ents:
                if ent.label_ in ["SYMPTOM", "DISEASE", "MEDICAL_CONDITION"]:
                    symptoms.append(ExtractedSymptom(
                        text=ent.text.lower(),
                        normalized=self._normalize_symptom(ent.text.lower()),
                        confidence=0.8,
                        span=(ent.start_char, ent.end_char)
                    ))
        
        # Remove duplicates (keep highest confidence)
        unique_symptoms = {}
        for symptom in symptoms:
            if symptom.normalized not in unique_symptoms:
                unique_symptoms[symptom.normalized] = symptom
            elif symptom.confidence > unique_symptoms[symptom.normalized].confidence:
                unique_symptoms[symptom.normalized] = symptom
        
        extracted = list(unique_symptoms.values())
        
        logger.debug(f"Extracted {len(extracted)} symptoms from text")
        
        return extracted
    
    def _normalize_symptom(self, symptom: str) -> str:
        """
        Normalize symptom to standard form.
        
        Args:
            symptom: Raw symptom text
            
        Returns:
            str: Normalized symptom name
        """
        # Convert to lowercase and strip
        symptom = symptom.lower().strip()
        
        # Remove articles
        for article in ["a ", "an ", "the "]:
            if symptom.startswith(article):
                symptom = symptom[len(article):]
        
        # Normalize common variations
        normalizations = {
            "difficulty breathing": "shortness of breath",
            "cant breathe": "shortness of breath",
            "heart pain": "chest pain",
            "stomach pain": "abdominal pain",
            "belly pain": "abdominal pain",
            "throw up": "vomiting",
            "throwing up": "vomiting",
        }
        
        return normalizations.get(symptom, symptom)
    
    def get_symptom_categories(self, symptoms: List[ExtractedSymptom]) -> dict:
        """
        Categorize symptoms by body system.
        
        Args:
            symptoms: List of extracted symptoms
            
        Returns:
            dict: Symptoms grouped by category
        """
        categories = {
            "cardiovascular": ["chest pain", "heart palpitations", "shortness of breath"],
            "neurological": ["headache", "dizziness", "confusion", "seizure", "numbness"],
            "gastrointestinal": ["nausea", "vomiting", "diarrhea", "abdominal pain"],
            "respiratory": ["cough", "wheezing", "sore throat"],
            "general": ["fever", "fatigue", "weakness", "sweating"],
        }
        
        categorized = {cat: [] for cat in categories}
        categorized["other"] = []
        
        for symptom in symptoms:
            found = False
            for category, category_symptoms in categories.items():
                if any(cs in symptom.normalized for cs in category_symptoms):
                    categorized[category].append(symptom)
                    found = True
                    break
            
            if not found:
                categorized["other"].append(symptom)
        
        return categorized
