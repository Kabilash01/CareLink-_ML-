"""
CARELINK AI Microservice - Text Preprocessor

NLP preprocessing pipeline for symptom text normalization.
Handles tokenization, lemmatization, and medical text cleaning.
"""

import re
from typing import List
from app.core.logging import logger


class TextPreprocessor:
    """
    Text preprocessing for medical symptom descriptions.
    
    Performs:
    - Lowercasing
    - Punctuation removal
    - Token normalization
    - Medical abbreviation expansion
    - Stop word removal (selective)
    """
    
    def __init__(self):
        """Initialize preprocessor with medical abbreviations and stop words."""
        
        # Medical abbreviations mapping
        self.abbreviations = {
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "rx": "medication",
            "hx": "history",
            "dx": "diagnosis",
            "sx": "symptoms",
            "pt": "patient",
            "sob": "shortness of breath",
            "cp": "chest pain",
            "n/v": "nausea vomiting",
            "h/a": "headache",
            "abd": "abdominal",
            "c/o": "complaining of",
        }
        
        # Stop words to remove (selective - keep medical context)
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "as", "by", "with", "from", "about"
        }
        
        logger.info("TextPreprocessor initialized")
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess symptom text.
        
        Args:
            text: Raw symptom description
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Remove special characters but keep spaces and some punctuation
        text = re.sub(r'[^a-z0-9\s\-/]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove excessive punctuation
        text = re.sub(r'[-/]{2,}', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize preprocessed text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List[str]: List of tokens
        """
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Filter very short tokens (likely not meaningful)
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List[str]: Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations.
        
        Args:
            text: Text with abbreviations
            
        Returns:
            str: Text with expanded abbreviations
        """
        for abbr, expansion in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text)
        
        return text
    
    def normalize_medical_terms(self, text: str) -> str:
        """
        Normalize medical terminology variations.
        
        Args:
            text: Text with medical terms
            
        Returns:
            str: Text with normalized terms
        """
        # Common symptom variations
        normalizations = {
            r'\bfever|feverish|febrile\b': 'fever',
            r'\bnausea|nauseous|nauseated\b': 'nausea',
            r'\bvomit|vomiting|throw up|throwing up\b': 'vomiting',
            r'\bdizzy|dizziness|lightheaded|light headed\b': 'dizziness',
            r'\bpain|painful|ache|aching|hurt|hurting\b': 'pain',
            r'\bshort of breath|difficulty breathing|cant breathe|can\'t breathe\b': 'shortness of breath',
        }
        
        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def process_pipeline(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw symptom text
            
        Returns:
            List[str]: Processed tokens
        """
        # Step 1: Basic preprocessing
        text = self.preprocess(text)
        
        # Step 2: Normalize medical terms
        text = self.normalize_medical_terms(text)
        
        # Step 3: Tokenize
        tokens = self.tokenize(text)
        
        # Step 4: Remove stop words (optional - keep for context)
        # tokens = self.remove_stop_words(tokens)
        
        logger.debug(f"Preprocessed: {text} -> {tokens}")
        
        return tokens
