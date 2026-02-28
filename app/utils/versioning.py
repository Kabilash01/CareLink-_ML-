"""
CARELINK AI Microservice - Versioning Utilities

Model and service version management utilities.
"""

import json
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

from app.core.logging import logger


class VersionManager:
    """
    Manage model and service versions.
    
    Tracks:
    - Model versions
    - Service versions
    - Compatibility matrix
    - Version history
    """
    
    def __init__(self, version_file: Optional[Path] = None):
        """
        Initialize version manager.
        
        Args:
            version_file: Path to version metadata file
        """
        self.version_file = version_file or Path("app/models/model_metadata.json")
        self.current_version = None
        self.metadata = {}
        
        if self.version_file.exists():
            self._load_version_metadata()
    
    def _load_version_metadata(self):
        """Load version metadata from file."""
        try:
            with open(self.version_file, 'r') as f:
                self.metadata = json.load(f)
                self.current_version = self.metadata.get("version")
                logger.info(f"Loaded version metadata: {self.current_version}")
        except Exception as e:
            logger.error(f"Failed to load version metadata: {str(e)}")
    
    def get_version(self) -> str:
        """
        Get current version.
        
        Returns:
            str: Current version
        """
        return self.current_version or "unknown"
    
    def get_metadata(self) -> Dict:
        """
        Get full version metadata.
        
        Returns:
            Dict: Version metadata
        """
        return self.metadata
    
    def validate_compatibility(self, required_version: str) -> bool:
        """
        Check if current version is compatible with required version.
        
        Args:
            required_version: Required version string
            
        Returns:
            bool: True if compatible
        """
        if not self.current_version:
            return False
        
        # Simple version comparison (major.minor.patch)
        current_parts = self.current_version.lstrip('v').split('.')
        required_parts = required_version.lstrip('v').split('.')
        
        # Check major version compatibility
        if current_parts[0] != required_parts[0]:
            return False
        
        # Minor and patch versions can be greater or equal
        return True
    
    def create_version_metadata(
        self,
        version: str,
        training_date: datetime,
        accuracy: float,
        model_type: str,
        feature_count: int,
        training_samples: int,
        additional_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create version metadata dictionary.
        
        Args:
            version: Version string
            training_date: Training date
            accuracy: Model accuracy
            model_type: Model type
            feature_count: Number of features
            training_samples: Number of training samples
            additional_metadata: Additional metadata
            
        Returns:
            Dict: Version metadata
        """
        metadata = {
            "version": version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "training_date": training_date.isoformat() + "Z",
            "model_type": model_type,
            "performance": {
                "accuracy": accuracy
            },
            "features": {
                "feature_count": feature_count,
                "training_samples": training_samples
            }
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def save_version_metadata(self, metadata: Dict):
        """
        Save version metadata to file.
        
        Args:
            metadata: Version metadata
        """
        try:
            self.version_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.version_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Version metadata saved: {metadata.get('version')}")
            
        except Exception as e:
            logger.error(f"Failed to save version metadata: {str(e)}")
