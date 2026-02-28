"""Test configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_triage_request():
    """Sample triage request payload."""
    return {
        "symptoms_text": "I have chest pain and sweating",
        "age": 52,
        "duration_days": 1,
        "chronic_conditions": ["diabetes"],
        "language": "en"
    }


@pytest.fixture
def sample_feature_vector():
    """Sample feature vector."""
    import numpy as np
    return np.random.random(100)
