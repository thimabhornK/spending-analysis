"""Spending Analysis — RAG-based UNSPSC spend classification (synthetic data).

Public API:

    from spend_classifier import SpendClassifier, POItem, MasterItem
    from spend_classifier.config import ClassifierConfig
    from spend_classifier.knowledge_base import load_master_items
"""

from .config import DEFAULT_CONFIG, ClassifierConfig
from .pipeline import SpendClassifier
from .schemas import (
    Candidate,
    Classification,
    ClassifyDecision,
    MasterItem,
    MatchMethod,
    POItem,
)
from .taxonomy import UNSPSCCode, parse, safe_parse

__version__ = "1.0.0"
__all__ = [
    "SpendClassifier",
    "ClassifierConfig",
    "DEFAULT_CONFIG",
    "POItem",
    "MasterItem",
    "Classification",
    "ClassifyDecision",
    "MatchMethod",
    "Candidate",
    "UNSPSCCode",
    "parse",
    "safe_parse",
]
