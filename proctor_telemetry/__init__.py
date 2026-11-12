"""Proctor Telemetry Package: Multimodal behavioral integrity detection engine."""
from .features import (
    percentile,
    iqr,
    median,
    sigmoid,
    calculate_entropy,
    extract_fusion_features,
)
from .rules import evaluate_rules
from .models import evaluate_xgboost, evaluate_cnn1d
from .engine import extract_features_and_predict

__version__ = "0.1.0"
__all__ = [
    "percentile",
    "iqr",
    "median",
    "sigmoid",
    "calculate_entropy",
    "extract_fusion_features",
    "evaluate_rules",
    "evaluate_xgboost",
    "evaluate_cnn1d",
    "extract_features_and_predict",
]

