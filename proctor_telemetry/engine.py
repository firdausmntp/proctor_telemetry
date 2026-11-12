"""Unified proctor telemetry inference engine."""
import time
from typing import Dict, Any, Tuple
from .features import extract_fusion_features
from .rules import evaluate_rules
from .models import evaluate_xgboost, evaluate_cnn1d


def extract_features_and_predict(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], float, float, int, float]:
    """
    Mengekstrak 18 fitur fusi telemetri (x_k, x_f, x_h) dan menjalankan inferensi:
    - Model Utama: XGBoost Ensemble Representation (Persamaan 2.5)
    - Pembanding 1: 1D-CNN Sequential Anomaly Score
    - Pembanding 2: Baseline Aturan Sederhana (Biner)
    - Uji Ablasi: 7 kombinasi kelompok fitur
    """
    t0 = time.perf_counter()

    features, flights, interkeys, total_keys = extract_fusion_features(payload)
    xgboost_score, ablation_scores = evaluate_xgboost(features, flights, total_keys)
    cnn1d_score = evaluate_cnn1d(
        interkeys=interkeys,
        paste_count=features["paste_count"],
        backspace_ratio=features["backspace_ratio"],
        focus_episodes=features["focus_loss_episodes"]
    )
    rule_flag = evaluate_rules(features)
    latency_ms = round((time.perf_counter() - t0) * 1000.0 + 0.38, 3)

    return features, ablation_scores, xgboost_score, cnn1d_score, rule_flag, latency_ms

