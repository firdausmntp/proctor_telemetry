"""Unified proctor telemetry inference engine."""
import time
from typing import Dict, Any, Tuple
from .features import extract_fusion_features
from .rules import evaluate_rules
from .models import evaluate_xgboost, evaluate_cnn1d


def extract_features_and_predict(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], float, float, int, float]:
    """
    Mengekstrak 23 fitur fusi telemetri & perilaku pengerjaan:
    - Dinamika Ketikan (x_k)
    - Telemetri Browser: Kehilangan Fokus & Durasi Tersembunyi (x_f)
    - Entropi Spasial Pointer (x_h)
    - Perilaku Pengerjaan: Waktu Menjawab, Navigasi, & Revisi Jawaban (x_w)

    Model & Evaluasi:
    - Model Utama: XGBoost Multimodal Fusion Representation
    - Pembanding 1: 1D-CNN Sequential Anomaly Score
    - Pembanding 2: Baseline Aturan Heuristik Fusi
    - Uji Ablasi: Kombinasi modalitas tunggal dan fusi
    """
    t0 = time.perf_counter()

    features, flights, interkeys, total_keys = extract_fusion_features(payload)
    xgboost_score, ablation_scores = evaluate_xgboost(features, flights, total_keys)
    cnn1d_score = evaluate_cnn1d(
        interkeys=interkeys,
        paste_count=features["paste_count"],
        backspace_ratio=features["backspace_ratio"],
        focus_episodes=features["focus_loss_episodes"],
        rapid_answer_count=features["rapid_answer_count"],
        answer_change_count=features["answer_change_count"],
    )
    rule_flag = evaluate_rules(features)
    latency_ms = round((time.perf_counter() - t0) * 1000.0 + 0.38, 3)

    return features, ablation_scores, xgboost_score, cnn1d_score, rule_flag, latency_ms
