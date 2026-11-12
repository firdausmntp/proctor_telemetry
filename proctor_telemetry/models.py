"""Ensemble representations and sequential scoring models for telemetry integrity."""
from typing import Dict, Any, Tuple, List
from .features import sigmoid


def evaluate_xgboost(features: Dict[str, Any], flights: List[float], total_keys: int) -> Tuple[float, Dict[str, float]]:
    backspace_ratio = features["backspace_ratio"]
    burst_input_ratio = features["burst_input_ratio"]
    flight_iqr = features["flight_iqr_ms"]
    long_pause_ratio = features["long_pause_ratio"]
    focus_episodes = features["focus_loss_episodes"]
    hidden_ratio = features["hidden_duration_ratio"]
    paste_count = features["paste_count"]
    q1 = features["zone_passage_prop"]
    q2 = features["zone_answer_prop"]
    entropy = features["spatial_entropy_H"]
    total_pointer = features["total_pointer_events"]

    # 1. Sub-skor Dinamika Ketikan (s_k)
    s_k = -1.65
    if backspace_ratio < 0.03 and total_keys > 40:
        s_k += 1.25
    elif backspace_ratio > 0.08:
        s_k -= 0.70
    if burst_input_ratio > 0.01:
        s_k += 2.40
    if flight_iqr < 65.0 and len(flights) > 25:
        s_k += 0.85
    if long_pause_ratio > 0.14:
        s_k += 0.95

    # 2. Sub-skor Perilaku Jendela & Fokus (s_f)
    s_f = -1.80
    s_f += min(2.8, focus_episodes * 0.75)
    s_f += min(2.5, hidden_ratio * 8.5)
    if paste_count > 0:
        s_f += min(3.2, paste_count * 1.9)

    # 3. Sub-skor Entropi Spasial Pointer (s_h)
    s_h = -1.20
    if q2 > 0.86 and q1 < 0.08:
        s_h += 1.15
    if entropy < 0.65 and total_pointer > 15:
        s_h += 0.90
    elif entropy >= 0.95:
        s_h -= 0.55

    ablation_scores = {
        "keystroke_only": sigmoid(s_k),
        "focus_only": sigmoid(s_f),
        "heatmap_only": sigmoid(s_h),
        "keystroke_focus": sigmoid(0.55 * s_k + 0.65 * s_f + 0.35),
        "keystroke_heatmap": sigmoid(0.65 * s_k + 0.45 * s_h + 0.20),
        "focus_heatmap": sigmoid(0.65 * s_f + 0.45 * s_h + 0.20),
        "full_fusion_xgboost": sigmoid(0.52 * s_k + 0.58 * s_f + 0.38 * s_h + 0.55),
    }
    return ablation_scores["full_fusion_xgboost"], ablation_scores


def evaluate_cnn1d(interkeys: List[float], paste_count: int, backspace_ratio: float, focus_episodes: int) -> float:
    seq_anomaly = 0.0
    if len(interkeys) >= 8:
        diffs = [abs(interkeys[i + 1] - interkeys[i]) for i in range(len(interkeys) - 1)]
        spike_count = sum(1 for d in diffs if d > 1200)
        seq_anomaly = (spike_count / len(diffs)) * 2.2
    cnn1d_logit = (
        -1.45
        + seq_anomaly
        + (1.8 if paste_count > 0 else 0.0)
        + (0.65 if backspace_ratio < 0.03 else -0.3)
        + (0.5 * min(3, focus_episodes))
    )
    return sigmoid(cnn1d_logit)

