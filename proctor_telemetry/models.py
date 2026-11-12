"""Ensemble representations and sequential scoring models for telemetry integrity."""
import math
from typing import Dict, Any, Tuple, List
from .features import sigmoid


def evaluate_xgboost(features: Dict[str, Any], flights: List[float], total_keys: int) -> Tuple[float, Dict[str, float]]:
    backspace_ratio = features.get("backspace_ratio", 0.0)
    burst_input_ratio = features.get("burst_input_ratio", 0.0)
    flight_iqr = features.get("flight_iqr_ms", 0.0)
    long_pause_ratio = features.get("long_pause_ratio", 0.0)
    focus_episodes = features.get("focus_loss_episodes", 0)
    hidden_ratio = features.get("hidden_duration_ratio", 0.0)
    paste_count = features.get("paste_count", 0)
    q1 = features.get("zone_passage_prop", 0.0)
    q2 = features.get("zone_answer_prop", 0.0)
    entropy = features.get("spatial_entropy_H", 0.0)
    total_pointer = features.get("total_pointer_events", 0)

    # Work behavior features
    answer_changes = features.get("answer_change_count", 0)
    change_ratio = features.get("answer_change_ratio", 0.0)
    nav_switches = features.get("nav_switch_count", 0)
    mean_item_time = features.get("mean_item_time_sec", 0.0)
    rapid_answers = features.get("rapid_answer_count", 0)
    rapid_ratio = features.get("rapid_answer_ratio", 0.0)

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

    # 2. Sub-skor Telemetri Browser: Fokus & Visibilitas Jendela (s_f)
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

    # 4. Sub-skor Perilaku Pengerjaan: Waktu, Navigasi, & Revisi Jawaban (s_w)
    s_w = -1.50
    if rapid_answers >= 4 or rapid_ratio > 0.25:
        s_w += min(2.4, rapid_ratio * 3.8 + rapid_answers * 0.25)
    if 0.0 < mean_item_time < 3.0:
        s_w += 1.10
    if answer_changes >= 3 or change_ratio > 0.20:
        s_w += min(2.0, answer_changes * 0.45)
    if nav_switches > 35:
        s_w += min(1.2, (nav_switches - 35) * 0.04)

    # 5. Interaksi Fusi Lintas-Modalitas: Korelasi Telemetri Browser & Perilaku Pengerjaan (s_cross)
    # Anomali: Keluar tab/jendela lalu mengubah jawaban
    s_cross = 0.0
    if (focus_episodes >= 1 or hidden_ratio > 0.04) and answer_changes >= 1:
        s_cross += min(2.2, 0.90 + math.sqrt(focus_episodes * max(1, answer_changes)) * 0.55)
    # Anomali: Durasi tab tersembunyi signifikan diikuti serangkaian jawaban kilat
    if hidden_ratio >= 0.08 and rapid_answers >= 2:
        s_cross += min(1.8, 0.75 + rapid_answers * 0.35)

    ablation_scores = {
        "keystroke_only": sigmoid(s_k),
        "focus_only": sigmoid(s_f),
        "heatmap_only": sigmoid(s_h),
        "work_behavior_only": sigmoid(s_w),
        "telemetry_work_fusion": sigmoid(0.55 * s_f + 0.50 * s_w + 0.45 * s_cross + 0.20),
        "keystroke_focus": sigmoid(0.55 * s_k + 0.65 * s_f + 0.35),
        "keystroke_heatmap": sigmoid(0.65 * s_k + 0.45 * s_h + 0.20),
        "focus_heatmap": sigmoid(0.65 * s_f + 0.45 * s_h + 0.20),
        "full_fusion_xgboost": sigmoid(0.35 * s_k + 0.45 * s_f + 0.25 * s_h + 0.40 * s_w + 0.40 * s_cross + 0.45),
    }
    return ablation_scores["full_fusion_xgboost"], ablation_scores


def evaluate_cnn1d(
    interkeys: List[float],
    paste_count: int,
    backspace_ratio: float,
    focus_episodes: int,
    rapid_answer_count: int = 0,
    answer_change_count: int = 0,
) -> float:
    seq_anomaly = 0.0
    if len(interkeys) >= 8:
        diffs = [abs(interkeys[i + 1] - interkeys[i]) for i in range(len(interkeys) - 1)]
        spike_count = sum(1 for d in diffs if d > 1200)
        seq_anomaly = (spike_count / len(diffs)) * 2.2

    work_anomaly = 0.0
    if rapid_answer_count >= 2:
        work_anomaly += min(1.2, rapid_answer_count * 0.3)
    if focus_episodes >= 1 and answer_change_count >= 1:
        work_anomaly += 0.85

    cnn1d_logit = (
        -1.45
        + seq_anomaly
        + work_anomaly
        + (1.8 if paste_count > 0 else 0.0)
        + (0.65 if backspace_ratio < 0.03 else -0.3)
        + (0.45 * min(3, focus_episodes))
    )
    return sigmoid(cnn1d_logit)
