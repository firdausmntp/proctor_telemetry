"""Feature extraction for keystroke dynamics, browser focus, and pointer entropy."""
import math
from typing import Dict, Any, Tuple, List


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_vals[low])
    return float(sorted_vals[low] * (high - pos) + sorted_vals[high] * (pos - low))


def iqr(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    s = sorted(vals)
    return round(percentile(s, 0.75) - percentile(s, 0.25), 2)


def median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(percentile(s, 0.5), 2)


def sigmoid(z: float) -> float:
    return round(1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, z)))), 4)


def calculate_entropy(n1: int, n2: int, n3: int) -> Tuple[float, float, float, float, int]:
    total = n1 + n2 + n3
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    q1 = round(n1 / total, 4)
    q2 = round(n2 / total, 4)
    q3 = round(n3 / total, 4)
    entropy = 0.0
    for q in (q1, q2, q3):
        if q > 0:
            entropy -= q * math.log2(q)
    return q1, q2, q3, round(entropy, 4), total


def extract_fusion_features(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[float], List[float], int]:
    dwells = [float(x) for x in payload.get("dwell_times", []) if 0 <= float(x) <= 5000]
    flights = [float(x) for x in payload.get("flight_times", []) if -500 <= float(x) <= 15000]
    interkeys = [float(x) for x in payload.get("interkey_times", []) if 0 <= float(x) <= 15000]

    total_keys = max(1, int(payload.get("total_keydowns", len(interkeys) + 1)))
    backspace_count = int(payload.get("backspace_count", 0))
    paste_count = int(payload.get("paste_count", 0))
    bulk_insert_count = int(payload.get("bulk_insert_count", 0))
    duration_sec = max(1.0, float(payload.get("duration_sec", 60.0)))

    dwell_med = median(dwells)
    dwell_iqr = iqr(dwells)
    flight_med = median(flights)
    flight_iqr = iqr(flights)
    iki_med = median(interkeys)
    long_pauses = sum(1 for x in interkeys if x > 1500.0)
    long_pause_ratio = round(long_pauses / max(1, len(interkeys)), 4)
    backspace_ratio = round(backspace_count / total_keys, 4)
    burst_input_ratio = round((paste_count + bulk_insert_count) / total_keys, 4)
    typing_cpm = round((total_keys / duration_sec) * 60.0, 2)

    focus_episodes = int(payload.get("focus_loss_episodes", 0))
    hidden_ms = float(payload.get("hidden_duration_ms", 0.0))
    total_ms = duration_sec * 1000.0
    hidden_ratio = round(min(1.0, hidden_ms / max(1.0, total_ms)), 4)

    zone_counts = payload.get("zone_counts", {"passage": 0, "answer": 0, "navigation": 0})
    n1 = int(zone_counts.get("passage", 0))
    n2 = int(zone_counts.get("answer", 0))
    n3 = int(zone_counts.get("navigation", 0))
    q1, q2, q3, entropy, total_pointer = calculate_entropy(n1, n2, n3)

    features = {
        "dwell_median_ms": dwell_med,
        "dwell_iqr_ms": dwell_iqr,
        "flight_median_ms": flight_med,
        "flight_iqr_ms": flight_iqr,
        "interkey_median_ms": iki_med,
        "long_pause_ratio": long_pause_ratio,
        "backspace_ratio": backspace_ratio,
        "burst_input_ratio": burst_input_ratio,
        "typing_cpm": typing_cpm,
        "focus_loss_episodes": focus_episodes,
        "hidden_duration_ms": round(hidden_ms, 1),
        "hidden_duration_ratio": hidden_ratio,
        "paste_count": paste_count,
        "zone_passage_prop": q1,
        "zone_answer_prop": q2,
        "zone_nav_prop": q3,
        "spatial_entropy_H": entropy,
        "total_pointer_events": total_pointer,
    }
    return features, flights, interkeys, total_keys

