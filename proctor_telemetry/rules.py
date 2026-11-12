"""Heuristic rule verification for baseline proctor checks."""
from typing import Dict, Any


def evaluate_rules(features: Dict[str, Any]) -> int:
    """
    Evaluasi aturan deterministik berbasis batas heuristik:
    1. Paste terdeteksi (paste_count >= 1)
    2. Kehilangan fokus berat (focus_loss_episodes >= 3 atau hidden_duration_ratio >= 0.15)
    3. Fusi Telemetri + Perilaku: Keluar jendela lalu merevisi jawaban (focus >= 1 dan revisions >= 2)
    4. Fusi Telemetri + Perilaku: Tab disembunyikan lalu menjawab kilat (hidden_ratio >= 0.08 dan rapid >= 3)
    """
    paste_count = int(features.get("paste_count", 0))
    focus_episodes = int(features.get("focus_loss_episodes", 0))
    hidden_ratio = float(features.get("hidden_duration_ratio", 0.0))
    answer_changes = int(features.get("answer_change_count", 0))
    rapid_answers = int(features.get("rapid_answer_count", 0))

    if paste_count >= 1 or focus_episodes >= 3 or hidden_ratio >= 0.15:
        return 1
    if (focus_episodes >= 1 or hidden_ratio >= 0.04) and answer_changes >= 2:
        return 1
    if hidden_ratio >= 0.08 and rapid_answers >= 3:
        return 1
    return 0
