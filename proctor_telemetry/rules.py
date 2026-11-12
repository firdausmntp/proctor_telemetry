"""Heuristic rule verification for baseline proctor checks."""
from typing import Dict, Any


def evaluate_rules(features: Dict[str, Any]) -> int:
    paste_count = int(features.get("paste_count", 0))
    focus_episodes = int(features.get("focus_loss_episodes", 0))
    hidden_ratio = float(features.get("hidden_duration_ratio", 0.0))
    return 1 if (paste_count >= 1 or focus_episodes >= 3 or hidden_ratio >= 0.15) else 0

