"""Unit tests for proctor-telemetry package."""
import unittest
from proctor_telemetry import (
    percentile,
    iqr,
    median,
    sigmoid,
    calculate_entropy,
    extract_fusion_features,
    evaluate_rules,
    evaluate_xgboost,
    evaluate_cnn1d,
    extract_features_and_predict,
)


class TestProctorTelemetry(unittest.TestCase):
    def setUp(self):
        self.sample_payload = {
            "dwell_times": [85.0, 92.0, 78.0, 110.0, 95.0, 88.0],
            "flight_times": [140.0, 120.0, 190.0, 130.0, 150.0],
            "interkey_times": [225.0, 212.0, 268.0, 240.0, 238.0],
            "total_keydowns": 240,
            "backspace_count": 14,
            "paste_count": 0,
            "bulk_insert_count": 0,
            "focus_loss_episodes": 0,
            "hidden_duration_ms": 0.0,
            "duration_sec": 300.0,
            "zone_counts": {
                "passage": 45,
                "answer": 80,
                "navigation": 15,
            },
            "answer_change_count": 1,
            "nav_switch_count": 8,
            "item_response_times": [12.5, 14.0, 8.2, 19.1, 15.3],
            "total_questions_attempted": 20,
        }

    def test_math_utilities(self):
        self.assertEqual(percentile([], 0.5), 0.0)
        self.assertEqual(percentile([10.0, 20.0, 30.0], 0.5), 20.0)
        self.assertEqual(median([1.0, 3.0, 2.0]), 2.0)
        self.assertEqual(iqr([10.0]), 0.0)
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=3)

    def test_spatial_entropy(self):
        q1, q2, q3, h, total = calculate_entropy(45, 80, 15)
        self.assertEqual(total, 140)
        self.assertGreater(h, 0.0)
        self.assertAlmostEqual(q1 + q2 + q3, 1.0, places=2)

    def test_feature_extraction(self):
        features, flights, interkeys, total_keys = extract_fusion_features(self.sample_payload)
        self.assertEqual(total_keys, 240)
        self.assertIn("dwell_median_ms", features)
        self.assertIn("spatial_entropy_H", features)
        self.assertIn("burst_input_ratio", features)
        self.assertIn("answer_change_count", features)
        self.assertIn("nav_switch_count", features)
        self.assertIn("mean_item_time_sec", features)
        self.assertIn("rapid_answer_count", features)
        self.assertEqual(features["paste_count"], 0)
        self.assertEqual(features["answer_change_count"], 1)
        self.assertEqual(features["nav_switch_count"], 8)

    def test_rule_evaluation(self):
        clean_features = {
            "paste_count": 0,
            "focus_loss_episodes": 0,
            "hidden_duration_ratio": 0.02,
            "answer_change_count": 0,
            "rapid_answer_count": 0,
        }
        self.assertEqual(evaluate_rules(clean_features), 0)

        flagged_paste = {
            "paste_count": 1,
            "focus_loss_episodes": 0,
            "hidden_duration_ratio": 0.0,
            "answer_change_count": 0,
            "rapid_answer_count": 0,
        }
        self.assertEqual(evaluate_rules(flagged_paste), 1)

        flagged_focus = {
            "paste_count": 0,
            "focus_loss_episodes": 4,
            "hidden_duration_ratio": 0.0,
            "answer_change_count": 0,
            "rapid_answer_count": 0,
        }
        self.assertEqual(evaluate_rules(flagged_focus), 1)

        # Cross-modal fusion rule: focus loss accompanied by answer revisions
        flagged_cross = {
            "paste_count": 0,
            "focus_loss_episodes": 1,
            "hidden_duration_ratio": 0.05,
            "answer_change_count": 2,
            "rapid_answer_count": 0,
        }
        self.assertEqual(evaluate_rules(flagged_cross), 1)

        # Cross-modal fusion rule: hidden duration followed by rapid answers
        flagged_rapid = {
            "paste_count": 0,
            "focus_loss_episodes": 0,
            "hidden_duration_ratio": 0.10,
            "answer_change_count": 0,
            "rapid_answer_count": 4,
        }
        self.assertEqual(evaluate_rules(flagged_rapid), 1)

    def test_model_inference_pipeline(self):
        features, ablation, xgb_score, cnn_score, rule_flag, latency = extract_features_and_predict(self.sample_payload)
        self.assertIsInstance(features, dict)
        self.assertEqual(len(ablation), 9)
        self.assertIn("work_behavior_only", ablation)
        self.assertIn("telemetry_work_fusion", ablation)
        self.assertGreaterEqual(xgb_score, 0.0)
        self.assertLessEqual(xgb_score, 1.0)
        self.assertGreaterEqual(cnn_score, 0.0)
        self.assertLessEqual(cnn_score, 1.0)
        self.assertIn(rule_flag, [0, 1])
        self.assertGreater(latency, 0.0)


if __name__ == "__main__":
    unittest.main()
