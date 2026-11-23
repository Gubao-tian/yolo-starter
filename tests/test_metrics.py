"""指标归一化、对比表与自动结论的行为测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import metrics


def row(spec_id, map50, map5095, precision, recall, ms, params=3.2):
    return {
        "id": spec_id,
        "name": spec_id,
        "model": "yolov8n.pt",
        "imgsz": 320,
        "conf": 0.25,
        "hypothesis": "h",
        "metrics": {
            "map50": map50,
            "map5095": map5095,
            "precision": precision,
            "recall": recall,
            "ms_per_image": ms,
            "params_m": params,
        },
        "delta_map50": 0.0,
        "delta_map5095": 0.0,
        "delta_ms_per_image": 0.0,
        "is_baseline": spec_id == "E1",
    }


class TestNormalize(unittest.TestCase):
    def test_normalize_keeps_six_metrics_and_casts_to_float(self):
        raw = {"map50": "0.512", "map5095": 0.331, "precision": 0.64,
               "recall": 0.47, "ms_per_image": 18.4, "params_m": 3.2}
        out = metrics.normalize(raw)
        self.assertEqual(set(out), set(metrics.METRIC_KEYS))
        self.assertIsInstance(out["map50"], float)
        self.assertAlmostEqual(out["map50"], 0.512)

    def test_normalize_fills_missing_metric_with_zero(self):
        out = metrics.normalize({"map50": 0.5})
        self.assertEqual(out["map5095"], 0.0)
        self.assertEqual(out["ms_per_image"], 0.0)

    def test_normalize_clamps_ratio_metrics_into_unit_interval(self):
        out = metrics.normalize({"map50": 1.4, "precision": -0.2, "ms_per_image": -5})
        self.assertEqual(out["map50"], 1.0)
        self.assertEqual(out["precision"], 0.0)
        self.assertEqual(out["ms_per_image"], 0.0)

    def test_normalize_handles_none_and_bad_strings_as_zero(self):
        out = metrics.normalize({"map50": None, "recall": "n/a"})
        self.assertEqual(out["map50"], 0.0)
        self.assertEqual(out["recall"], 0.0)


class TestBuildComparison(unittest.TestCase):
    def setUp(self):
        from yolo_starter.experiments import load_config
        self.cfg = load_config()

    def test_build_comparison_produces_one_row_per_experiment(self):
        results = {s.id: {"map50": 0.5, "map5095": 0.3, "precision": 0.6,
                          "recall": 0.5, "ms_per_image": 20.0, "params_m": 3.2}
                   for s in self.cfg.select(["E1", "E2", "E3"])}
        rows = metrics.build_comparison(self.cfg.select(["E1", "E2", "E3"]), results, "E1")
        self.assertEqual([r["id"] for r in rows], ["E1", "E2", "E3"])

    def test_baseline_row_has_zero_deltas(self):
        results = {"E1": {"map50": 0.5, "map5095": 0.3, "precision": 0.6,
                          "recall": 0.5, "ms_per_image": 20.0, "params_m": 3.2}}
        rows = metrics.build_comparison(self.cfg.select(["E1"]), results, "E1")
        self.assertTrue(rows[0]["is_baseline"])
        self.assertEqual(rows[0]["delta_map50"], 0.0)
        self.assertEqual(rows[0]["delta_ms_per_image"], 0.0)

    def test_non_baseline_deltas_are_signed_against_baseline(self):
        results = {
            "E1": {"map50": 0.50, "map5095": 0.30, "precision": 0.60,
                   "recall": 0.50, "ms_per_image": 20.0, "params_m": 3.2},
            "E2": {"map50": 0.60, "map5095": 0.40, "precision": 0.70,
                   "recall": 0.55, "ms_per_image": 45.0, "params_m": 3.2},
        }
        rows = metrics.build_comparison(self.cfg.select(["E1", "E2"]), results, "E1")
        self.assertAlmostEqual(rows[1]["delta_map50"], 0.10)
        self.assertAlmostEqual(rows[1]["delta_map5095"], 0.10)
        self.assertAlmostEqual(rows[1]["delta_ms_per_image"], 25.0)

    def test_missing_result_is_treated_as_zero_metrics_not_crash(self):
        rows = metrics.build_comparison(self.cfg.select(["E1"]), {}, "E1")
        self.assertEqual(rows[0]["metrics"]["map50"], 0.0)


class TestConclusions(unittest.TestCase):
    def test_conclusions_name_best_accuracy_and_fastest_experiment(self):
        rows = [row("E1", 0.50, 0.30, 0.60, 0.50, 20.0),
                row("E2", 0.60, 0.40, 0.70, 0.55, 45.0),
                row("E3", 0.55, 0.35, 0.80, 0.40, 30.0)]
        text = "\n".join(metrics.derive_conclusions(rows))
        self.assertIn("E2", text)
        self.assertIn("E1", text)
        self.assertIn("mAP", text)

    def test_conclusions_handle_empty_input(self):
        conclusions = metrics.derive_conclusions([])
        self.assertTrue(conclusions)
        self.assertIn("暂无", conclusions[0])

    def test_conclusions_quantify_tradeoff_when_best_is_slower(self):
        rows = [row("E1", 0.50, 0.30, 0.60, 0.50, 20.0),
                row("E2", 0.60, 0.40, 0.70, 0.55, 45.0)]
        text = "\n".join(metrics.derive_conclusions(rows))
        self.assertIn("耗时", text)


class TestFormatMetric(unittest.TestCase):
    def test_format_metric_renders_three_decimals(self):
        self.assertEqual(metrics.format_metric(0.51234), "0.512")

    def test_format_metric_renders_signed_delta(self):
        self.assertEqual(metrics.format_delta(0.1), "+0.100")
        self.assertEqual(metrics.format_delta(-0.1), "-0.100")


if __name__ == "__main__":
    unittest.main()
