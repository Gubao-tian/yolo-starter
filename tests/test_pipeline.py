"""双模式流水线的端到端行为测试（在临时目录中运行，不污染项目）。"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import pipeline


class TestEnvironmentInfo(unittest.TestCase):
    def test_environment_info_reports_dependency_availability(self):
        info = pipeline.environment_info()
        for key in ("python", "platform", "ultralytics_available", "torch_available", "device"):
            self.assertIn(key, info)
        self.assertIsInstance(info["ultralytics_available"], bool)


class TestDemoPipeline(unittest.TestCase):
    def test_demo_run_produces_all_four_kinds_of_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(mode="demo", root=tmp)
            for key in ("results", "comparison", "summary", "report"):
                self.assertTrue(Path(result["artifacts"][key]).exists(), key)
            self.assertEqual(result["mode"], "demo")
            self.assertIn("演示", result["source_note"])

    def test_results_json_records_mode_dataset_experiments_and_conclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(mode="demo", root=tmp)
            payload = json.loads(Path(result["artifacts"]["results"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "demo")
            self.assertEqual(payload["dataset"], "datasets/campus_trash/data.yaml")
            self.assertEqual(len(payload["experiments"]), 3)
            self.assertTrue(payload["conclusions"])

    def test_comparison_csv_has_expected_header_and_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(mode="demo", root=tmp)
            with open(result["artifacts"]["comparison"], encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("map5095", rows[0])
            self.assertIn("delta_map50", rows[0])

    def test_selected_experiments_limit_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(mode="demo", experiment_ids=["E1", "E2"], root=tmp)
            self.assertEqual([r["id"] for r in result["rows"]], ["E1", "E2"])

    def test_report_html_contains_illustrative_data_badge(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(mode="demo", root=tmp)
            html = Path(result["artifacts"]["report"]).read_text(encoding="utf-8")
            self.assertIn("演示数据", html)
            self.assertIn("能对比", html)

    def test_demo_run_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = pipeline.run(mode="demo", root=tmp)
            first_rows = [row["metrics"] for row in first["rows"]]
            second = pipeline.run(mode="demo", root=tmp)
            second_rows = [row["metrics"] for row in second["rows"]]
            self.assertEqual(first_rows, second_rows)

    def test_invalid_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                pipeline.run(mode="magic", root=tmp)

    def test_custom_dataset_label_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(mode="demo", dataset="neu-det.yaml", root=tmp)
            self.assertEqual(result["dataset"], "neu-det.yaml")


class TestFullModeGuard(unittest.TestCase):
    def test_missing_dataset_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.yaml"
            with self.assertRaises(FileNotFoundError):
                pipeline.full_mode_guard(str(missing), epochs=20)

    def test_epochs_zero_returns_untrained_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.yaml"
            data.write_text("nc: 1\n", encoding="utf-8")
            warning = pipeline.full_mode_guard(str(data), epochs=0)
            self.assertIn("未针对本数据集训练", warning)

    def test_trained_epochs_has_no_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.yaml"
            data.write_text("nc: 1\n", encoding="utf-8")
            self.assertIsNone(pipeline.full_mode_guard(str(data), epochs=20))


if __name__ == "__main__":
    unittest.main()
