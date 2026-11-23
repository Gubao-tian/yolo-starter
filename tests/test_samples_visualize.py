"""示例数据与纯 SVG 可视化的行为测试。"""
import base64
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import samples, visualize
from yolo_starter.metrics import METRIC_KEYS

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class TestSamples(unittest.TestCase):
    def test_demo_results_only_returns_requested_experiments(self):
        out = samples.demo_results(["E1", "E3"])
        self.assertEqual(set(out), {"E1", "E3"})

    def test_every_demo_result_has_all_six_metrics(self):
        for raw in samples.demo_results(["E1", "E2", "E3", "E4"]).values():
            self.assertEqual(set(raw), set(METRIC_KEYS))

    def test_demo_note_marks_data_as_illustrative(self):
        note = samples.demo_note()
        self.assertIn("演示", note)
        self.assertIn("非真实", note)

    def test_demo_detections_have_labels_scores_and_normalized_boxes(self):
        boxes = samples.demo_detections()
        self.assertGreaterEqual(len(boxes), 2)
        for box in boxes:
            self.assertTrue(box["label"])
            self.assertGreater(box["score"], 0)
            self.assertLessEqual(box["score"], 1)
            for key in ("x", "y", "w", "h"):
                self.assertGreaterEqual(box[key], 0)
                self.assertLessEqual(box[key], 1)


class TestBarSvg(unittest.TestCase):
    def test_bar_svg_renders_one_bar_per_value(self):
        svg = visualize.bar_svg([0.3, 0.4, 0.5], ["E1", "E2", "E3"], "mAP", unit="")
        self.assertIn("<svg", svg)
        self.assertEqual(svg.count('class="bar"'), 3)

    def test_bar_svg_escapes_labels_and_title(self):
        svg = visualize.bar_svg([0.5], ["<script>&"], "a&b")
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("a&amp;b", svg)

    def test_bar_svg_handles_empty_input(self):
        svg = visualize.bar_svg([], [], "空")
        self.assertIn("<svg", svg)
        self.assertIn("暂无数据", svg)

    def test_bar_svg_handles_all_zero_values_without_error(self):
        svg = visualize.bar_svg([0.0, 0.0], ["E1", "E2"], "零")
        self.assertIn("<svg", svg)


class TestTableAndDetection(unittest.TestCase):
    def test_comparison_table_contains_all_experiment_ids(self):
        rows = [
            {"id": "E1", "name": "基准", "model": "yolov8n.pt", "imgsz": 320,
             "conf": 0.25, "metrics": {k: 0.5 for k in METRIC_KEYS},
             "delta_map50": 0.0, "delta_map5095": 0.0, "delta_ms_per_image": 0.0,
             "is_baseline": True},
            {"id": "E2", "name": "提高分辨率", "model": "yolov8n.pt", "imgsz": 640,
             "conf": 0.25, "metrics": {k: 0.6 for k in METRIC_KEYS},
             "delta_map50": 0.1, "delta_map5095": 0.1, "delta_ms_per_image": 20.0,
             "is_baseline": False},
        ]
        html = visualize.comparison_table_html(rows)
        self.assertIn("<table", html)
        self.assertIn("E1", html)
        self.assertIn("E2", html)
        self.assertIn("+0.100", html)

    def test_comparison_table_escapes_experiment_name(self):
        rows = [{"id": "E1", "name": "<img src=x>", "model": "yolov8n.pt",
                 "imgsz": 320, "conf": 0.25, "metrics": {k: 0.0 for k in METRIC_KEYS},
                 "delta_map50": 0.0, "delta_map5095": 0.0, "delta_ms_per_image": 0.0,
                 "is_baseline": True}]
        self.assertNotIn("<img src=x>", visualize.comparison_table_html(rows))

    def test_detection_svg_draws_boxes_and_labels(self):
        svg = visualize.detection_svg(samples.demo_detections())
        self.assertIn("<svg", svg)
        self.assertIn("plastic", svg)
        self.assertTrue(svg.count("<rect") >= 2)

    def test_image_data_uri_uses_base64_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.png"
            path.write_bytes(TINY_PNG)
            uri = visualize.image_data_uri(path)
            self.assertTrue(uri.startswith("data:image/png;base64,"))
            self.assertIn(base64.b64encode(TINY_PNG).decode("ascii"), uri)

    def test_image_data_uri_returns_empty_string_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(visualize.image_data_uri(Path(tmp) / "nope.png"), "")


if __name__ == "__main__":
    unittest.main()
