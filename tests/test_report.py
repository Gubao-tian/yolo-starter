"""自包含 HTML 看板生成的行为测试。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import report
from yolo_starter.metrics import METRIC_KEYS


def context(mode="demo"):
    row = {"id": "E1", "name": "基准", "model": "yolov8n.pt", "imgsz": 320,
           "conf": 0.25, "hypothesis": "基线", "metrics": {k: 0.5 for k in METRIC_KEYS},
           "delta_map50": 0.0, "delta_map5095": 0.0, "delta_ms_per_image": 0.0,
           "is_baseline": True}
    return {
        "mode": mode,
        "generated_at": "2026-11-23 10:00:00",
        "dataset": "datasets/campus_trash/data.yaml",
        "source_note": "演示数据（非真实推理结果）" if mode == "demo" else "真实 YOLOv8 验证结果",
        "rows": [row],
        "conclusions": ["E1 的 mAP@0.50:0.95 为 0.500。"],
        "charts": {"mAP@0.50:0.95": "<svg></svg>", "单图耗时 (ms)": "<svg></svg>"},
        "visual_html": "<svg></svg>",
        "env": {"python": "3.13.14", "ultralytics_available": False},
        "commands": ["python scripts/run_all.py --mode demo"],
        "warnings": [],
    }


class TestBuildReport(unittest.TestCase):
    def test_report_covers_all_four_required_capabilities(self):
        html = report.build_report(context())
        for phrase in ("能看", "能跑", "能对比", "能复述"):
            self.assertIn(phrase, html)

    def test_demo_report_shows_illustrative_data_warning(self):
        html = report.build_report(context("demo"))
        self.assertIn("演示数据", html)
        self.assertIn("非真实", html)

    def test_full_report_does_not_claim_illustrative_data(self):
        html = report.build_report(context("full"))
        self.assertNotIn("演示数据（非真实推理结果）", html)
        self.assertIn("真实", html)

    def test_report_includes_experiment_ids_and_conclusions(self):
        html = report.build_report(context())
        self.assertIn("E1", html)
        self.assertIn("0.500", html)
        self.assertIn("<table", html)

    def test_report_escapes_untrusted_text(self):
        ctx = context()
        ctx["source_note"] = "<script>alert(1)</script>"
        html = report.build_report(ctx)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_report_is_self_contained_offline_html(self):
        html = report.build_report(context())
        self.assertIn("<style>", html)
        self.assertNotIn("http://", html.replace("http://www.w3.org/2000/svg", ""))
        self.assertTrue(html.strip().endswith("</html>"))

    def test_save_report_writes_file_and_returns_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.html"
            returned = report.save_report(context(), path)
            self.assertEqual(returned, path)
            self.assertTrue(path.exists())
            self.assertIn("能看", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
