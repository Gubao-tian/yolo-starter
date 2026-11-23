"""垃圾分类建议与看板面板的行为测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import trash_rules, visualize  # noqa: E402


class TestTrashRules(unittest.TestCase):
    def test_normalize_label_maps_english_alias_to_canonical_class(self):
        self.assertEqual(trash_rules.normalize_label(" Plastic Bottle "), "plastic")

    def test_advice_for_battery_is_hazardous_waste(self):
        self.assertIn("有害垃圾", trash_rules.advice_for("battery"))

    def test_summarize_counts_detections_and_returns_total(self):
        out = trash_rules.summarize([
            {"label": "plastic", "score": 0.9},
            {"label": "Plastic Bottle", "score": 0.8},
            {"label": "paper", "score": 0.7},
        ])
        self.assertEqual(out["total"], 3)
        counts = {item["label"]: item["count"] for item in out["items"]}
        self.assertEqual(counts, {"plastic": 2, "paper": 1})
        self.assertIn("可回收", out["headline"])


class TestClassificationPanel(unittest.TestCase):
    def test_panel_shows_counts_advice_and_escapes_html(self):
        panel = visualize.classification_panel_html([
            {"label": "plastic", "score": 0.9},
            {"label": "<script>", "score": 0.5},
        ])
        self.assertIn("投放建议", panel)
        self.assertIn("plastic", panel)
        self.assertIn("trash", panel)  # 未知类别被规整为其他垃圾，原始危险标签不应出现
        self.assertNotIn("<script>", panel)
        self.assertNotIn("&lt;script&gt;", panel)

    def test_caption_html_escapes_file_name(self):
        caption = visualize.caption_html("img_<a>&.png")
        self.assertIn("&lt;a&gt;", caption)
        self.assertNotIn("<a>", caption)


if __name__ == "__main__":
    unittest.main()
