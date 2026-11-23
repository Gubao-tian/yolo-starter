"""COCO 识别类别 -> 垃圾投放建议 的行为测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import waste_map  # noqa: E402


class TestWasteMap(unittest.TestCase):
    def test_bottle_is_recyclable_blue_bin(self):
        info = waste_map.waste_advice("bottle")
        self.assertTrue(info["is_waste"])
        self.assertEqual(info["type"], "可回收物")
        self.assertEqual(info["bin"], "蓝色")
        self.assertIn("瓶", info["advice"])

    def test_banana_is_kitchen_green_bin(self):
        info = waste_map.waste_advice("banana")
        self.assertEqual(info["type"], "厨余垃圾")
        self.assertEqual(info["bin"], "绿色")

    def test_cell_phone_is_hazardous(self):
        info = waste_map.waste_advice("cell phone")
        self.assertEqual(info["type"], "有害垃圾")
        self.assertEqual(info["bin"], "红色")

    def test_person_is_not_waste(self):
        info = waste_map.waste_advice("person")
        self.assertFalse(info["is_waste"])

    def test_unknown_label_falls_back_to_other_waste(self):
        info = waste_map.waste_advice("xyz_unknown")
        self.assertTrue(info["is_waste"])
        self.assertEqual(info["type"], "其他垃圾")
        self.assertEqual(info["bin"], "灰色")

    def test_summary_counts_and_headline(self):
        detections = [
            {"label": "bottle", "score": 0.9},
            {"label": "bottle", "score": 0.8},
            {"label": "banana", "score": 0.7},
            {"label": "person", "score": 0.6},
        ]
        s = waste_map.summarize(detections)
        self.assertEqual(s["total"], 4)
        self.assertEqual(s["waste_count"], 3)
        self.assertEqual(s["recyclable"], 2)
        labels = {item["label"] for item in s["items"]}
        self.assertIn("bottle", labels)
        self.assertIn("banana", labels)
        self.assertNotIn("person", labels)
        zh_by_label = {item["label"]: item.get("zh") for item in s["items"]}
        self.assertEqual(zh_by_label["bottle"], "瓶子")
        self.assertEqual(zh_by_label["banana"], "香蕉")
        self.assertIn("可回收", s["headline"])


if __name__ == "__main__":
    unittest.main()
