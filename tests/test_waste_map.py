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


class TestExtendedHouseholdRules(unittest.TestCase):
    """双模型融合后，开放词汇检出的家居物品也要有投放规则。"""

    def test_lamp_is_recyclable(self):
        info = waste_map.waste_advice("lamp")
        self.assertTrue(info["is_waste"])
        self.assertTrue(info["known"])
        self.assertEqual(info["type"], "可回收物")

    def test_mirror_is_recyclable_not_non_waste(self):
        info = waste_map.waste_advice("mirror")
        self.assertTrue(info["is_waste"])
        self.assertEqual(info["type"], "可回收物")

    def test_vase_is_other_waste_not_non_waste(self):
        info = waste_map.waste_advice("vase")
        self.assertTrue(info["is_waste"])
        self.assertEqual(info["type"], "其他垃圾")

    def test_clothes_go_to_textile_recycling(self):
        info = waste_map.waste_advice("clothes")
        self.assertTrue(info["is_waste"])
        self.assertEqual(info["type"], "可回收物")

    def test_basket_is_recyclable(self):
        info = waste_map.waste_advice("basket")
        self.assertTrue(info["is_waste"])
        self.assertTrue(info["known"])

    def test_large_furniture_needs_bulk_pickup(self):
        for label in ("cabinet", "dresser", "bookshelf", "nightstand", "desk"):
            info = waste_map.waste_advice(label)
            self.assertTrue(info["is_waste"], label)
            self.assertEqual(info["type"], "其他垃圾", label)
            self.assertIn("大件", info["advice"], label)

    def test_curtain_and_blanket_are_other_waste(self):
        for label in ("curtain", "blanket", "carpet", "frame", "pillow",
                      "teddy bear", "shoes"):
            info = waste_map.waste_advice(label)
            self.assertTrue(info["is_waste"], label)
            self.assertTrue(info["known"], label)

    def test_world_only_electronics_have_rules(self):
        for label, kind in (("remote", "有害垃圾"), ("washing machine", "有害垃圾"),
                            ("air conditioner", "有害垃圾"), ("camera", "可回收物"),
                            ("headphones", "可回收物"), ("plastic bag", "其他垃圾"),
                            ("jar", "可回收物"), ("straw", "其他垃圾")):
            info = waste_map.waste_advice(label)
            self.assertTrue(info["is_waste"], label)
            self.assertTrue(info["known"], label)
            self.assertEqual(info["type"], kind, label)

    def test_extended_food_is_kitchen_waste(self):
        for label in ("egg", "tomato", "bread", "rice", "cheese", "cookie"):
            info = waste_map.waste_advice(label)
            self.assertEqual(info["type"], "厨余垃圾", label)

    def test_person_still_not_waste(self):
        self.assertFalse(waste_map.waste_advice("person")["is_waste"])


if __name__ == "__main__":
    unittest.main()
