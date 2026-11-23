"""书本伪标签数据集的纯逻辑测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _mod():
    try:
        from yolo_starter import book_dataset
        return book_dataset
    except ImportError:
        return None


class TestBookDataset(unittest.TestCase):
    def test_mirror_boxes_flips_x(self):
        m = _mod()
        self.assertIsNotNone(m)
        self.assertEqual(m.mirror_boxes([[10, 20, 30, 40]], 100),
                         [[70, 20, 90, 40]])

    def test_crop_boxes_clips_and_drops_outside(self):
        m = _mod()
        self.assertIsNotNone(m)
        out = m.crop_boxes([[10, 20, 30, 40], [200, 0, 210, 10]],
                           (5, 15, 25, 25))
        self.assertEqual(out, [[5, 5, 25, 25]])

    def test_to_yolo_label_normalizes(self):
        m = _mod()
        self.assertIsNotNone(m)
        lines = m.to_yolo_label([[10, 20, 30, 40]], 100, 200)
        self.assertEqual(lines, ["0 0.2 0.15 0.2 0.1"])

    def test_build_dataset_multi_writes_negative_labels(self):
        import tempfile
        from PIL import Image

        m = _mod()
        self.assertIsNotNone(m)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.png", root / "b.png"
            Image.new("RGB", (60, 60), "white").save(a)
            Image.new("RGB", (60, 60), "black").save(b)
            stats = m.build_dataset_multi(
                [(a, [{"label": "book", "score": 0.9, "box": [5, 5, 20, 20]}]),
                 (b, [])],
                root / "ds", n_crops=0)
            labels = list((root / "ds" / "labels").rglob("*.txt"))
            self.assertGreater(len(labels), 0)
            self.assertTrue(any(p.read_text(encoding="utf-8") == "" for p in labels))
            self.assertEqual(stats["train"] + stats["val"], len(labels))
            self.assertTrue((root / "ds" / "data.yaml").exists())


if __name__ == "__main__":
    unittest.main()
