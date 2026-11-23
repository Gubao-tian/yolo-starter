"""坚果数据集构建：从人工标注框 + 数据增强生成 YOLO 训练集。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_starter import nut_dataset  # noqa: E402


def _png(path, size=(200, 150)):
    from PIL import Image
    Image.new("RGB", size, "white").save(path)
    return path


class TestNutDataset(unittest.TestCase):
    def test_load_annotations_reads_boxes(self):
        data = nut_dataset.load_annotations(ROOT / "datasets" / "nuts_real" / "annotations.json")
        self.assertEqual(data["class"], "nut")
        self.assertEqual(len(data["boxes"]), 4)
        self.assertTrue(all(len(b) == 4 for b in data["boxes"]))

    def test_to_yolo_label_normalizes_nut_box(self):
        lines = nut_dataset.to_yolo_label([[100, 50, 140, 90]], 200, 200)
        self.assertEqual(lines, ["0 0.6 0.35 0.2 0.2"])

    def test_build_creates_yolo_structure_with_nut_class(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _png(root / "nut.jpg", (200, 150))
            out = root / "ds"
            stats = nut_dataset.build([(src, [[50, 30, 80, 60], [120, 70, 150, 100]])],
                                      out, n_crops=2)
            self.assertTrue((out / "data.yaml").exists())
            self.assertTrue((out / "images" / "train").is_dir())
            self.assertTrue((out / "labels" / "train").is_dir())
            yaml = (out / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("names: [nut]", yaml)
            self.assertIn("nc: 1", yaml)
            train_imgs = list((out / "images" / "train").glob("*.jpg"))
            self.assertGreater(len(train_imgs), 0)
            self.assertGreater(stats["boxes"], 0)
            # 每个训练图片都有对应标签
            for img in train_imgs:
                label = out / "labels" / "train" / (img.stem + ".txt")
                self.assertTrue(label.exists())

    def test_build_handles_empty_boxes_as_negative(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = _png(root / "empty.jpg")
            out = root / "ds"
            stats = nut_dataset.build([(src, [])], out, n_crops=1)
            self.assertEqual(stats["boxes"], 0)
            self.assertEqual(stats["negative_sources"], 1)


if __name__ == "__main__":
    unittest.main()
