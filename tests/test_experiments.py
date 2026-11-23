"""实验矩阵加载与校验的行为测试。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import experiments as ex


class TestLoadConfig(unittest.TestCase):
    def test_load_default_config_has_four_controlled_experiments(self):
        cfg = ex.load_config()
        self.assertEqual(len(cfg.experiments), 4)
        self.assertEqual(cfg.baseline, "E1")
        self.assertEqual(cfg.dataset["default"], "datasets/campus_trash/data.yaml")

    def test_each_experiment_has_complete_and_valid_parameters(self):
        cfg = ex.load_config()
        for spec in cfg.experiments:
            self.assertTrue(spec.id)
            self.assertIsInstance(spec.imgsz, int)
            self.assertGreater(spec.imgsz, 0)
            self.assertGreater(spec.conf, 0)
            self.assertLessEqual(spec.conf, 1)
            self.assertGreater(spec.iou, 0)
            self.assertLessEqual(spec.iou, 1)
            self.assertTrue(spec.model.endswith(".pt"))
            self.assertTrue(spec.hypothesis)

    def test_baseline_spec_is_resolvable(self):
        cfg = ex.load_config()
        self.assertEqual(cfg.baseline_spec().id, "E1")

    def test_select_returns_requested_ids_in_requested_order(self):
        cfg = ex.load_config()
        picked = cfg.select(["E3", "E1"])
        self.assertEqual([s.id for s in picked], ["E3", "E1"])

    def test_select_without_ids_uses_fast_set(self):
        cfg = ex.load_config()
        self.assertEqual([s.id for s in cfg.select()], list(cfg.fast_set))

    def test_select_rejects_unknown_id(self):
        cfg = ex.load_config()
        with self.assertRaises(ValueError):
            cfg.select(["E1", "NOPE"])

    def test_duplicate_ids_are_rejected(self):
        payload = {
            "schema_version": 1,
            "dataset": {"default": "datasets/campus_trash/data.yaml"},
            "baseline": "E1",
            "fast_set": ["E1"],
            "experiments": [
                {"id": "E1", "name": "a", "model": "yolov8n.pt", "imgsz": 320,
                 "conf": 0.25, "iou": 0.7, "hypothesis": "h"},
                {"id": "E1", "name": "b", "model": "yolov8n.pt", "imgsz": 640,
                 "conf": 0.25, "iou": 0.7, "hypothesis": "h"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                ex.load_config(path)

    def test_baseline_not_in_experiments_is_rejected(self):
        payload = {
            "schema_version": 1,
            "dataset": {"default": "datasets/campus_trash/data.yaml"},
            "baseline": "E9",
            "fast_set": ["E1"],
            "experiments": [
                {"id": "E1", "name": "a", "model": "yolov8n.pt", "imgsz": 320,
                 "conf": 0.25, "iou": 0.7, "hypothesis": "h"}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                ex.load_config(path)


if __name__ == "__main__":
    unittest.main()
