"""YOLO-World 后端测试：用假模型验证词表设置与结果转换，不依赖真实权重。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import detector  # noqa: E402


def _fake_result(names, boxes_data):
    """boxes_data: [(cls_id, conf, [x1,y1,x2,y2]), ...]"""
    r = MagicMock()
    r.names = names
    boxes = []
    for cls_id, conf, xyxy in boxes_data:
        b = MagicMock()
        b.cls = [cls_id]
        b.conf = [conf]
        b.xyxy = xyxy
        boxes.append(b)
    r.boxes = boxes
    return r


class TestWorldBackend(unittest.TestCase):
    def tearDown(self):
        detector.load_world_model.cache_clear()

    def test_load_world_model_checks_file(self):
        with self.assertRaises(FileNotFoundError):
            detector.load_world_model("does/not/exist.pt")

    def test_set_classes_called_once_per_load(self):
        fake_model = MagicMock()
        fake_model.predict.return_value = []
        with patch.object(detector, "load_world_model", return_value=fake_model) as load:
            detector.predict_world("x.jpg", words=["bread", "bowl"])
            detector.predict_world("x.jpg", words=["bread", "bowl"])
        # 两次调用复用同一模型，但 set_classes 每次都会调用以确保词表正确
        self.assertEqual(load.call_count, 2)
        self.assertEqual(fake_model.set_classes.call_count, 2)
        fake_model.set_classes.assert_called_with(["bread", "bowl"])

    def test_predict_world_returns_structured(self):
        names = {0: "bread", 1: "bowl"}
        fake_model = MagicMock()
        fake_model.predict.return_value = [_fake_result(
            names, [(0, 0.39, [[1, 2, 3, 4]]), (1, 0.72, [[5, 6, 7, 8]])])]
        with patch.object(detector, "load_world_model", return_value=fake_model):
            out = detector.predict_world(
                "x.jpg", words=["bread", "bowl"], weights_path="w.pt", conf=0.2, device="cpu")
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["mode"], "world")
        self.assertEqual(out["detections"][0]["label"], "bowl")  # 按分数降序
        self.assertEqual(out["detections"][0]["score"], 0.72)
        self.assertEqual(out["detections"][1]["label"], "bread")
        self.assertEqual(out["words"], ["bread", "bowl"])

    def test_predict_world_defaults_to_vocab(self):
        from yolo_starter import objects_vocab as ov
        fake_model = MagicMock()
        fake_model.predict.return_value = []
        with patch.object(detector, "load_world_model", return_value=fake_model):
            out = detector.predict_world("x.jpg")
        self.assertEqual(out["words"], ov.WORDS)

    def test_predictions_from_result_handles_no_boxes(self):
        self.assertEqual(detector.predictions_from_result(MagicMock(boxes=None)), [])


if __name__ == "__main__":
    unittest.main()
