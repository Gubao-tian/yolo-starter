"""检测结果转换的纯逻辑测试（用假对象，不依赖 ultralytics）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import detector  # noqa: E402


class _Box:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy


class _Boxes:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


class _Result:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class TestPredictionsFromResult(unittest.TestCase):
    def test_maps_class_id_to_label_and_keeps_score_box(self):
        r = _Result({0: "bottle", 1: "banana"},
                    _Boxes([_Box([0], [0.87], [[10.2, 20.1, 100.0, 200.0]])]))
        out = detector.predictions_from_result(r)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "bottle")
        self.assertEqual(out[0].get("zh"), "瓶子")
        self.assertEqual(out[0]["score"], 0.87)
        self.assertEqual(out[0]["box"], [10.2, 20.1, 100.0, 200.0])

    def test_multiple_boxes_and_missing_names_fallback(self):
        r = _Result({}, _Boxes([
            _Box([2], [0.5], [[0, 0, 1, 1]]),
            _Box([3], [0.6], [[2, 2, 3, 3]]),
        ]))
        out = detector.predictions_from_result(r)
        self.assertEqual([o["label"] for o in out], ["2", "3"])
        self.assertEqual([o["score"] for o in out], [0.5, 0.6])

    def test_none_boxes_returns_empty(self):
        r = _Result({}, None)
        self.assertEqual(detector.predictions_from_result(r), [])

    def test_dining_table_gets_chinese_name(self):
        r = _Result({0: "dining table"},
                    _Boxes([_Box([0], [0.42], [[1, 2, 3, 4]])]))
        out = detector.predictions_from_result(r)
        self.assertEqual(out[0].get("zh"), "餐桌")

    def test_label_text_uses_chinese_name_and_score(self):
        fn = getattr(detector, "label_text", None)
        self.assertIsNotNone(fn)
        self.assertEqual(fn({"label": "tv", "score": 0.93}), "电视 0.93")
        self.assertEqual(fn({"label": "dining table", "score": 0.42}), "餐桌 0.42")

    def test_draw_detections_renders_chinese_boxes(self):
        fn = getattr(detector, "draw_detections", None)
        self.assertIsNotNone(fn)
        from PIL import Image
        img = Image.new("RGB", (100, 100), "white")
        out = fn(img, [{"label": "tv", "score": 0.93, "box": [5, 5, 50, 50]}])
        self.assertEqual(out.size, (100, 100))
        self.assertNotEqual(out.getpixel((5, 5)), (255, 255, 255))

    def test_top_label_indices_selects_highest_scores(self):
        fn = getattr(detector, "top_label_indices", None)
        self.assertIsNotNone(fn)
        dets = [{"score": 0.1}, {"score": 0.5}, {"score": 0.3}]
        self.assertEqual(fn(dets, 2), {1, 2})
        self.assertEqual(fn(dets, None), {0, 1, 2})


class TestDenseBooks(unittest.TestCase):
    def test_generate_tiles_with_overlap(self):
        fn = getattr(detector, "generate_tiles", None)
        self.assertIsNotNone(fn)
        self.assertEqual(
            fn(300, 250, 200, 40),
            [(0, 0, 200, 200), (160, 0, 140, 200),
             (0, 160, 200, 90), (160, 160, 140, 90)])

    def test_nms_suppresses_duplicate_boxes(self):
        fn = getattr(detector, "nms_detections", None)
        self.assertIsNotNone(fn)
        dets = [
            {"label": "book", "score": 0.9, "box": [0, 0, 10, 10]},
            {"label": "book", "score": 0.8, "box": [0, 0, 10, 10]},
            {"label": "book", "score": 0.7, "box": [20, 20, 30, 30]},
        ]
        out = fn(dets)
        self.assertEqual([d["score"] for d in out], [0.9, 0.7])

    def test_nms_keeps_different_labels_same_box(self):
        fn = getattr(detector, "nms_detections", None)
        dets = [
            {"label": "book", "score": 0.9, "box": [0, 0, 10, 10]},
            {"label": "cup", "score": 0.8, "box": [0, 0, 10, 10]},
        ]
        self.assertEqual(len(fn(dets)), 2)

    def test_detect_dense_books_maps_tile_boxes_back(self):
        from unittest.mock import MagicMock
        from PIL import Image
        import tempfile

        fn = getattr(detector, "detect_dense_books", None)
        self.assertIsNotNone(fn)
        b = MagicMock()
        b.cls = [73]
        b.conf = [0.4]
        b.xyxy = [[5, 5, 15, 15]]
        r = MagicMock()
        r.names = {73: "book"}
        r.boxes = [b]
        fake_model = MagicMock()
        fake_model.names = {73: "book"}
        fake_model.predict.return_value = [r]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "books.png"
            Image.new("RGB", (240, 240), "white").save(path)
            out = fn(fake_model, str(path), conf=0.3)
        self.assertEqual(fake_model.predict.call_count, 4)
        self.assertEqual(len(out), 4)
        self.assertIn([165, 165, 175, 175], [d["box"] for d in out])
        self.assertTrue(all(d["zh"] == "书本" for d in out))


class TestFusedWaste(unittest.TestCase):
    """垃圾分类双模型融合：COCO 框优先，世界模式只补 COCO 认不出的。"""

    def test_merge_keeps_non_overlapping_world_boxes(self):
        fn = getattr(detector, "merge_fused_detections", None)
        self.assertIsNotNone(fn)
        coco = [{"label": "bottle", "score": 0.9, "box": [0, 0, 10, 10]}]
        world = [{"label": "lamp", "score": 0.8, "box": [50, 50, 60, 60]}]
        merged = fn(coco, world)
        self.assertEqual(len(merged), 2)
        sources = {d["source"] for d in merged}
        self.assertEqual(sources, {"coco", "world"})

    def test_merge_drops_world_box_overlapping_coco(self):
        fn = getattr(detector, "merge_fused_detections", None)
        coco = [{"label": "cup", "score": 0.9, "box": [0, 0, 10, 10]}]
        world = [{"label": "bottle", "score": 0.95, "box": [1, 1, 9, 9]}]
        merged = fn(coco, world, iou_threshold=0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["label"], "cup")
        self.assertEqual(merged[0]["source"], "coco")

    def test_merge_keeps_world_box_below_iou_threshold(self):
        fn = getattr(detector, "merge_fused_detections", None)
        coco = [{"label": "cup", "score": 0.9, "box": [0, 0, 10, 10]}]
        world = [{"label": "lamp", "score": 0.8, "box": [8, 8, 40, 40]}]  # IoU 很低
        merged = fn(coco, world, iou_threshold=0.5)
        self.assertEqual(len(merged), 2)

    def test_merge_sorts_by_score_desc(self):
        fn = getattr(detector, "merge_fused_detections", None)
        coco = [{"label": "cup", "score": 0.5, "box": [0, 0, 5, 5]}]
        world = [{"label": "lamp", "score": 0.9, "box": [50, 50, 60, 60]}]
        merged = fn(coco, world)
        self.assertEqual([d["score"] for d in merged], [0.9, 0.5])

    def test_predict_fused_combines_and_counts(self):
        from unittest.mock import patch
        coco = [{"label": "bottle", "zh": "瓶子", "score": 0.9, "box": [0, 0, 10, 10]}]
        world = [{"label": "lamp", "zh": "台灯", "score": 0.8, "box": [50, 50, 60, 60]}]
        with patch.object(detector, "predict", return_value={"detections": coco}) as p, \
             patch.object(detector, "predict_world",
                          return_value={"detections": world}) as w:
            out = detector.predict_fused("fake.jpg", conf=0.25, imgsz=960)
        self.assertEqual(out["mode"], "waste")
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["coco_count"], 1)
        self.assertEqual(out["world_count"], 1)
        self.assertEqual(p.call_args.kwargs.get("imgsz"), 960)
        self.assertTrue(out["detections"][0]["score"] >= out["detections"][1]["score"])


if __name__ == "__main__":
    unittest.main()
