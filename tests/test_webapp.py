"""网页上传识别服务的行为测试（不加载真实模型）。"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import app as webapp  # noqa: E402


class TestWebApp(unittest.TestCase):
    def setUp(self):
        webapp.app.config["TESTING"] = True
        self.client = webapp.app.test_client()

    def test_health_reports_weights_state(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["ok"])

    def test_index_page_has_upload_controls(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("开始识别", html)
        self.assertIn("置信度阈值", html)

    def test_samples_lists_selected_test_images(self):
        resp = self.client.get("/samples")
        self.assertEqual(resp.status_code, 200)
        items = resp.get_json()["items"]
        self.assertGreaterEqual(len(items), 8)
        self.assertTrue(all(i["url"].startswith("/sample_image/") for i in items))

    def test_sample_image_rejects_path_traversal(self):
        resp = self.client.get("/sample_image/..%2F..%2FREADME.md")
        self.assertIn(resp.status_code, (403, 404))

    def test_predict_rejects_bad_extension(self):
        data = {"image": (io.BytesIO(b"x"), "note.txt")}
        resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_predict_rejects_missing_image(self):
        resp = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_index_page_has_mode_switch(self):
        resp = self.client.get("/")
        html = resp.get_data(as_text=True)
        self.assertIn("全物体提取", html)
        self.assertIn("密集书本", html)
        self.assertIn("坚果识别", html)
        self.assertIn("mode", html)

    def test_nut_mode_filters_large_boxes(self):
        # 坚果是小目标：应按框占图面积比例过滤掉误检的大框
        filter_fn = getattr(webapp, "filter_nut_boxes", None)
        self.assertIsNotNone(filter_fn)
        detections = [
            {"label": "nut", "score": 0.97, "box": [158, 86, 189, 149]},   # 小框，真坚果
            {"label": "nut", "score": 0.92, "box": [326, 176, 622, 434]},   # 大框，误检
        ]
        kept = filter_fn(detections, 640, 480)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["score"], 0.97)

    def test_predict_nut_returns_nut_summary(self):
        self.assertTrue(hasattr(webapp, "_draw_annotated_nuts"))
        fake = [
            {"label": "nut", "zh": "坚果", "score": 0.97, "box": [158, 86, 189, 149]},
            {"label": "nut", "zh": "坚果", "score": 0.84, "box": [251, 184, 272, 229]},
        ]
        orig = webapp._draw_annotated_nuts
        webapp._draw_annotated_nuts = lambda path, conf: ("data:image/png;base64,AAA", fake, 0.25)
        try:
            png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            data = {"image": (io.BytesIO(png), "test.png"), "mode": "nuts"}
            resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload["mode"], "nuts")
            self.assertIn("坚果", payload["headline"])
            self.assertEqual(payload["nut_count"], 2)
        finally:
            webapp._draw_annotated_nuts = orig

    def test_predict_nut_checks_weights(self):
        png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        data = {"image": (io.BytesIO(png), "test.png"), "mode": "nuts"}
        with patch.object(webapp, "NUT_WEIGHTS", "missing-nuts-xyz.pt"):
            resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 500)
        self.assertIn("权重", resp.get_json()["error"])

    def test_world_mode_sets_safe_default_confidence(self):
        resp = self.client.get("/")
        html = resp.get_data(as_text=True)
        self.assertIn("mode === 'world') { conf.value = 25", html)

    def test_world_mode_warns_about_low_confidence_false_matches(self):
        resp = self.client.get("/")
        html = resp.get_data(as_text=True)
        self.assertIn("误认", html)
        self.assertIn("乱框", html)

    def test_predict_world_returns_object_list(self):
        from yolo_starter import objects_vocab as ov
        fake_detections = [
            {"label": "bread", "score": 0.39, "box": [0, 0, 10, 10]},
            {"label": "bread", "score": 0.33, "box": [1, 1, 9, 9]},
            {"label": "plate", "score": 0.61, "box": [2, 2, 8, 8]},
        ]
        orig_draw = webapp._draw_annotated_world
        webapp._draw_annotated_world = lambda path, conf, words=None: (
            "data:image/png;base64,AAA", fake_detections)
        try:
            png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            data = {"image": (io.BytesIO(png), "test.png"), "mode": "world", "conf": "0.2"}
            resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload["mode"], "world")
            self.assertIn("data:image/png", payload["image"])
            self.assertEqual(len(payload["objects"]), 2)  # bread x2, plate x1 -> 2 类
            by_label = {o["label"]: o for o in payload["objects"]}
            self.assertEqual(by_label["bread"]["count"], 2)
            self.assertEqual(by_label["bread"]["zh"], "面包")
            self.assertEqual(by_label["plate"]["zh"], "盘子")
            self.assertEqual(payload["total_objects"], 3)
            self.assertIn("面包", payload["headline"])
        finally:
            webapp._draw_annotated_world = orig_draw

    def test_predict_dense_returns_book_summary(self):
        self.assertTrue(hasattr(webapp, "_draw_annotated_dense"))
        fake_detections = [
            {"label": "book", "zh": "书本", "score": 0.5, "box": [0, 0, 5, 5]},
        ]
        orig = webapp._draw_annotated_dense
        webapp._draw_annotated_dense = lambda path, conf: (
            "data:image/png;base64,AAA", fake_detections, 0.15)
        try:
            png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            data = {"image": (io.BytesIO(png), "test.png"), "mode": "dense"}
            resp = self.client.post("/predict", data=data,
                                    content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertEqual(payload["mode"], "dense")
            self.assertIn("密集书本", payload["headline"])
            self.assertEqual(payload["items"][0]["zh"], "书本")
        finally:
            webapp._draw_annotated_dense = orig

    def test_predict_world_checks_weights(self):
        png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        data = {"image": (io.BytesIO(png), "test.png"), "mode": "world"}
        with patch.object(webapp, "WORLD_WEIGHTS", "missing-world-xyz.pt"):
            resp = self.client.post("/predict", data=data,
                                    content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 500)
        self.assertIn("error", resp.get_json())
        self.assertIn("权重", resp.get_json()["error"])

    def test_predict_returns_advice_with_mocked_inference(self):
        fake_detections = [
            {"label": "bottle", "score": 0.9, "box": [0, 0, 10, 10]},
            {"label": "person", "score": 0.8, "box": [1, 1, 9, 9]},
        ]
        original = webapp._draw_annotated
        webapp._draw_annotated = lambda path, conf: ("data:image/png;base64,AAA", fake_detections)
        try:
            png = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            data = {"image": (io.BytesIO(png), "test.png"), "conf": "0.3"}
            resp = self.client.post("/predict", data=data, content_type="multipart/form-data")
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertIn("data:image/png", payload["image"])
            self.assertIn("可回收", payload["headline"])
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(payload["items"][0]["label"], "bottle")
            self.assertEqual(payload["items"][0].get("zh"), "瓶子")
            self.assertIn("人", payload["non_waste"])
            self.assertNotIn("person", payload["non_waste"])
        finally:
            webapp._draw_annotated = original


if __name__ == "__main__":
    unittest.main()
