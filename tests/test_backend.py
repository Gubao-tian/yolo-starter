"""YOLO 后端封装的行为测试：用真实的小型替身对象，不依赖权重文件。

第三方库只在 full 模式被导入；这里通过 model_factory 注入替身，
验证指标抽取、参数传递与文件输出逻辑。
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_starter import yolo_backend as backend


class FakeBox:
    map50 = 0.512
    map = 0.331
    mp = 0.641
    mr = 0.472


class FakeMetrics:
    box = FakeBox()
    speed = {"preprocess": 1.0, "inference": 15.0, "postprocess": 2.4}


class FakePlotResult:
    def __init__(self, payload):
        self.payload = payload

    def plot(self, *args, **kwargs):
        return self.payload


class FakeModel:
    def __init__(self, plots=None):
        self.val_kwargs = None
        self.train_kwargs = None
        self.plots = plots or []

    def val(self, **kwargs):
        self.val_kwargs = kwargs
        return FakeMetrics()

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return Path(kwargs["project"]) / kwargs["name"]

    def predict(self, **kwargs):
        return self.plots


SPEC = {
    "id": "E1", "name": "基准", "model": "yolov8n.pt", "imgsz": 320,
    "conf": 0.25, "iou": 0.7, "hypothesis": "基线",
}


class TestBackendStatus(unittest.TestCase):
    def test_backend_status_reports_availability_booleans(self):
        status = backend.backend_status()
        self.assertIn("ultralytics_available", status)
        self.assertIsInstance(status["ultralytics_available"], bool)
        self.assertIsInstance(status["torch_available"], bool)


class TestRunValidation(unittest.TestCase):
    def test_validation_extracts_six_metrics_from_ultralytics_style_object(self):
        raw = backend.run_validation(SPEC, "datasets/campus_trash/data.yaml", model_factory=lambda w: FakeModel())
        for key in ("map50", "map5095", "precision", "recall", "ms_per_image", "params_m"):
            self.assertIn(key, raw)
        self.assertAlmostEqual(raw["map50"], 0.512)
        self.assertAlmostEqual(raw["map5095"], 0.331)
        self.assertAlmostEqual(raw["precision"], 0.641)
        self.assertAlmostEqual(raw["recall"], 0.472)

    def test_validation_speed_excludes_loss_component(self):
        raw = backend.run_validation(SPEC, "datasets/campus_trash/data.yaml", model_factory=lambda w: FakeModel())
        self.assertAlmostEqual(raw["ms_per_image"], 18.4)

    def test_validation_passes_controlled_parameters_to_model(self):
        model = FakeModel()
        backend.run_validation(SPEC, "datasets/campus_trash/data.yaml", device="cpu", model_factory=lambda w: model)
        self.assertEqual(model.val_kwargs["imgsz"], 320)
        self.assertAlmostEqual(model.val_kwargs["conf"], 0.25)
        self.assertEqual(model.val_kwargs["data"], "datasets/campus_trash/data.yaml")
        self.assertEqual(model.val_kwargs["device"], "cpu")

    def test_validation_uses_known_parameter_count_when_model_has_no_torch_module(self):
        raw = backend.run_validation(SPEC, "datasets/campus_trash/data.yaml", model_factory=lambda w: FakeModel())
        self.assertGreater(raw["params_m"], 0)


class TestTraining(unittest.TestCase):
    def test_training_returns_save_directory_and_passes_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel()
            saved = backend.run_training(SPEC, "datasets/campus_trash/data.yaml", epochs=1,
                                         project=tmp, model_factory=lambda w: model)
            self.assertIn("E1", str(saved))
            self.assertEqual(model.train_kwargs["epochs"], 1)


class TestInference(unittest.TestCase):
    def test_inference_writes_one_image_per_result(self):
        png = b"\x89PNG\r\n\x1a\n" + b"fake-bytes"
        model = FakeModel(plots=[FakePlotResult(png), FakePlotResult(png)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "detections"
            written = backend.run_inference(SPEC, [Path("a.jpg"), Path("b.jpg")],
                                            out, model_factory=lambda w: model)
            self.assertEqual(len(written), 2)
            self.assertTrue(all(Path(p).exists() for p in written))

    def test_inference_skips_results_without_plot_payload(self):
        model = FakeModel(plots=[FakePlotResult(None)])
        with tempfile.TemporaryDirectory() as tmp:
            written = backend.run_inference(SPEC, [Path("a.jpg")], Path(tmp) / "d",
                                            model_factory=lambda w: model)
            self.assertEqual(written, [])


class TestCollectImagePaths(unittest.TestCase):
    def test_collect_image_paths_finds_nested_images_sorted_and_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            for name in ("c.jpg", "a.jpg", "b.png"):
                (root / "images" / "train" / name).write_bytes(b"x")
            (root / "images" / "train" / "note.txt").write_text("x", encoding="utf-8")
            found = backend.collect_image_paths(root, limit=2)
            self.assertEqual(len(found), 2)
            self.assertEqual([p.name for p in found], ["a.jpg", "b.png"])

    def test_collect_image_paths_returns_empty_list_for_no_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(backend.collect_image_paths(Path(tmp)), [])


class FakeDetBox:
    cls = [0]
    conf = [0.9]
    xyxy = [[10, 20, 110, 220]]


class FakeDetResult:
    names = {0: "plastic"}
    orig_shape = (240, 320)
    boxes = [FakeDetBox()]


class FakeDetectionModel:
    def __init__(self):
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [FakeDetResult()]


class TestCollectDetections(unittest.TestCase):
    def test_collect_detections_extracts_label_score_and_normalized_box(self):
        model = FakeDetectionModel()
        records = backend.collect_detections(
            SPEC, [Path("a.jpg")], "cpu", model_factory=lambda w: model
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["label"], "plastic")
        self.assertAlmostEqual(record["score"], 0.9)
        self.assertAlmostEqual(record["x"], 10 / 320)
        self.assertAlmostEqual(record["y"], 20 / 240)
        self.assertAlmostEqual(record["w"], 100 / 320)
        self.assertAlmostEqual(record["h"], 200 / 240)
        self.assertEqual(model.predict_kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
