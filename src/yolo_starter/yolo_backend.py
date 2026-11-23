"""Ultralytics YOLOv8 封装：验证、训练、推理。

关键约束：ultralytics / torch / PIL / numpy 一律延迟导入，
只有 full 模式才会触发，因此 demo 模式可以在裸 Python 上运行。

model_factory 参数用于注入模型来源（测试替身或自定义加载逻辑），
使指标抽取逻辑无需真实权重即可被验证。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

# 官方权重的公开参数量（百万），仅在无法从模型对象读取时作为兜底。
KNOWN_PARAMS_M: dict[str, float] = {
    "yolov8n.pt": 3.2,
    "yolov8s.pt": 11.2,
    "yolov8m.pt": 25.9,
    "yolov8l.pt": 43.7,
    "yolov8x.pt": 68.2,
    "yolo11n.pt": 2.6,
    "yolo11s.pt": 9.4,
}


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def backend_status() -> dict:
    """返回依赖可用性，不导入重量级库。"""
    status = {
        "ultralytics_available": _has("ultralytics"),
        "torch_available": _has("torch"),
        "pillow_available": _has("PIL"),
        "ultralytics_version": None,
    }
    if status["ultralytics_available"]:
        try:
            from importlib.metadata import version
            status["ultralytics_version"] = version("ultralytics")
        except Exception:
            pass
    return status


def _default_model_factory(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def _factory(model_factory: Callable[[str], Any] | None) -> Callable[[str], Any]:
    return model_factory or _default_model_factory


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _model_params_m(model: Any) -> float:
    """优先从 torch 模块统计参数量；不可用时返回 0，由调用方回退到查表。"""
    module = getattr(model, "model", None)
    if module is not None and hasattr(module, "parameters"):
        try:
            return sum(p.numel() for p in module.parameters()) / 1_000_000.0
        except Exception:
            pass
    return 0.0


def _timed_predict_ms(model: Any, spec: dict, device: str) -> float:
    """当 val() 未返回 speed 时，用合成图实测单图推理耗时作为兜底。"""
    try:
        import time
        import numpy as np
        dummy = np.zeros((int(spec["imgsz"]), int(spec["imgsz"]), 3), dtype=np.uint8)
        kwargs = dict(imgsz=spec["imgsz"], conf=spec["conf"], device=device, verbose=False)
        model.predict(dummy, **kwargs)  # 预热
        start = time.perf_counter()
        for _ in range(3):
            model.predict(dummy, **kwargs)
        return (time.perf_counter() - start) * 1000.0 / 3.0
    except Exception:
        return 0.0


def run_validation(spec: dict, dataset: str, device: str = "cpu",
                   model_factory: Callable[[str], Any] | None = None) -> dict[str, float]:
    """在给定数据集上验证一组实验参数，返回六个统一口径的指标。"""
    model = _factory(model_factory)(spec["model"])
    metrics_obj = model.val(
        data=dataset,
        imgsz=spec["imgsz"],
        conf=spec["conf"],
        iou=spec["iou"],
        device=device,
        verbose=False,
        plots=False,
    )
    box = getattr(metrics_obj, "box", None)
    speed = getattr(metrics_obj, "speed", {}) or {}
    ms = sum(_number(v) for k, v in speed.items() if k != "loss") if hasattr(speed, "items") else 0.0
    if ms <= 0:
        ms = _timed_predict_ms(model, spec, device)
    params = _model_params_m(model) or KNOWN_PARAMS_M.get(Path(str(spec["model"])).name, 0.0)
    return {
        "map50": _number(getattr(box, "map50", 0)),
        "map5095": _number(getattr(box, "map", 0)),
        "precision": _number(getattr(box, "mp", 0)),
        "recall": _number(getattr(box, "mr", 0)),
        "ms_per_image": ms,
        "params_m": params,
    }


def run_training(spec: dict, dataset: str, epochs: int = 1, device: str = "cpu",
                 project: str | Path = "runs",
                 model_factory: Callable[[str], Any] | None = None) -> str:
    """可选训练步骤；返回本次训练的保存目录。"""
    model = _factory(model_factory)(spec["model"])
    name = f"exp_{spec['id']}"
    result = model.train(
        data=dataset,
        epochs=epochs,
        imgsz=spec["imgsz"],
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    return str(result) if result is not None else str(Path(project) / name)


def _save_plot(plotted: Any, path: Path) -> bool:
    """兼容 bytes / PIL.Image / numpy(BGR) 三种 plot() 返回值。"""
    try:
        if isinstance(plotted, (bytes, bytearray)):
            path.write_bytes(bytes(plotted))
            return True
        from PIL import Image
        if hasattr(plotted, "convert"):
            plotted.convert("RGB").save(path)
            return True
        import numpy as np
        array = np.asarray(plotted)
        if array.ndim == 3 and array.shape[2] == 3:
            array = array[..., ::-1]  # OpenCV BGR -> RGB
        Image.fromarray(array).save(path)
        return True
    except Exception:
        return False


def run_inference(spec: dict, image_paths: list[Path], out_dir: str | Path,
                  device: str = "cpu",
                  model_factory: Callable[[str], Any] | None = None) -> list[str]:
    """对若干图片做推理并把画框结果保存为 PNG，返回写出的文件路径。"""
    if not image_paths:
        return []
    model = _factory(model_factory)(spec["model"])
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=spec["conf"],
        imgsz=spec["imgsz"],
        device=device,
        verbose=False,
        save=False,
    )
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for index, item in enumerate(results or []):
        plotted = item.plot() if hasattr(item, "plot") else None
        if plotted is None:
            continue
        path = target / f"det_{index:02d}.png"
        if _save_plot(plotted, path):
            written.append(str(path))
    return written


def collect_detections(spec: dict, image_paths: list[Path], device: str = "cpu",
                       model_factory: Callable[[str], Any] | None = None) -> list[dict]:
    """从真实推理结果中抽取类别、置信度与归一化检测框。"""
    if not image_paths:
        return []
    model = _factory(model_factory)(spec["model"])
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=spec["conf"],
        imgsz=spec["imgsz"],
        device=device,
        verbose=False,
        save=False,
    )
    records: list[dict] = []
    for result in results or []:
        names = getattr(result, "names", {}) or {}
        shape = getattr(result, "orig_shape", None)
        if shape and len(shape) >= 2:
            height, width = float(shape[0]), float(shape[1])
        else:
            height = width = 1.0
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            records.append({
                "label": str(names.get(class_id, class_id)),
                "score": score,
                "x": x1 / width,
                "y": y1 / height,
                "w": (x2 - x1) / width,
                "h": (y2 - y1) / height,
                "color": "#2563eb",
            })
    return records


def collect_image_paths(root: str | Path, limit: int = 3) -> list[Path]:
    """在目录树中按路径排序查找图片，返回前 limit 个。"""
    directory = Path(root)
    if not directory.exists():
        return []
    found = sorted(
        (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: str(p).lower(),
    )
    return found[:max(0, limit)]
