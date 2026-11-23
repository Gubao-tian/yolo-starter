"""YOLO 推理后端：加载权重、对图片推理、返回结构化检测结果。

把 ultralytics 的调用集中在这里，网页层只依赖纯数据结构，便于测试。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from yolo_starter import objects_vocab

DEFAULT_WEIGHTS = "weights/yolov8n.pt"
DEFAULT_WORLD_WEIGHTS = "weights/yolov8s-world.pt"


@lru_cache(maxsize=2)
def load_world_model(weights_path: str = DEFAULT_WORLD_WEIGHTS,
                     device: str = "cpu"):
    """加载 YOLO-World 开放词汇模型并缓存（进程内只加载一次）。"""
    from ultralytics import YOLO

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"权重文件不存在: {path}")
    model = YOLO(str(path))
    if device:
        try:
            model.to(device)
        except Exception:
            pass
    return model


def predict_world(image_path: str | Path,
                  words: list[str] | None = None,
                  weights_path: str = DEFAULT_WORLD_WEIGHTS,
                  conf: float = 0.25, device: str = "cpu") -> dict:
    """用 YOLO-World 按给定词表检测图片中的所有对象。

    返回 {detections, count, mode, words}。模型进程内缓存，词表每次设置
    （set_classes 约数秒，推理约 0.3 秒 CPU），保证词表与请求一致。
    """
    word_list = list(words) if words else list(objects_vocab.WORDS)
    model = load_world_model(str(weights_path), device)
    model.set_classes(word_list)
    results = model.predict(source=str(image_path), conf=conf, device=device, verbose=False)
    detections: list[dict] = []
    for r in results:
        detections.extend(predictions_from_result(r))
    detections.sort(key=lambda d: -d["score"])
    return {"detections": detections, "count": len(detections),
            "mode": "world", "words": word_list}


@lru_cache(maxsize=4)
def load_model(weights_path: str, device: str = "cpu"):
    """加载 YOLO 模型并缓存，避免每次上传都重新读权重。"""
    from ultralytics import YOLO

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"权重文件不存在: {path}")
    model = YOLO(str(path))
    if device:
        try:
            model.to(device)
        except Exception:
            pass
    return model


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def predictions_from_result(result) -> list[dict]:
    """把一个 ultralytics Results 对象转成 [{label, score, box}, ...]。"""
    out: list[dict] = []
    names = getattr(result, "names", None) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out
    for b in boxes:
        try:
            cls_id = int(b.cls[0])
            conf = _to_float(b.conf[0])
            xyxy = [round(float(v), 1) for v in b.xyxy[0]]
        except (IndexError, TypeError, ValueError):
            continue
        label = names.get(cls_id, str(cls_id))
        out.append({"label": label, "zh": objects_vocab.zh_name(label),
                    "score": round(conf, 3), "box": xyxy})
    return out


def label_text(detection: dict) -> str:
    """把一条检测结果显示为『中文名 置信度』。"""
    label = str(detection.get("label", "") or "")
    zh = detection.get("zh") or objects_vocab.zh_name(label)
    try:
        score = float(detection.get("score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0
    return f"{zh} {score:.2f}"


def _load_cjk_font(size: int):
    """优先加载系统中文字体，失败时退回 PIL 默认字体。"""
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def top_label_indices(detections: list[dict], max_labels: int | None) -> set[int]:
    """目标很多时只给分数最高的若干框写标签，避免文字糊成一片。"""
    if max_labels is None or len(detections) <= max_labels:
        return set(range(len(detections)))

    def _score(d: dict) -> float:
        try:
            return float(d.get("score", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(((_score(d), i) for i, d in enumerate(detections)),
                    key=lambda p: p[0], reverse=True)
    return {i for _, i in ranked[:max_labels]}


def draw_detections(image, detections: list[dict], max_labels: int | None = None):
    """在图片上画出检测框与中文标签，返回新的 RGB PIL 图片。"""
    from PIL import Image, ImageDraw

    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_cjk_font(16)
    palette = [(239, 68, 68), (37, 99, 235), (22, 163, 74),
               (234, 88, 12), (124, 58, 237)]
    labeled = top_label_indices(detections, max_labels)
    for i, det in enumerate(detections):
        box = det.get("box") or []
        if len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        color = palette[i % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if i not in labeled:
            continue
        text = label_text(det)
        try:
            left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        except AttributeError:
            left, top, right, bottom = x1, y1, x1 + len(text) * 12, y1 + 18
        draw.rectangle([left, top, right, bottom], fill=color)
        draw.text((left, top), text, fill="white", font=font)
    return img


def generate_tiles(width: int, height: int, tile: int = 200,
                   overlap: int = 40) -> list[tuple[int, int, int, int]]:
    """把大图切成带重叠的小块，返回 [(x, y, w, h), ...]。"""
    if width <= 0 or height <= 0 or tile <= 0:
        return []
    stride = max(1, tile - max(0, overlap))
    out: list[tuple[int, int, int, int]] = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            w, h = min(tile, width - x), min(tile, height - y)
            if w >= 10 and h >= 10:
                out.append((x, y, w, h))
    return out


def nms_detections(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """同类框的非极大值抑制：重叠超过阈值的低分框被丢弃。"""
    def _iou(a: list[float], b: list[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _score(d: dict) -> float:
        try:
            return float(d.get("score", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(detections, key=_score, reverse=True)
    kept: list[dict] = []
    for det in ordered:
        box = det.get("box") or []
        if len(box) != 4:
            continue
        if any(k.get("label") == det.get("label")
               and _iou(box, k.get("box") or [0, 0, 0, 0]) >= iou_threshold
               for k in kept):
            continue
        kept.append(det)
    return kept


def detect_dense_books(model, image_path: str | Path, conf: float = 0.15,
                       imgsz: int = 640, tile: int = 200, overlap: int = 40,
                       iou: float = 0.3, nms_iou: float = 0.5) -> list[dict]:
    """密集书架增强：切片推理 + 只找 book 类 + 坐标回映射 + NMS。

    密集小目标在整图缩放到 640 后只剩十几像素，切成小块再放大能显著
    提高召回；只检测 book 可避免其他类别干扰。返回结构化检测列表。
    """
    from PIL import Image

    names = getattr(model, "names", None) or {}
    book_id = next((i for i, n in names.items()
                    if str(n).strip().lower() == "book"), None)
    if book_id is None:
        return []

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    out: list[dict] = []
    for x, y, w, h in generate_tiles(width, height, tile, overlap):
        crop = image.crop((x, y, x + w, y + h))
        results = model.predict(source=crop, imgsz=imgsz, conf=conf, iou=iou,
                                classes=[book_id], verbose=False)
        for r in results:
            for det in predictions_from_result(r):
                box = det.get("box") or []
                if len(box) != 4:
                    continue
                shifted = dict(det)
                shifted["box"] = [box[0] + x, box[1] + y,
                                  box[2] + x, box[3] + y]
                out.append(shifted)
    return nms_detections(out, nms_iou)


def predict(image_path: str | Path, weights_path: str = DEFAULT_WEIGHTS,
            conf: float = 0.25, device: str = "cpu") -> dict:
    """对一张图片推理，返回结构化结果（不含画框后的图片）。"""
    model = load_model(str(weights_path), device)
    results = model.predict(source=str(image_path), conf=conf, device=device, verbose=False)
    detections: list[dict] = []
    for r in results:
        detections.extend(predictions_from_result(r))
    detections.sort(key=lambda d: -d["score"])
    return {"detections": detections, "count": len(detections)}
