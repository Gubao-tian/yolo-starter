"""拍照识垃圾：本地网页上传识别服务。

启动： .venv\\Scripts\\python.exe webapp\\app.py
打开： http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for p in (str(SRC), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask, jsonify, render_template, request  # noqa: E402

from yolo_starter import detector, objects_vocab, waste_map  # noqa: E402

app = Flask(__name__, template_folder="templates", static_folder="static")

WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(PROJECT_ROOT / "weights" / "yolov8n.pt"))
WORLD_WEIGHTS = os.environ.get("YOLO_WORLD_WEIGHTS",
                               str(PROJECT_ROOT / "weights" / "yolov8s-world.pt"))
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")
NUT_WEIGHTS = os.environ.get("YOLO_NUT_WEIGHTS",
                             str(PROJECT_ROOT / "weights" / "yolov8n-nuts.pt"))
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 坚果是小目标：框面积超过整图这个比例即视为误检大框，过滤掉。
NUT_MAX_AREA_RATIO = 0.04


def filter_nut_boxes(detections: list[dict], width: int, height: int) -> list[dict]:
    """过滤坚果误检：只保留面积占比不超过阈值的框（坚果都是小目标）。"""
    max_area = width * height * NUT_MAX_AREA_RATIO
    kept: list[dict] = []
    for d in detections:
        box = d.get("box") or []
        if len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        area = abs(x2 - x1) * abs(y2 - y1)
        if area <= max_area:
            kept.append(d)
    return kept


def _draw_annotated_world(image_path: str, conf: float, words: list[str] | None = None):
    """YOLO-World 全物体检测并画框，返回 (data URI, detections)。"""
    from PIL import Image

    if not Path(WORLD_WEIGHTS).exists():
        raise FileNotFoundError(
            f"全物体模式权重不存在：{WORLD_WEIGHTS}。请将 yolov8s-world.pt 放入 weights/ 目录。")
    result = detector.predict_world(image_path, words=words, weights_path=WORLD_WEIGHTS,
                                    conf=conf, device=DEVICE)
    source = Image.open(image_path).convert("RGB")
    annotated = detector.draw_detections(source, result["detections"])
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, result["detections"]


def _draw_annotated(image_path: str, conf: float):
    """用模型推理并画出检测框，返回 (带框图片的 base64 data URI, detections)。"""
    from PIL import Image

    model = detector.load_model(WEIGHTS, DEVICE)
    results = model.predict(source=image_path, conf=conf, device=DEVICE, verbose=False)
    detections: list[dict] = []
    for r in results:
        detections.extend(detector.predictions_from_result(r))
    detections.sort(key=lambda d: -d["score"])

    source = Image.open(image_path).convert("RGB")
    annotated = detector.draw_detections(source, detections)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, detections


def _draw_annotated_dense(image_path: str, conf: float):
    """密集书本模式：切片推理 + 中文画框，返回 (data URI, detections, 实际阈值)。"""
    from PIL import Image

    used_conf = conf  # 前端切换到密集模式时会把滑块设为 0.15，用户仍可自行调高/调低
    model = detector.load_model(WEIGHTS, DEVICE)
    detections = detector.detect_dense_books(model, image_path, conf=used_conf)
    source = Image.open(image_path).convert("RGB")
    annotated = detector.draw_detections(source, detections, max_labels=12)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, detections, used_conf


def _draw_annotated_nuts(image_path: str, conf: float):
    """坚果识别：用微调权重推理，过滤误检大框后画框。返回 (data URI, detections, 阈值)。"""
    from PIL import Image

    if not Path(NUT_WEIGHTS).exists():
        raise FileNotFoundError(
            f"坚果模式权重不存在：{NUT_WEIGHTS}。请先运行 scripts/train_nuts.py 训练。")
    model = detector.load_model(NUT_WEIGHTS, DEVICE)
    results = model.predict(source=image_path, conf=conf, device=DEVICE, verbose=False)
    detections: list[dict] = []
    for r in results:
        detections.extend(detector.predictions_from_result(r))
    source = Image.open(image_path).convert("RGB")
    detections = filter_nut_boxes(detections, source.width, source.height)
    detections.sort(key=lambda d: -d["score"])
    annotated = detector.draw_detections(source, detections)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, detections, conf


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "weights": WEIGHTS, "weights_exists": Path(WEIGHTS).exists()})


@app.route("/samples")
def samples():
    """返回精选测试图列表，供页面一键加载。"""
    folder = PROJECT_ROOT / "test_images" / "selected"
    items = []
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in ALLOWED_EXT:
                items.append({"name": p.stem, "url": "/sample_image/" + p.name})
    return jsonify({"items": items})


@app.route("/sample_image/<path:name>")
def sample_image(name: str):
    """安全地提供精选测试图（仅允许 selected 目录内的文件名）。"""
    if Path(name).name != name:  # 防目录穿越
        return "forbidden", 403
    folder = PROJECT_ROOT / "test_images" / "selected"
    target = folder / name
    if not target.is_file() or target.suffix.lower() not in ALLOWED_EXT:
        return "not found", 404
    from flask import send_file
    return send_file(str(target), mimetype="image/jpeg")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "没有收到图片"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"不支持的图片格式：{ext}"}), 400
    try:
        conf = float(request.form.get("conf", "0.25"))
    except ValueError:
        conf = 0.25
    conf = min(0.9, max(0.05, conf))
    mode = request.form.get("mode", "waste")
    if mode not in ("waste", "world", "dense", "nuts"):
        mode = "waste"

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / ("upload" + ext)
        file.save(str(img_path))
        try:
            if mode == "world":
                uri, detections = _draw_annotated_world(str(img_path), conf)
            elif mode == "dense":
                uri, detections, used_conf = _draw_annotated_dense(str(img_path), conf)
            elif mode == "nuts":
                uri, detections, used_conf = _draw_annotated_nuts(str(img_path), conf)
            else:
                uri, detections = _draw_annotated(str(img_path), conf)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            return jsonify({"error": f"推理失败：{exc}"}), 500

    if mode == "world":
        objects = objects_vocab.summarize_objects(detections)
        total = sum(o["count"] for o in objects)
        names = "、".join(f"{o['zh']}×{o['count']}" for o in objects[:6]) if objects else "无"
        headline = f"全物体提取：共发现 {total} 个物品，{len(objects)} 类（{names}{'…' if len(objects) > 6 else ''}）。"
        return jsonify({
            "mode": "world", "image": uri, "detections": detections,
            "headline": headline, "objects": objects, "total_objects": total,
            "vocab_size": len(objects_vocab.WORDS),
        })

    if mode == "dense":
        summary = waste_map.summarize(detections)
        headline = (f"密集书本增强：检测到 {summary['waste_count']} 本书"
                    f"（切片推理，置信度阈值 {used_conf:.2f}）。")
        return jsonify({
            "mode": "dense", "image": uri, "detections": detections,
            "headline": headline, "items": summary["items"], "non_waste": "",
        })

    if mode == "nuts":
        nut_count = len(detections)
        headline = (f"坚果识别（微调模型）：检测到 {nut_count} 颗坚果"
                    f"（阈值 {used_conf:.2f}，已过滤误检大框）。")
        return jsonify({
            "mode": "nuts", "image": uri, "detections": detections,
            "headline": headline, "nut_count": nut_count,
        })

    summary = waste_map.summarize(detections)
    non_waste = []
    for d in detections:
        label = str(d.get("label", "") or "")
        if waste_map.waste_advice(label)["is_waste"]:
            continue
        zh = d.get("zh") or objects_vocab.zh_name(label)
        if zh not in non_waste:
            non_waste.append(zh)
    non_waste_text = ("另识别到非垃圾对象：" + "、".join(non_waste) + "（不产生投放建议）") \
        if non_waste else ""

    return jsonify({
        "mode": "waste",
        "image": uri,
        "detections": detections,
        "headline": summary["headline"],
        "items": summary["items"],
        "non_waste": non_waste_text,
    })


def main():
    port = int(os.environ.get("PORT", "5000"))
    print(f"服务已启动：http://127.0.0.1:{port}")
    print(f"权重文件：{WEIGHTS}（存在：{Path(WEIGHTS).exists()}）")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
