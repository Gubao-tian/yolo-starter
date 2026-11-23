"""把密集书本的伪标签转成 YOLO 训练数据集。

只负责数据与标注变换，不依赖 ultralytics；训练由 ultralytics 负责。
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


def _as_list(box) -> list[float]:
    return [float(v) for v in box]


def mirror_boxes(boxes, width: int) -> list[list[float]]:
    """水平翻转图片时同步翻转 x 坐标。"""
    w = float(width)
    return [[round(w - b[2], 1), round(b[1], 1),
             round(w - b[0], 1), round(b[3], 1)]
            for b in (_as_list(x) for x in boxes)]


def crop_boxes(boxes, crop: tuple[int, int, int, int]) -> list[list[float]]:
    """裁剪图片后把框裁剪到裁剪区内，丢弃几乎全在区外的框。"""
    cx, cy, cw, ch = crop
    out: list[list[float]] = []
    for b in (_as_list(x) for x in boxes):
        nx1, ny1 = max(b[0] - cx, 0.0), max(b[1] - cy, 0.0)
        nx2, ny2 = min(b[2] - cx, float(cw)), min(b[3] - cy, float(ch))
        if nx2 - nx1 >= 2.0 and ny2 - ny1 >= 2.0:
            out.append([round(nx1, 1), round(ny1, 1),
                        round(nx2, 1), round(ny2, 1)])
    return out


def to_yolo_label(boxes, width: int, height: int,
                  class_id: int = 0) -> list[str]:
    """转成 YOLO 标签行：class xc yc w h（坐标归一化到 0~1）。"""
    lines: list[str] = []
    for b in (_as_list(x) for x in boxes):
        xc = (b[0] + b[2]) / 2 / width
        yc = (b[1] + b[3]) / 2 / height
        bw = (b[2] - b[0]) / width
        bh = (b[3] - b[1]) / height
        lines.append(f"{class_id} {round(xc, 4)} {round(yc, 4)} "
                     f"{round(bw, 4)} {round(bh, 4)}")
    return lines


def _variants(image: Image.Image, boxes: list[list[float]],
              n_crops: int, seed: int):
    """产生 (图片, 框列表) 变体：原图、镜像、亮度/对比度、随机裁剪。"""
    rng = random.Random(seed)
    w, h = image.size
    yield image, boxes
    yield ImageOps.mirror(image), mirror_boxes(boxes, w)
    yield ImageEnhance.Brightness(image).enhance(0.7), boxes
    yield ImageEnhance.Brightness(image).enhance(1.3), boxes
    yield ImageEnhance.Contrast(image).enhance(1.4), boxes

    for _ in range(n_crops):
        ratio = rng.uniform(0.78, 0.98)
        cw, ch = max(40, int(w * ratio)), max(40, int(h * ratio))
        cx = rng.randint(0, w - cw)
        cy = rng.randint(0, h - ch)
        crop = (cx, cy, cw, ch)
        kept = crop_boxes(boxes, crop)
        if len(kept) < 3:
            continue
        piece = image.crop((cx, cy, cx + cw, cy + ch))
        yield piece, kept
        yield ImageOps.mirror(piece), mirror_boxes(kept, cw)


def build_dataset_multi(sources: list[tuple[str | Path, list[dict]]],
                        out_dir: str | Path, n_crops: int = 8,
                        min_score: float = 0.20, seed: int = 7) -> dict:
    """用多张图的伪标签生成 train/val 数据集；空检测图即负样本。"""
    target = Path(out_dir)
    for split in ("train", "val"):
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "boxes": 0, "sources": 0,
              "negative_sources": 0}
    for source_index, (source_image, detections) in enumerate(sources):
        boxes: list[list[float]] = []
        for d in detections:
            try:
                score = float(d.get("score", 0) or 0)
            except (TypeError, ValueError):
                continue
            box = d.get("box") or []
            if score >= min_score and len(box) == 4:
                boxes.append([float(v) for v in box])
        if not boxes:
            counts["negative_sources"] += 1

        image = Image.open(source_image).convert("RGB")
        stem = f"{Path(source_image).stem}_{source_index:02d}"
        counts["sources"] += 1
        for index, (variant, variant_boxes) in enumerate(
                _variants(image, boxes, n_crops, seed + source_index)):
            split = "val" if index % 5 == 0 else "train"
            name = f"{stem}_{index:03d}"
            variant.save(target / "images" / split / f"{name}.jpg", quality=92)
            lines = to_yolo_label(variant_boxes, variant.width, variant.height)
            (target / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[split] += 1
            counts["boxes"] += len(lines)

    yaml_path = target / "data.yaml"
    yaml_path.write_text(
        f'path: "{target.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [book]\n",
        encoding="utf-8")
    counts["yaml"] = str(yaml_path)
    return counts


def build_dataset(source_image: str | Path, detections: list[dict],
                  out_dir: str | Path, n_crops: int = 8,
                  min_score: float = 0.20, seed: int = 7) -> dict:
    """单图版本的兼容入口。"""
    return build_dataset_multi([(source_image, detections)], out_dir,
                               n_crops, min_score, seed)
