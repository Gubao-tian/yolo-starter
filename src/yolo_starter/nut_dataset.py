"""把人工标注的坚果框转成 YOLO 训练数据集。

复用 book_dataset 的增强与坐标变换，仅把类别名改为 nut。
只负责数据与标注变换，不依赖 ultralytics；训练由 ultralytics 负责。
"""
from __future__ import annotations

import json
from pathlib import Path

from yolo_starter import book_dataset


def load_annotations(path: str | Path) -> dict:
    """读取标注 JSON，返回含 class/boxes/source 的字典。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_yolo_label(boxes, width: int, height: int, class_id: int = 0) -> list[str]:
    """转成 YOLO 标签行：class xc yc w h（坐标归一化到 0~1）。"""
    return book_dataset.to_yolo_label(boxes, width, height, class_id)


def build(sources: list[tuple[str | Path, list[list[float]]]],
          out_dir: str | Path, n_crops: int = 6, seed: int = 7) -> dict:
    """从 (图片路径, 框列表) 生成 train/val 数据集，类别为 nut。"""
    # 转成 book_dataset 期望的 detections 形态（带 score=1.0）
    pairs = [(p, [{"box": b, "score": 1.0} for b in boxes]) for p, boxes in sources]
    stats = book_dataset.build_dataset_multi(pairs, out_dir, n_crops=n_crops,
                                             min_score=0.5, seed=seed)
    yaml_path = Path(out_dir) / "data.yaml"
    yaml_path.write_text(
        f'path: "{Path(out_dir).as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [nut]\n",
        encoding="utf-8")
    stats["yaml"] = str(yaml_path)
    return stats
