"""demo 模式的示例数据。

重要：这里的数值是**教学用示例值**，不是真实推理结果。
看板会显著标注数据来源，避免被误当作实验结论引用。
"""
from __future__ import annotations

from typing import Iterable

DEMO_RESULTS: dict[str, dict[str, float]] = {
    "E1": {"map50": 0.512, "map5095": 0.331, "precision": 0.641,
           "recall": 0.472, "ms_per_image": 18.4, "params_m": 3.2},
    "E2": {"map50": 0.596, "map5095": 0.412, "precision": 0.688,
           "recall": 0.538, "ms_per_image": 42.7, "params_m": 3.2},
    "E3": {"map50": 0.571, "map5095": 0.399, "precision": 0.762,
           "recall": 0.446, "ms_per_image": 42.7, "params_m": 3.2},
    "E4": {"map50": 0.648, "map5095": 0.468, "precision": 0.713,
           "recall": 0.587, "ms_per_image": 91.3, "params_m": 11.2},
}

DEMO_DETECTIONS: list[dict] = [
    {"label": "plastic", "score": 0.91, "x": 0.08, "y": 0.20, "w": 0.18, "h": 0.56, "color": "#2563eb"},
    {"label": "paper", "score": 0.83, "x": 0.32, "y": 0.30, "w": 0.22, "h": 0.40, "color": "#059669"},
    {"label": "glass", "score": 0.74, "x": 0.60, "y": 0.16, "w": 0.20, "h": 0.62, "color": "#7c3aed"},
    {"label": "cardboard", "score": 0.71, "x": 0.28, "y": 0.65, "w": 0.20, "h": 0.25, "color": "#b45309"},
    {"label": "metal", "score": 0.66, "x": 0.52, "y": 0.62, "w": 0.14, "h": 0.28, "color": "#0891b2"},
    {"label": "trash", "score": 0.68, "x": 0.78, "y": 0.55, "w": 0.16, "h": 0.30, "color": "#dc2626"},
]


def demo_results(ids: Iterable[str]) -> dict[str, dict[str, float]]:
    """返回指定实验编号的示例指标（副本，避免调用方修改常量）。"""
    return {item: dict(DEMO_RESULTS[item]) for item in ids}


def demo_detections() -> list[dict]:
    """返回归一化的示例检测框，坐标均为 0~1 的相对值。"""
    return [dict(box) for box in DEMO_DETECTIONS]


def demo_note() -> str:
    return (
        "演示数据（非真实推理结果）：指标来自 src/yolo_starter/samples.py 的内置示例值，"
        "检测框与投放建议用于展示「校园垃圾分类投放识别」流程，仅用于验证流水线、"
        "展示看板与答辩预演。请用 --mode full 配合 datasets/campus_trash/data.yaml 获取真实实验结果。"
    )
