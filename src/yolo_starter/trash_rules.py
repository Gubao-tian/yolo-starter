"""校园垃圾分类规则：把检测类别映射到投放建议。

这里刻意用纯规则而不是模型：YOLO 只负责"是什么、在哪里"，
"投哪个桶"由可解释的映射表给出，便于课堂讲清楚分工。
"""
from __future__ import annotations

from collections import Counter

CANONICAL_CLASSES = (
    "plastic", "paper", "cardboard", "glass", "metal", "trash", "battery", "organic"
)
RECYCLABLE = frozenset({"plastic", "paper", "cardboard", "glass", "metal"})

ALIASES = {
    "plastic bottle": "plastic", "bottle": "plastic", "塑料瓶": "plastic", "塑料": "plastic",
    "paper": "paper", "纸张": "paper", "纸": "paper",
    "cardboard": "cardboard", "纸箱": "cardboard", "纸板": "cardboard",
    "glass": "glass", "玻璃": "glass",
    "metal": "metal", "金属": "metal", "易拉罐": "metal", "can": "metal",
    "trash": "trash", "other": "trash", "其他垃圾": "trash",
    "battery": "battery", "电池": "battery",
    "organic": "organic", "food": "organic", "厨余": "organic", "果皮": "organic",
}

ADVICE = {
    "plastic": "可回收物：清空残留并简单冲洗，投入蓝色可回收桶。",
    "paper": "可回收物：保持干燥整洁，投入蓝色可回收桶。",
    "cardboard": "可回收物：拆开压平后投入蓝色可回收桶。",
    "glass": "可回收物：轻放投入玻璃回收箱，避免破碎。",
    "metal": "可回收物：清空内容物后投入蓝色可回收桶。",
    "battery": "有害垃圾：单独包装后投入红色有害垃圾桶。",
    "organic": "厨余垃圾：去除包装后投入绿色厨余桶。",
    "trash": "其他垃圾：无法回收时投入灰色其他垃圾桶。",
}


def normalize_label(label: object) -> str:
    """把中英文别名规整为标准类别；未知类别按其他垃圾处理。"""
    key = str(label or "").strip().lower()
    if key in CANONICAL_CLASSES:
        return key
    return ALIASES.get(key, "trash")


def advice_for(label: object) -> str:
    """返回该类别的投放建议。"""
    return ADVICE[normalize_label(label)]


def summarize(detections: list[dict]) -> dict:
    """汇总检测结果的类别数量、投放建议与一句可讲结论。"""
    counts = Counter(normalize_label(box.get("label", "")) for box in detections)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    items = [
        {"label": label, "count": count, "advice": ADVICE[label]}
        for label, count in ordered
    ]
    total = sum(counts.values())
    recyclable = sum(counts[label] for label in RECYCLABLE)
    headline = (
        f"共检测 {total} 件物品，其中可回收物 {recyclable} 件，"
        f"其余 {total - recyclable} 件需按有害/厨余/其他垃圾分别投放。"
    )
    return {"total": total, "recyclable": recyclable, "items": items, "headline": headline}
