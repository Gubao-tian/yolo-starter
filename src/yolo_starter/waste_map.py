"""COCO 识别类别 -> 垃圾投放建议。

演示用模型是 COCO 预训练的 yolov8n，识别的是 80 类日常物品（瓶子、
香蕉、手机、人、车……），不是专门的垃圾模型。这张表把其中与垃圾
相关的常见类别翻译成「四类垃圾 + 桶色 + 建议」，让人/动物/车辆等
非垃圾对象不产生投放建议。未知类别一律按其他垃圾兜底。
"""
from __future__ import annotations

from yolo_starter import objects_vocab

# COCO 类别 -> (大类, 桶色, 建议)
_COCO_WASTE: dict[str, tuple[str, str, str]] = {
    # 可回收物（蓝色）
    "bottle": ("可回收物", "蓝色", "塑料瓶：倒空并简单冲洗，投入蓝色可回收桶。"),
    "wine glass": ("可回收物", "蓝色", "玻璃杯：轻放投入玻璃回收箱，避免破碎。"),
    "cup": ("其他垃圾", "灰色", "一次性杯子：多数带覆膜难回收，投入灰色其他垃圾桶。"),
    "book": ("可回收物", "蓝色", "书本：保持干燥整洁，投入蓝色可回收桶。"),
    "newspaper": ("可回收物", "蓝色", "报纸：捆扎整齐后投入蓝色可回收桶。"),
    "paper": ("可回收物", "蓝色", "纸张：保持干燥整洁，投入蓝色可回收桶。"),
    "cardboard": ("可回收物", "蓝色", "纸板：拆开压平后投入蓝色可回收桶。"),
    "metal": ("可回收物", "蓝色", "金属：清空内容物后投入蓝色可回收桶。"),
    "can": ("可回收物", "蓝色", "易拉罐：清空压扁后投入蓝色可回收桶。"),
    "keyboard": ("可回收物", "蓝色", "键盘：属电子废弃物，投放至电子回收点。"),
    "mouse": ("可回收物", "蓝色", "鼠标：属电子废弃物，投放至电子回收点。"),
    # 厨余垃圾（绿色）
    "banana": ("厨余垃圾", "绿色", "香蕉皮：去除标签后投入绿色厨余桶。"),
    "apple": ("厨余垃圾", "绿色", "果核果壳：投入绿色厨余桶。"),
    "orange": ("厨余垃圾", "绿色", "橘皮：投入绿色厨余桶。"),
    "broccoli": ("厨余垃圾", "绿色", "菜叶：投入绿色厨余桶。"),
    "carrot": ("厨余垃圾", "绿色", "菜叶菜根：投入绿色厨余桶。"),
    "sandwich": ("厨余垃圾", "绿色", "剩余食物：去除包装后投入绿色厨余桶。"),
    "hot dog": ("厨余垃圾", "绿色", "剩余食物：去除包装后投入绿色厨余桶。"),
    "pizza": ("厨余垃圾", "绿色", "剩余食物：去除包装后投入绿色厨余桶。"),
    "donut": ("厨余垃圾", "绿色", "剩余食物：去除包装后投入绿色厨余桶。"),
    "cake": ("厨余垃圾", "绿色", "剩余食物：去除包装后投入绿色厨余桶。"),
    "apple core": ("厨余垃圾", "绿色", "果核：投入绿色厨余桶。"),
    # 有害垃圾（红色）
    "cell phone": ("有害垃圾", "红色", "废旧手机：含电池，单独回收，投入红色有害垃圾桶或电子回收点。"),
    "battery": ("有害垃圾", "红色", "电池：单独包装后投入红色有害垃圾桶。"),
    "tv": ("有害垃圾", "红色", "废旧电器：投放至电子废弃物回收点。"),
    "laptop": ("有害垃圾", "红色", "笔记本电脑：含电池，投放至电子回收点。"),
    "microwave": ("有害垃圾", "红色", "废旧电器：投放至电子废弃物回收点。"),
    "toaster": ("有害垃圾", "红色", "废旧电器：投放至电子废弃物回收点。"),
    "refrigerator": ("有害垃圾", "红色", "废旧电器：投放至电子废弃物回收点。"),
    "light bulb": ("有害垃圾", "红色", "灯管灯泡：轻包后投入红色有害垃圾桶。"),
    # 其他垃圾（灰色）
    "trash": ("其他垃圾", "灰色", "其他垃圾：投入灰色其他垃圾桶。"),
    "chair": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "couch": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "bed": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "dining table": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "toothbrush": ("其他垃圾", "灰色", "牙刷：属其他垃圾，投入灰色桶。"),
    "tissue": ("其他垃圾", "灰色", "用过的纸巾：投入灰色其他垃圾桶。"),
    "bowl": ("其他垃圾", "灰色", "碗碟：陶瓷/一次性餐具难回收，投入灰色其他垃圾桶。"),
    "plate": ("其他垃圾", "灰色", "盘子：陶瓷/一次性餐具难回收，投入灰色其他垃圾桶。"),
    "spoon": ("其他垃圾", "灰色", "一次性餐具：投入灰色其他垃圾桶。"),
    "fork": ("其他垃圾", "灰色", "一次性餐具：投入灰色其他垃圾桶。"),
    "knife": ("其他垃圾", "灰色", "一次性餐具：投入灰色其他垃圾桶。"),
}

# 明确不是垃圾的对象（人、动物、交通工具、家具以外的场景物等）：不产生投放建议
_NON_WASTE = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "bird", "fish", "rabbit", "monkey", "pigeon", "chicken", "duck",
    "traffic light", "stop sign", "fire hydrant", "bench", "potted plant", "vase",
    "clock", "umbrella", "handbag", "backpack", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle cap", "hair drier", "scissors",
    "door", "window", "stairs", "mirror",
})

_BIN_COLORS = ("蓝色", "绿色", "红色", "灰色")


def waste_advice(label: object) -> dict:
    """把一个 COCO 类别名翻译成投放建议字典。

    返回字段：label / type / bin / advice / is_waste / known。
    - 非垃圾对象：is_waste=False，known=True。
    - 已收录垃圾：is_waste=True，known=True，给出具体建议。
    - 未知类别：is_waste=True，known=False，按其他垃圾兜底。
    """
    name = str(label or "").strip().lower()
    if name in _NON_WASTE:
        return {"label": name, "type": "非垃圾对象", "bin": "-", "advice": "",
                "is_waste": False, "known": True}
    if name in _COCO_WASTE:
        kind, color, advice = _COCO_WASTE[name]
        return {"label": name, "type": kind, "bin": color, "advice": advice,
                "is_waste": True, "known": True}
    return {"label": name, "type": "其他垃圾", "bin": "灰色",
            "advice": "未收录类别：若无法回收，投入灰色其他垃圾桶。",
            "is_waste": True, "known": False}


def summarize(detections: list[dict]) -> dict:
    """汇总检测结果：只统计垃圾对象，按类别给出数量与投放建议。"""
    from collections import Counter

    waste_items = [d for d in detections if waste_advice(d.get("label", ""))["is_waste"]]
    total = len(detections)
    waste_count = len(waste_items)
    type_counts = Counter(waste_advice(d.get("label", ""))["type"] for d in waste_items)
    label_counts = Counter(str(d.get("label", "")).strip().lower() for d in waste_items)
    recyclable = type_counts.get("可回收物", 0)

    items = []
    for label, count in sorted(label_counts.items(), key=lambda p: (-p[1], p[0])):
        info = waste_advice(label)
        items.append({"label": label, "zh": objects_vocab.zh_name(label),
                      "count": count, "type": info["type"],
                      "bin": info["bin"], "advice": info["advice"]})

    if waste_count == 0:
        headline = f"共检测 {total} 个对象，未发现需要投放的垃圾。"
    else:
        headline = (
            f"共检测 {total} 个对象，其中垃圾 {waste_count} 件、可回收物 {recyclable} 件；"
            f"请按各类建议分别投放。"
        )
    return {"total": total, "waste_count": waste_count, "recyclable": recyclable,
            "items": items, "headline": headline}
