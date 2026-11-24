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

# 明确不是垃圾的对象（人、动物、交通工具、场景物等）：不产生投放建议
_NON_WASTE = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "bird", "fish", "rabbit", "monkey", "pigeon", "chicken", "duck",
    "traffic light", "stop sign", "fire hydrant", "bench", "potted plant",
    "umbrella", "handbag", "backpack", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle cap", "hair drier", "scissors",
    "door", "window", "stairs",
})

# 开放词汇补充类别（YOLO-World 独有词表）-> (大类, 桶色, 建议)
# 双模型融合模式下，这些类别会由 YOLO-World 检出并套用以下规则。
_EXTENDED_WASTE: dict[str, tuple[str, str, str]] = {
    # 家居 -> 可回收物
    "lamp": ("可回收物", "蓝色", "台灯：拆除灯泡（灯泡投有害垃圾）后，灯体属可回收物，投入蓝色可回收桶或废品回收点。"),
    "mirror": ("可回收物", "蓝色", "镜子：玻璃制品，轻放投入玻璃/可回收回收点。"),
    "basket": ("可回收物", "蓝色", "篮子：塑料/金属篮倒空清洗后投入蓝色可回收桶。"),
    "clothes": ("可回收物", "蓝色", "旧衣物：清洗晾干后投入衣物回收箱。"),
    "trash can": ("可回收物", "蓝色", "废旧垃圾桶：倒空清洗后投入蓝色可回收桶。"),
    # 家居 -> 其他垃圾
    "curtain": ("其他垃圾", "灰色", "窗帘：大件布艺，可洗净后投放衣物回收箱，或按其他垃圾处理。"),
    "blanket": ("其他垃圾", "灰色", "毯子/床上用品：洗净后可投衣物回收箱，或投入灰色其他垃圾桶。"),
    "carpet": ("其他垃圾", "灰色", "地毯：大件布艺，按其他垃圾或预约大件清运处理。"),
    "frame": ("其他垃圾", "灰色", "相框：拆除玻璃后按其他垃圾处理。"),
    "pillow": ("其他垃圾", "灰色", "枕头：投入灰色其他垃圾桶或衣物回收箱。"),
    "teddy bear": ("其他垃圾", "灰色", "毛绒玩具：洗净后可投衣物回收箱，或投入灰色其他垃圾桶。"),
    "shoes": ("其他垃圾", "灰色", "旧鞋：较完整的可投衣物回收箱，破损的投入灰色其他垃圾桶。"),
    "vase": ("其他垃圾", "灰色", "花瓶：陶瓷制品难回收，投入灰色其他垃圾桶。"),
    "clock": ("其他垃圾", "灰色", "时钟：先取出电池投有害垃圾，钟体投入灰色其他垃圾桶。"),
    # 大件家具 -> 预约清运
    "cabinet": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "dresser": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "bookshelf": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "nightstand": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    "desk": ("其他垃圾", "灰色", "大件家具：预约大件垃圾清运，勿投入普通桶。"),
    # 电子补充
    "remote": ("有害垃圾", "红色", "遥控器：含电池，取出电池（电池投有害垃圾）后投电子回收点。"),
    "washing machine": ("有害垃圾", "红色", "废旧电器：投放至电子废弃物回收点。"),
    "air conditioner": ("有害垃圾", "红色", "废旧电器：投放至电子废弃物回收点。"),
    "camera": ("可回收物", "蓝色", "相机：属电子废弃物，投放至电子回收点。"),
    "headphones": ("可回收物", "蓝色", "耳机：属电子废弃物，投放至电子回收点。"),
    # 日用品
    "plastic bag": ("其他垃圾", "灰色", "塑料袋：难降解，投入灰色其他垃圾桶。"),
    "straw": ("其他垃圾", "灰色", "吸管：一次性塑料，投入灰色其他垃圾桶。"),
    "soap": ("其他垃圾", "灰色", "皂类：用剩的肥皂投入灰色其他垃圾桶。"),
    "jar": ("可回收物", "蓝色", "玻璃罐：洗净倒空后投入玻璃回收点。"),
    "detergent": ("可回收物", "蓝色", "洗涤剂瓶：倒空冲洗后投入蓝色可回收桶，残留液体按说明处理。"),
    # 食物补充 -> 厨余垃圾
    "egg": ("厨余垃圾", "绿色", "蛋壳：投入绿色厨余桶。"),
    "tomato": ("厨余垃圾", "绿色", "果皮菜叶：投入绿色厨余桶。"),
    "potato": ("厨余垃圾", "绿色", "果皮菜叶：投入绿色厨余桶。"),
    "cucumber": ("厨余垃圾", "绿色", "果皮菜叶：投入绿色厨余桶。"),
    "grape": ("厨余垃圾", "绿色", "果皮：投入绿色厨余桶。"),
    "strawberry": ("厨余垃圾", "绿色", "果皮：投入绿色厨余桶。"),
    "watermelon": ("厨余垃圾", "绿色", "瓜皮：投入绿色厨余桶。"),
    "lemon": ("厨余垃圾", "绿色", "果皮：投入绿色厨余桶。"),
    "peach": ("厨余垃圾", "绿色", "果核果皮：投入绿色厨余桶。"),
    "pear": ("厨余垃圾", "绿色", "果核果皮：投入绿色厨余桶。"),
    "corn": ("厨余垃圾", "绿色", "玉米芯：投入绿色厨余桶。"),
    "mushroom": ("厨余垃圾", "绿色", "剩菜：投入绿色厨余桶。"),
    "bread": ("厨余垃圾", "绿色", "剩面包：投入绿色厨余桶。"),
    "butter": ("厨余垃圾", "绿色", "剩余食品：投入绿色厨余桶。"),
    "cheese": ("厨余垃圾", "绿色", "剩余食品：投入绿色厨余桶。"),
    "pineapple": ("厨余垃圾", "绿色", "果皮：投入绿色厨余桶。"),
    "nuts": ("其他垃圾", "灰色", "坚果壳：硬壳难腐烂，投入灰色其他垃圾桶（果仁属厨余）。"),
    "meatball": ("厨余垃圾", "绿色", "剩菜：投入绿色厨余桶。"),
    "rice": ("厨余垃圾", "绿色", "剩饭：投入绿色厨余桶。"),
    "noodle": ("厨余垃圾", "绿色", "剩面：投入绿色厨余桶。"),
    "cookie": ("厨余垃圾", "绿色", "剩余食品：投入绿色厨余桶。"),
    "candy": ("其他垃圾", "灰色", "糖果及包装纸：投入灰色其他垃圾桶。"),
    "fish": ("厨余垃圾", "绿色", "鱼骨内脏：投入绿色厨余桶。"),
    "chicken": ("厨余垃圾", "绿色", "剩余食品：大骨难腐烂可投其他垃圾。"),
    "beef": ("厨余垃圾", "绿色", "剩余食品：大骨难腐烂可投其他垃圾。"),
}

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
    rule = _COCO_WASTE.get(name) or _EXTENDED_WASTE.get(name)
    if rule:
        kind, color, advice = rule
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
