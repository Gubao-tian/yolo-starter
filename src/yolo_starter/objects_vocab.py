"""全物体提取词表：给 YOLO-World 的开放词汇类别清单。

YOLO-World 不再局限于固定的 80 类：给定一组类别名，它会去图里找这些
东西。本模块维护一份覆盖「食物 / 餐具 / 电子 / 垃圾 / 家居 / 人物动物」
的常见物品词表，并提供中文名、分组与去重汇总。

注意：词表决定能识别的上限。没写进词表的东西识别不出；写进词表的东西
也可能被近似成最相近的一类（如菠萝→orange）。这是开放词汇模型的特性，
不是 bug。可通过扩充 WORDS / ZH_NAMES 覆盖更多物品。
"""
from __future__ import annotations

from collections import Counter

# 含空格的多词类别（YOLO-World 支持，避免被"无空格"断言误判）
MULTIWORD = {
    "cell phone", "hot dog", "wine glass", "light bulb", "teddy bear",
    "traffic light", "fire hydrant", "sports ball", "potted plant",
    "dining table", "washing machine", "air conditioner", "trash can",
    "stop sign", "plastic bag",
}

# 按分组组织，键仅用于文档归类；WORDS 为扁平去重后的完整清单
GROUPS: dict[str, list[str]] = {
    "食物": [
        "apple", "banana", "orange", "broccoli", "carrot", "bread", "butter",
        "cheese", "pineapple", "nuts", "meatball", "hot dog", "sandwich", "cake",
        "donut", "pizza", "egg", "tomato", "potato", "cucumber", "grape",
        "strawberry", "watermelon", "lemon", "peach", "pear", "corn", "mushroom",
        "fish", "chicken", "beef", "rice", "noodle", "cookie", "candy",
    ],
    "餐具": [
        "bowl", "plate", "cup", "bottle", "wine glass", "spoon", "fork", "knife",
        "can", "jar", "straw", "dining table",
    ],
    "电子": [
        "cell phone", "laptop", "tv", "keyboard", "mouse", "camera", "headphones",
        "battery", "light bulb", "remote", "washing machine", "air conditioner",
    ],
    "垃圾/日用品": [
        "trash", "trash can", "paper", "cardboard", "newspaper", "book", "tissue",
        "toothbrush", "soap", "detergent", "plastic bag",
    ],
    "家居": [
        "chair", "couch", "bed", "desk", "door", "window", "clock", "vase",
        "umbrella", "backpack", "handbag", "key", "mirror", "lamp", "pillow",
        "teddy bear", "potted plant",
        "basket", "curtain", "cabinet", "dresser", "bookshelf", "clothes",
        "carpet", "frame", "shoes", "blanket", "nightstand",
    ],
    "交通/户外": [
        "car", "bicycle", "bus", "truck", "motorcycle", "train", "airplane",
        "traffic light", "stop sign", "fire hydrant", "bench", "sports ball",
    ],
    "人物/动物": [
        "person", "dog", "cat", "bird", "horse", "cow", "sheep",
    ],
}

# 扁平化并去重（保持顺序）
WORDS: list[str] = []
for _group in GROUPS.values():
    for _w in _group:
        _w = _w.strip().lower()
        if _w not in WORDS:
            WORDS.append(_w)

# 分组归属（去重后）
_group_of: dict[str, str] = {}
for _g, _ws in GROUPS.items():
    for _w in _ws:
        _group_of.setdefault(_w.strip().lower(), _g)

ZH_NAMES: dict[str, str] = {
    "apple": "苹果", "banana": "香蕉", "orange": "橙子", "broccoli": "西兰花",
    "carrot": "胡萝卜", "bread": "面包", "butter": "黄油", "cheese": "奶酪",
    "pineapple": "菠萝", "nuts": "坚果", "meatball": "肉丸", "hot dog": "热狗",
    "sandwich": "三明治", "cake": "蛋糕", "donut": "甜甜圈", "pizza": "披萨",
    "egg": "鸡蛋", "tomato": "番茄", "potato": "土豆", "cucumber": "黄瓜",
    "grape": "葡萄", "strawberry": "草莓", "watermelon": "西瓜", "lemon": "柠檬",
    "peach": "桃子", "pear": "梨", "corn": "玉米", "mushroom": "蘑菇",
    "fish": "鱼", "chicken": "鸡肉", "beef": "牛肉", "rice": "米饭",
    "noodle": "面条", "cookie": "饼干", "candy": "糖果",
    "bowl": "碗", "plate": "盘子", "cup": "杯子", "bottle": "瓶子",
    "wine glass": "玻璃杯", "spoon": "勺子", "fork": "叉子", "knife": "刀",
    "can": "易拉罐", "jar": "罐子", "straw": "吸管", "dining table": "餐桌",
    "cell phone": "手机", "laptop": "笔记本电脑", "tv": "电视", "keyboard": "键盘",
    "mouse": "鼠标", "camera": "相机", "headphones": "耳机", "battery": "电池",
    "light bulb": "灯泡", "remote": "遥控器", "washing machine": "洗衣机",
    "air conditioner": "空调",
    "trash": "垃圾", "trash can": "垃圾桶", "paper": "纸张", "cardboard": "纸板",
    "newspaper": "报纸", "book": "书本", "tissue": "纸巾", "toothbrush": "牙刷",
    "soap": "肥皂", "detergent": "洗涤剂", "plastic bag": "塑料袋",
    "chair": "椅子", "couch": "沙发", "bed": "床", "desk": "桌子",
    "door": "门", "window": "窗", "clock": "时钟", "vase": "花瓶",
    "umbrella": "雨伞", "backpack": "背包", "handbag": "手提包", "key": "钥匙",
    "mirror": "镜子", "lamp": "台灯", "pillow": "枕头", "teddy bear": "泰迪熊",
    "potted plant": "盆栽",
    "basket": "篮子", "curtain": "窗帘", "cabinet": "柜子", "dresser": "抽屉柜",
    "bookshelf": "书架", "clothes": "衣服", "carpet": "地毯", "frame": "相框",
    "shoes": "鞋", "blanket": "毯子", "nightstand": "床头柜",
    "car": "汽车", "bicycle": "自行车", "bus": "公交车", "truck": "卡车",
    "motorcycle": "摩托车", "train": "火车", "airplane": "飞机",
    "traffic light": "红绿灯", "stop sign": "停止标志", "fire hydrant": "消防栓",
    "bench": "长椅", "sports ball": "球",
    "person": "人", "dog": "狗", "cat": "猫", "bird": "鸟", "horse": "马",
    "cow": "牛", "sheep": "羊",
    # COCO 80 类补充：垃圾分类模式会用到这些名称，统一在此提供中文显示
    "boat": "船", "parking meter": "停车计时器", "tie": "领带",
    "suitcase": "行李箱", "frisbee": "飞盘", "skis": "滑雪板",
    "snowboard": "单板滑雪板", "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "elephant": "大象", "bear": "熊",
    "zebra": "斑马", "giraffe": "长颈鹿", "toilet": "马桶",
    "microwave": "微波炉", "oven": "烤箱", "toaster": "烤面包机",
    "sink": "水槽", "refrigerator": "冰箱", "scissors": "剪刀",
    "hair drier": "吹风机",
}


def group_of(label: str) -> str:
    """返回类别所属分组，未知类别归入『其他』。"""
    return _group_of.get(str(label or "").strip().lower(), "其他")


# 微调模型 data.yaml 里的类别名（单数/简写）→ 中文名。
# 这些不在开放词汇 WORDS 里，仅用于把自定义模型的输出翻译成中文。
MODEL_CLASS_ALIASES = {
    "nut": "坚果",
    "book": "书本",
}


def zh_name(label: str) -> str:
    """返回中文名，未知类别原样返回。"""
    name = str(label or "").strip().lower()
    if name in ZH_NAMES:
        return ZH_NAMES[name]
    return MODEL_CLASS_ALIASES.get(name, name)


def summarize_objects(detections: list[dict]) -> list[dict]:
    """按类别去重汇总检测结果，返回 [{label, zh, group, count, best_score}]。

    按数量降序、类别名升序排列；best_score 为该类别最高置信度。
    """
    counts = Counter(str(d.get("label", "")).strip().lower() for d in detections)
    best: dict[str, float] = {}
    for d in detections:
        label = str(d.get("label", "")).strip().lower()
        score = float(d.get("score", 0) or 0)
        best[label] = max(best.get(label, 0.0), score)
    rows = [
        {"label": label, "zh": zh_name(label), "group": group_of(label),
         "count": count, "best_score": round(best.get(label, 0.0), 3)}
        for label, count in counts.items()
    ]
    rows.sort(key=lambda r: (-r["count"], r["label"]))
    return rows
