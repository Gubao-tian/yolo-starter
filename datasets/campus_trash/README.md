# 校园垃圾检测数据集

本项目把检测对象限定为校园里最常见的 6 类垃圾，`data.yaml` 已经写好类别定义。

| 编号 | 类别 | 投放建议 |
| --- | --- | --- |
| 0 | plastic 塑料瓶 | 蓝色可回收桶 |
| 1 | paper 纸张 | 蓝色可回收桶 |
| 2 | cardboard 纸板 | 蓝色可回收桶 |
| 3 | glass 玻璃 | 玻璃回收箱 |
| 4 | metal 金属 | 蓝色可回收桶 |
| 5 | trash 其他垃圾 | 灰色其他垃圾桶 |

## 目录结构

```
datasets/campus_trash/
├── data.yaml
├── images/train/       训练图片
├── images/val/         验证图片
├── labels/train/       与图片同名的 .txt 标签
└── labels/val/
```

每行标签格式：`类别编号 x_center y_center width height`，坐标均为 0~1 相对值。

## 最小可用方案（课堂演示）

1. 用手机在教室、宿舍、食堂拍 30~50 张垃圾照片，每类至少 5 张。
2. 用 LabelImg 或 CVAT 框出垃圾，按上表编号保存为 YOLO 格式 `.txt`。
3. 按 8:2 划分到 `images/train` 与 `images/val`，标签放到对应 `labels` 目录。
4. 运行真实实验：

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/run_all.py --mode full --dataset datasets/campus_trash/data.yaml --epochs 20
```

如果暂时没有数据集，用 `python scripts/run_all.py --mode demo` 也能完整演示看板与讲解流程；
demo 模式会明确标注「演示数据」，不会把示例值冒充真实实验结果。

## 为什么不用公开 TrashNet 直接跑

TrashNet 主要是单物体分类图，没有检测框；YOLO 需要带边界框的标注。
若要节省标注时间，可以用 TACO 等带检测框的公开垃圾数据集，再把类别映射到上面的 6 类。
