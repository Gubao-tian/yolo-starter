# 设计说明：校园垃圾分类投放识别 · YOLOv8 可视化入门项目

## 1. 意图与交付场景

课程要求每个项目同时具备「能看、能跑、能对比、能复述」。本项目是一个**具体应用**：
用 YOLOv8 识别校园场景中的 6 类垃圾（塑料瓶、纸张、纸板、玻璃、金属、其他垃圾），
输出检测框、类别、置信度和对应的投放建议，并用受控实验对比不同分辨率与置信度阈值的效果。
面向第一次接触 YOLO 的同学，用于课堂汇报与课程作业提交，不追求 SOTA 精度。

## 2. 关键约束

- 目标机器可能只有 CPU、没有 GPU，且可能没有联网；
- 目标机器可能只装了裸 Python，没有 torch / ultralytics；
- 因此默认 `demo` 模式必须在**零第三方依赖**下产出完整看板；
- `full` 模式使用真实 Ultralytics YOLOv8，依赖 `requirements.txt`；需按
  `datasets/campus_trash/README.md` 准备 YOLO 格式数据集；
- 全部产物（HTML、SVG、CSV、JSON、Markdown）均离线可打开，不引用 CDN。

## 3. 四项能力 → 交付物

| 能力 | 判定标准 | 交付物 | 验证命令 |
| --- | --- | --- | --- |
| 能看 | 打开一个页面即可看到垃圾检测框、投放建议、指标图、对比表、结论 | `reports/index.html`（自包含 HTML + 内嵌 SVG）、`outputs/summary.md` | `python scripts/run_all.py --mode demo --open` |
| 能跑 | 一条命令产出全部产物，另有自检与单测 | `scripts/run_all.py`、`scripts/check.py`、`tests/` | `python scripts/check.py`、`python -m unittest discover -s tests -v` |
| 能对比 | ≥3 组受控实验，同一数据集同一指标口径，有表格、图、自动结论 | `configs/experiments.json`、`outputs/results.json`、`outputs/comparison.csv` | `python scripts/run_all.py --mode demo` 或 `--mode full --dataset datasets/campus_trash/data.yaml --epochs 20` |
| 能复述 | 讲清「YOLO 检测 + 规则投放建议」的分工、流程、结果、取舍 | `README.md`、`docs/WALKTHROUGH.md`、`docs/CHECKLIST.md` | 按讲稿复述 + 勾选清单 |

## 4. 双模式架构

```
                    configs/experiments.json
                              │
                              ▼
scripts/run_all.py ──> pipeline.run(mode=...)
                              │
                ┌─────────────┴─────────────┐
                │                           │
        mode = demo                   mode = full
    samples.py 示例检测框            datasets/campus_trash/data.yaml
    trash_rules.py 生成投放建议      yolo_backend → ultralytics 训练/验证/推理
    （明确标注"非真实推理"）          （真实 YOLOv8 结果）
                │                           │
                └─────────────┬─────────────┘
                              ▼
       metrics.py 归一化 → compare 表 → 自动结论
                              ▼
       visualize.py（纯 SVG + 投放建议表） + report.py（自包含 HTML）
                              ▼
       outputs/results.json / comparison.csv / summary.md
       reports/index.html
```

- **demo 模式**：不装包、不联网、不需要数据集，用固定示例检测框展示完整看板。
  指标与检测框都是教学示例值，看板顶部有醒目的「演示数据」标记，用于验证链路、答辩预演和现场兜底。
- **full 模式**：安装依赖并准备好 `datasets/campus_trash` 后，训练并验证真实 YOLOv8 模型，
  产出真实指标与检测图；真实模式不混入示例投放建议，避免误导。
- 两种模式共用实验配置、指标口径、可视化与报告代码。

## 5. 类别与投放建议

 YOLO 负责“是什么、在哪里”，`trash_rules.py` 用可解释规则负责“投哪个桶”。

| 类别 | 投放建议 |
| --- | --- |
| plastic 塑料瓶 | 可回收物：清空冲洗后投蓝色桶 |
| paper 纸张 | 可回收物：保持干燥投蓝色桶 |
| cardboard 纸板 | 可回收物：压平后投蓝色桶 |
| glass 玻璃 | 可回收物：轻放玻璃回收箱 |
| metal 金属 | 可回收物：清空后投蓝色桶 |
| trash 其他垃圾 | 其他垃圾：投灰色桶 |

## 6. 受控实验设计

| 编号 | 变量 | 期望现象 |
| --- | --- | --- |
| E1 | 基准 yolov8n / imgsz=320 / conf=0.25 | 最快，精度作为基线 |
| E2 | imgsz 320 → 640 | 小目标特征更完整，mAP 上升、耗时上升 |
| E3 | conf 0.25 → 0.50（沿用 640） | 误投建议减少，Precision 上升、Recall 下降 |
| E4 | 模型 n → s | mAP 上升、参数量与耗时上升 |

指标统一为 mAP@0.50、mAP@0.50:0.95、Precision、Recall、单图耗时(ms)、参数量(M)。
对比表以 E1 为基线给出差值，结论由代码自动生成。

## 7. 目录结构

```
yolo-starter/
├── README.md                      项目总览与快速开始
├── requirements.txt               full 模式依赖
├── configs/experiments.json       实验矩阵
├── datasets/campus_trash/         数据模板与标注说明
├── docs/
│   ├── DESIGN.md                  本文件
│   ├── WALKTHROUGH.md             5 分钟讲稿 + 答辩 Q&A
│   └── CHECKLIST.md               四项能力验收清单
├── src/yolo_starter/
│   ├── paths.py / experiments.py  路径与配置
│   ├── samples.py                 demo 示例检测框与指标
│   ├── trash_rules.py             类别 → 投放建议规则
│   ├── metrics.py                 指标归一化与自动结论
│   ├── visualize.py               SVG 图表、检测框、投放建议表
│   ├── report.py                  自包含 HTML 看板
│   ├── yolo_backend.py            ultralytics 封装（延迟导入）
│   └── pipeline.py                双模式编排
├── scripts/                       一键入口与环境自检
├── tests/                         unittest 测试
├── outputs/                       结构化产物
└── reports/                       看板 HTML
```

## 8. 我做出的选型（可调整）

1. 框架选 Ultralytics YOLOv8：文档完整、yolov8n 约 6 MB、CPU 可跑。
2. 数据集选自建校园垃圾检测集：贴近生活、类别定义明确、无版权风险；代价是需要拍照标注。
3. “投哪个桶”用规则表而不是再训练一个分类模型：分工清晰，便于课堂解释。
4. demo 模式用示例检测框而非图片识别：零依赖、可离线；必须显著标注非真实结果。
5. 图表用内嵌 SVG，测试用标准库 unittest，不引入前端构建或 pytest。

## 9. 非目标

- 不追求检测精度领先，不做调参竞赛；
- 不实现摄像头实时 GUI，不部署量化/ONNX；
- 不做多语言、不做云端服务。
