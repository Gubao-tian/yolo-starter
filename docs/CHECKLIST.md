# 四项能力验收清单

汇报前逐项勾选。Windows 用户请把命令逐条复制运行，不要用 `&&` 连接。

## 能看

- [ ] `reports/index.html` 存在，双击可在浏览器打开。
- [ ] 看板标题为「校园垃圾分类投放识别 · YOLOv8 实验看板」。
- [ ] 看板包含四项能力总览、实验设计表、指标对比表、三张柱状图、检测效果、投放建议表、自动结论。
- [ ] 看板顶部有「演示数据」徽标与醒目说明。
- [ ] 断网后仍能打开，页面不请求外部资源。
- [ ] `outputs/summary.md` 存在，文字摘要与看板结论一致。

验证命令（逐条运行）：

```bash
python scripts/run_all.py --mode demo
python -m http.server 8000 --directory reports
```

## 能跑

- [ ] `python scripts/check.py` 退出码为 0，并打印 demo 与 full 提示。
- [ ] `python -m unittest discover -s tests -v` 全部通过（103 个测试）。
- [ ] `python scripts/run_all.py --mode demo` 产出 JSON、CSV、Markdown、HTML 四类文件。
- [ ] 在含空格路径下运行入口脚本不报错。
- [ ] 准备数据集后 `--mode full` 可调用真实 YOLOv8（可选验收项）。

验证命令（逐条运行）：

```bash
python scripts/check.py
python -m unittest discover -s tests -v
python scripts/run_all.py --mode demo
```

## 能对比

- [ ] `configs/experiments.json` 包含 E1~E4，且基线 E1 存在。
- [ ] 四组实验只改分辨率、置信度、模型规模中的少数变量。
- [ ] `outputs/comparison.csv` 含 ΔmAP、Δ耗时等相对基线差值。
- [ ] 看板有柱状图与自动结论，说明精度-速度权衡。
- [ ] E3 的结论说明提高置信度会让 Precision 上升、Recall 下降。

验证命令：

```bash
python scripts/run_all.py --mode demo --experiments E1,E2,E3
```

## 能复述

- [ ] 能在 5 分钟内按 `docs/WALKTHROUGH.md` 的 6 段覆盖看板 8 节。
- [ ] 能说清 YOLO 是单阶段检测器：一次前向同时预测框和类别。
- [ ] 能说清「YOLO 负责检测、规则表负责投放建议」的分工。
- [ ] 能说出 6 类垃圾名称与对应投放桶颜色。
- [ ] 能解释 mAP@0.50 与 mAP@0.50:0.95 的区别。
- [ ] 能解释 Precision 与 Recall 的权衡。
- [ ] 能回答讲稿中的 7 个答辩 Q&A，包括完整复现命令。

## 现场加演：上传自己的照片实时识别（可选加分项）

- [ ] 双击 `webapp\启动识别.bat`（或运行 `.\.venv\Scripts\python.exe webapp\app.py`），浏览器自动打开 `http://127.0.0.1:5000`。
- [ ] 上传一张自带照片，点「开始识别」，能画出检测框并给出投放建议。
- [ ] 能说明这是 COCO 预训练模型（80 类日常物品），非专门的垃圾模型。
- [ ] 能演示人/车等非垃圾对象不产生投放建议。
- [ ] 能说明未收录类别按「其他垃圾」兜底，建议仅供教学演示。

## 现场加演：全物体提取（开放词汇，可选加分项）

- [ ] 在页面点「全物体提取」按钮切换到该模式。
- [ ] 上传饭盒图或点示例图 02，约 5 秒后输出带框图片 + 物品清单表（中文名/分组/数量/置信度）。
- [ ] 能说出：这是 YOLO-World，按 `objects_vocab.py` 的 117 类词表识别，不受 80 类限制。
- [ ] 能说明局限：没收录的东西会被近似成最相近类别（菠萝→柠檬/橙子），这是模型特性非 bug。
- [ ] 能说明首次会下载约 338MB 的 CLIP 文本模型并缓存，之后离线可用。

## 演示前 60 秒快速检查

- [ ] 在项目根目录运行 `python scripts/run_all.py --mode demo --open`。
- [ ] 确认看板徽标显示「演示数据」，并在讲解开头说明这一点。
- [ ] 确认投放建议表显示塑料/纸张/玻璃/其他垃圾及建议。
- [ ] 浏览器缩放调到适合投影的比例。
- [ ] 打开 `docs/WALKTHROUGH.md` 作为提词稿。
