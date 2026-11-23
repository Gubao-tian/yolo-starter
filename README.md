# 校园垃圾分类投放识别 · YOLOv8 可视化入门项目

一个可直接用于课堂汇报与课程作业的**具体项目**：用 YOLOv8 识别校园场景中的 6 类垃圾（塑料瓶、纸张、纸板、玻璃、金属、其他垃圾），输出检测框、类别、置信度与投放建议，并用受控实验对比不同输入分辨率和置信度阈值的效果。

课程要求的四件事都被做成了可验证的交付物：**能看、能跑、能对比、能复述**。

## 四项能力

| 能力 | 在本项目中的体现 | 现场验证 |
| --- | --- | --- |
| 能看 | `reports/index.html` 汇总垃圾检测框、投放建议表、三张柱状图、对比表和自动结论；双击即可离线打开 | 打开看板 |
| 能跑 | `scripts/run_all.py` 一键产出全部文件；`scripts/check.py` 自检；103 个 unittest 测试 | 运行下方命令 |
| 能对比 | E1~E4 受控实验，统一指标口径，以 E1 为基线给出 Δ 值与自动结论 | 看对比表 |
| 能复述 | 本 README + `docs/WALKTHROUGH.md` 讲稿 + `docs/CHECKLIST.md` 验收清单 | 按讲稿讲 5 分钟 |

## 快速开始

### 1. 零依赖演示（汇报现场推荐）

不需要 GPU、不需要联网、不需要安装第三方包，也不需要数据集：

```bash
python scripts/check.py
python -m unittest discover -s tests -v
python scripts/run_all.py --mode demo --open
```

`--open` 会自动打开浏览器；若没有自动打开，手动双击 `reports/index.html` 即可。

> **重要**：`demo` 模式的检测框、投放建议和指标都是教学示例值，看板顶部有醒目的「演示数据」标记。它用于验证链路、练习讲解和现场兜底，**不是真实 YOLO 推理结果**。

### 2. 真实 YOLOv8 实验（可选增强）

先按 `datasets/campus_trash/README.md` 拍照、标注并放好 YOLO 格式图片与标签，然后：

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/run_all.py --mode full --dataset datasets/campus_trash/data.yaml --epochs 20
```

Linux/macOS 用 `source .venv/bin/activate` 激活环境后运行同样的 `python scripts/run_all.py ...` 命令。

### 3. 本地预览看板

```bash
python -m http.server 8000 --directory reports
```

浏览器访问 `http://localhost:8000/`。

## 上传照片自动识别（真实 YOLO 推理）

这是新增的交互式功能：**在网页里上传你自己的照片 → YOLOv8 真实识别 → 画出检测框 → 自动给出投放建议**。与上面的静态看板（教学示例数据）不同，这里跑的是真实模型。

**首次准备（只需一次）**：

```powershell
cd "E:\work  buddy\2026-11-23-10-44-32\yolo-starter"
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ultralytics flask
```

两个权重文件都已放在 `weights/` 里：`yolov8n.pt`（垃圾分类模式，约 6.3MB）、`yolov8s-world.pt`（全物体提取模式，约 26MB）。若丢失，可重新下载：

```powershell
curl -L -o weights/yolov8n.pt "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
curl -L -o weights/yolov8s-world.pt "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt"
```

> 全物体提取模式额外依赖 CLIP 文本编码器（用于"看懂"类别名），已装好。换电脑重装时执行：
> ```powershell
> .\.venv\Scripts\python.exe -m pip install "git+https://gh-proxy.com/https://github.com/ultralytics/CLIP.git"
> ```
> 首次运行会自动下载约 338MB 的 CLIP 文本模型并缓存到 `weights/clip/`，之后离线可用。

**启动网页**：

```powershell
.\.venv\Scripts\python.exe webapp\app.py
```

浏览器打开 `http://127.0.0.1:5000`。页面有四种模式，点按钮切换：

- **垃圾分类（80 类）**：默认模式。上传照片 → 调置信度滑块 → 点「开始识别」，输出带检测框的照片 + 每类垃圾的投放建议。速度快。
- **全物体提取**：用 YOLO-World 开放词汇模型，按 117 类词表（`src/yolo_starter/objects_vocab.py`，覆盖食物 / 餐具 / 电子 / 垃圾 / 家居 / 交通 / 人物动物）把照片里的物品尽量都框出来，输出「物品清单」表（中文名、分组、数量、最高置信度）。首次识别约需 5 秒准备，之后每张约 1~2 秒。切到该模式会自动把阈值设为 0.25，避免低阈值下把花纹误认成人或玩偶。
- **密集书本**：切片推理 + 高分辨率，专攻书架上密集排列的小目标，自动把阈值设为 0.15。详见 `test_images/README.md`。
- **坚果识别**：用人工标注微调的模型识别饭盒里带壳坚果，自动过滤误检大框。权重 `weights/yolov8n-nuts.pt` 由 `python scripts/train_nuts.py --epochs 40` 训练得到，链路与局限见 `docs/WALKTHROUGH.md` Q12。

**没有照片也能演示**：页面下半部分有 8 张示例图，点任意一张即自动识别。这些是 COCO 公开数据集的真实照片，精选覆盖可回收/厨余/有害/其他和非垃圾对象，详见 `test_images/README.md`。

**两种模式的区别**：垃圾分类模式用 COCO 预训练权重，只能认固定的 80 类，靠 `waste_map.py` 把常见类别映射到四类垃圾和桶色，人/猫狗/汽车等非垃圾对象不产生投放建议。全物体提取模式不受 80 类限制——你给它什么词表它就找什么，所以饭盒里的面包、盘子、鸡蛋、柠檬等都能识别。**但开放词汇模型会把没收录的东西近似成最相近的一类**（如菠萝被认成柠檬/橙子、肉丸被认成热狗或牛肉），这是模型特性而非 bug；要更准就把类别名加进 `objects_vocab.py` 的词表。**投放建议仅供教学演示，实际请以当地分类标准为准。**

> 汇报演示建议：先用「垃圾分类」模式上传一瓶饮料 + 一张街景，展示"框出来 + 给建议 + 区分非垃圾对象"；再切到「全物体提取」模式上传饭盒图，展示 10 类物品清单——正好对比"固定 80 类 vs 开放词汇"的差别，是很好的加分点。

## 项目做了什么

1. **检测**：YOLOv8 找出画面中的垃圾，输出类别与置信度。
2. **决策**：`src/yolo_starter/trash_rules.py` 把类别映射成投放建议——YOLO 负责“是什么、在哪里”，规则表负责“投哪个桶”，分工清楚、便于讲解。
3. **对比**：四组实验只改少数变量，指标口径一致，结论由代码从数据生成。
4. **复现**：所有命令、参数、数据格式和讲稿都写在项目里。

## 实验矩阵

| 编号 | 配置 | 改动 | 预期现象 |
| --- | --- | --- | --- |
| E1 | yolov8n / 320 / conf 0.25 | 基准 | 最快，作为基线 |
| E2 | yolov8n / 640 / conf 0.25 | 分辨率 320→640 | 瓶罐、纸片等小目标检测更准，mAP 上升、耗时上升 |
| E3 | yolov8n / 640 / conf 0.50 | 置信度 0.25→0.50 | 低置信误检减少，Precision 上升、Recall 下降 |
| E4 | yolov8s / 640 / conf 0.25 | 模型 n→s | 相似类别区分更好，mAP 上升但参数量与耗时上升 |

指标：mAP@0.50、mAP@0.50:0.95、Precision、Recall、单图耗时(ms)、参数量(M)。Δ 列以 E1 为基线。

## 常用参数

```bash
python scripts/run_all.py --mode demo --experiments E1,E2
python scripts/run_all.py --mode full --dataset datasets/campus_trash/data.yaml --epochs 20 --limit 3
python scripts/run_all.py --mode demo --config configs/experiments.json --root /tmp/my-output
```

- `--mode`：`demo`（零依赖示例）或 `full`（真实 YOLOv8）。
- `--experiments`：逗号分隔的实验编号，默认用配置中的 `fast_set`。
- `--dataset`：数据集 YAML；真实模式默认 `datasets/campus_trash/data.yaml`。
- `--epochs`：训练轮数，默认 0；真实实验建议 20。
- `--limit`：**仅 full 模式**保存几张推理结果图，默认 3。
- `--root`：把产物写到别处，方便测试。
- `--open`：生成后打开浏览器。

## 产物

```
outputs/results.json      结构化结果（环境、指标、原始值、结论、警告）
outputs/comparison.csv    对比表，可用 Excel 打开
outputs/summary.md        文字摘要
reports/index.html        自包含看板（无外部资源依赖）
```

## 数据集与类别

`datasets/campus_trash/data.yaml` 定义了 6 类：plastic、paper、cardboard、glass、metal、trash。
数据准备步骤、目录结构和标签格式见 `datasets/campus_trash/README.md`。如果暂时没有数据集，用 demo 模式也能完整演示。

## 常见问题

- **没有 GPU 能跑吗？** 能。demo 完全不需要；full 用 CPU 也能跑，只是慢一些。
- **没有网络怎么办？** 用 demo 模式，所有产物都能离线生成和打开。
- **看板打开是空白？** 确认它是完整的 `reports/index.html`；本看板不加载任何 CDN 资源。
- **demo 模式的投放建议真实吗？** 它是教学示例，用来展示流程；真实结果要跑 full 模式并使用你自己的数据集。
- **怎么换成别的具体检测对象？** 修改 `datasets/campus_trash/data.yaml` 的类别并用 `--dataset` 指向新 YAML；投放建议表在 `src/yolo_starter/trash_rules.py`。
- **`full` 模式报缺包？** 运行 `python scripts/check.py` 查看提示，或安装 `requirements.txt`。

## 文档索引

- 设计说明：`docs/DESIGN.md`
- 5 分钟讲稿与答辩 Q&A：`docs/WALKTHROUGH.md`
- 验收清单：`docs/CHECKLIST.md`
- 实施计划：`docs/superpowers/plans/2026-11-23-yolo-visual-starter.md`
