# 发布文案包（LinkedIn / 公众号 / 路演）

> 原则：不讲「界面长什么样」，只讲**开发者用它完成了什么**；每个数字都能在仓库里找到来源。

## 一、中文主文案（LinkedIn / 朋友圈 / 课程群）

标题：**我用 45 秒讲清了一个 YOLOv8 项目能替你做完什么**

正文：

很多 OpenCV / YOLO 课程作业卡在同一处：模型跑通了，但别人 3 秒看不懂它能干什么。

所以这次我没做「功能介绍」，而是按「问题 → 方案 → Demo → A/B 结果 → 复现入口」重做了 `yolo-starter` 的宣传片：

- **问题**：一张杂乱桌面照片，哪些是垃圾、该投哪个桶？
- **方案**：不是 notebook，是 `configs/experiments.json → run_all.py → demo/full → outputs + reports` 一条能跑、能测、能讲的链路。
- **Demo**：真实 YOLOv8 推理（`yolov8n` / CPU / conf 0.25），框出 `tv 0.94`、`bottle 0.72`、`bowl 0.47`、`book 0.25`，规则表再给出投放建议；人、猫这类非垃圾对象不硬凑结果。
- **A/B 结果**：密集书架 `0 → 87 → 13` 本；固定 80 类 `3 类` vs 开放词汇 `10 类 20 件`；坚果微调 40 epoch 的 `mAP@0.50 0.80`（真实训练曲线）。
- **如何开始**：`python scripts/run_all.py --mode demo --open` —— 零 GPU、零联网、零第三方依赖，103 个 unittest 与离线看板一起产出。

有一处我特意没含糊：`outputs/results.json` 里的 E1~E3 指标是 demo 模式的教学示例值，看板和视频里都保留「演示数据」标记，没有拿它冒充真实精度。片中的检测框、置信度、训练曲线全部来自本地实跑产物，来源文件在画面右下角逐镜头标注。

片子本身也是可编辑的：时间线在 `studio/index.html`，实测数据在 `data.js`，改一行就改一帧；15s / 45s / 90s 三个版本共用同一份工程。

仓库：https://github.com/Gubao-tian/yolo-starter

#YOLOv8 #ComputerVision #开源项目 #机器视觉 #Ultralytics #课程设计

## 二、英文短版（LinkedIn International）

**I rebuilt my YOLOv8 project's promo around one question: what can a developer actually finish with it?**

45 seconds, five beats: problem → pipeline → real inference → A/B evidence → reproduce.

- Real `yolov8n` detections on COCO photos: `tv 0.94`, `bottle 0.72`, `bowl 0.47`, `book 0.25` — boxes and scores produced by the model, not by hand.
- A/B evidence from the repo: dense-shelf `0 → 87 → 13` books; fixed 80 classes `3` vs open-vocabulary `10 classes / 20 items`; a 40-epoch fine-tune reaching `mAP@0.50 0.80`.
- One command to reproduce, no GPU and no network: `python scripts/run_all.py --mode demo --open`.

The demo-mode metrics are explicitly labelled as teaching samples — I did not present them as measured accuracy. The video is editable HTML: timeline in `index.html`, measured data in `data.js`, 15s/45s/90s cuts from one source.

https://github.com/Gubao-tian/yolo-starter

## 三、45 秒路演口播稿（与视频时间码对齐）

| 时间 | 口播 |
| --- | --- |
| 0–4s | 「如果给你一张这样的照片，你能不能三秒内说清哪些是垃圾、各投哪个桶？」 |
| 4–11s | 「所以这个项目不是一段 notebook，它把实验配置、一键脚本、双模式和报告看板串成一条链路。」 |
| 11–17s | 「这是真实跑出来的检测框：电视 0.94、塑料瓶 0.72、碗 0.47、书本 0.25，坐标和置信度都来自模型输出。」 |
| 17–22s | 「YOLO 只负责是什么、在哪；投哪个桶交给规则表。人和猫不会产生投放建议。」 |
| 22–27s | 「网页里上传自己的照片就能识别，四种模式，从固定 80 类到开放词汇。」 |
| 27–32s | 「同一张饭盒图，固定词表只认出 3 类，换成开放词汇是 10 类 20 件。」 |
| 32–37s | 「密集书架更直观：默认 0 本，切片推理 87 本，阈值调到 0.35 剩 13 本；右边是 40 轮微调的真实曲线，mAP@0.50 到 0.80。」 |
| 37–41s | 「零 GPU、零联网、一条命令就能复现，还有 103 个测试和离线看板。」 |
| 41–45s | 「仓库地址在屏幕上。45 秒看完，5 分钟能复述，一条命令能复现。」 |

## 四、评论区 / 置顶补充

- 视频里所有检测框坐标与置信度：`tools/generate_evidence.py` 用 `weights/yolov8n.pt` 实跑生成，结果在 `assets/detections.json`。
- 训练曲线来源：`runs/exp_nuts/results.csv`（Ultralytics 真实训练日志）。
- demo 模式的 E1~E3 指标是教学示例值，真实精度请跑 `--mode full` 并使用自己的数据集。
- 参考片只用于学习镜头结构，成片未使用任何他人画面、音乐或人物形象。

## 五、发布前检查清单

- [ ] 视频上传到仓库 `docs/promo/`，README 相对链接可打开
- [ ] 封面 PNG / GIF 已生成并贴入 README 顶部
- [ ] 三问（解决了什么 / 为什么值得试 / 如何开始）在 README 与视频里答案一致
- [ ] 所有数字均可在仓库文件中溯源
- [ ] 未把 demo 示例值写成实测精度
- [ ] 15s / 90s 版本与 45s 版使用同一份时间线，无文案冲突
