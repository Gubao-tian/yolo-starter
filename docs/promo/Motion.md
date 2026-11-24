# Motion.md — OpenMotion 可编辑视频脚本

> 格式参照 `nexu-io/openmotion` 的 Motion.md 规范：Brand Kit（品牌与已确认事实）+ Design（视觉规则）+ Timeline（镜头/时间码/可替换素材）。
> 所有数值均来自 `Gubao-tian/yolo-starter` 仓库或本次实测产物，**禁止在制作过程中改写为未经核实的数字**。

## Brand Kit

| 项 | 值 |
| --- | --- |
| 项目名 | yolo-starter · 校园垃圾分类投放识别 / YOLOv8 可视化入门项目 |
| 仓库 | https://github.com/Gubao-tian/yolo-starter |
| 一句话定位 | 让第一次接触 YOLO 的人，用一条命令跑通「检测 → 决策 → 对比 → 复述」 |
| 目标观众 | 课程答辩评委、第一次接触 YOLO 的同学、开源仓库访客 |
| 语气 | 克制、工程化、可验证；用文件名和数据说话，不用形容词 |
| 主色 | 墨蓝 `#16202E`（文字）/ 云白 `#F6F8FA`（底）/ 工程绿 `#16A34A`（正向、可回收）/ 蓝 `#2563EB`（架构、工程）/ 琥珀 `#D97706`（演示数据警示） |
| 字体 | 中文 `PingFang SC / Microsoft YaHei / Noto Sans CJK SC`；数字与命令 `ui-monospace, Consolas, Menlo` |
| 声音 | 仅轻量 click / tick / whoosh，无商业音乐（规避版权） |

### 已确认事实清单（文案唯一来源）

| 事实 | 来源 |
| --- | --- |
| 真实推理：Ultralytics 8.4.216 / `yolov8n.pt` COCO 预训练 / CPU / imgsz 640 / conf 0.25 | `tools/generate_evidence.py` 本次实测 |
| 多物品图实测：`tv 0.943`、`bottle 0.722`、`bowl 0.465`、`book 0.254` | `assets/detections.json` |
| 网页 4 种模式：垃圾分类(80类) / 全物体提取 / 密集书本 / 坚果识别 | `README.md`、`assets/webapp.png` |
| 开放词汇：117 类词表，饭盒图实测 10 类 20 件 | `test_images/README.md` |
| 密集小目标：默认 0 本 → 切片推理 87 本 → conf 0.35 剩 13 本 | `test_images/README.md` |
| 坚果微调：40 epoch，mAP@0.50 `0.80009`、mAP@0.50:0.95 `0.70027`、Recall `0.95` | `runs/exp_nuts/results.csv` → `assets/training_curve.json` |
| 零依赖 demo 模式、103 个 unittest、离线看板 `reports/index.html`、5 分钟讲稿与验收清单 | `README.md`、`docs/DESIGN.md`、`docs/WALKTHROUGH.md` |
| 复现命令 `python scripts/run_all.py --mode demo --open` | `README.md` |

### 禁止项（Anti-goals）

- 不写「高精度 / 行业领先 / SOTA」等无法验证的表述。
- **不把 `outputs/results.json` 的 E1~E3 指标当作真实精度**：该文件 `mode=demo`，README 与看板均有「演示数据」标记；视频出现看板截图时必须同屏保留该标记。
- 不生成伪造的检测框、界面或结果图；核心镜头只用真实素材。
- 不使用参考片的画面、音乐、人物形象。

## Design

| 项 | 规格 |
| --- | --- |
| 画幅 | 1920×1080，30fps，安全区 96px；同时提供 1080×1920 竖版裁切参数（见 `studio/index.html` 的 `ASPECT`） |
| 字号 | H1 88 / H2 56 / 正文 36 / 大数字 120 / 字幕 34 / 命令 30（等宽） |
| 动效 | `cubic-bezier(.22,.9,.28,1)`，240–420ms；检测框 stroke-draw 320ms、stagger 140ms；数字 count-up 900ms |
| 转场 | 只允许四类：硬切、左推入、行级 stagger、数字 count-up；禁止旋转/粒子/光效 |
| 字幕 | 常驻底部安全区，每屏 ≤2 行；专有名词保留英文原文 |
| 数据标注 | 每个含数字的镜头右下角固定显示来源文件名（12–16px），保证可追溯 |

## Timeline

三个版本共用同一套场景组件，只切换时间线配置：`?v=15` / `?v=45`（默认）/ `?v=90`。

### 45s 标准版（课程推荐）

| # | ID | 时间码 | 场景 | 文案 | 素材 / 数据来源 | 可替换字段 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `hook` | 0–4s | 真实原图 + 问题压屏 | 一张照片里，哪些是垃圾？该投哪个桶？ | `assets/photo_01_multi.jpg` | 图片、问题文案 |
| 2 | `pipeline` | 4–11s | 四段架构依次点亮 | 不是 notebook，是一条能跑、能测、能讲的链路 | `docs/DESIGN.md` 双模式架构 | 节点名、节点数 |
| 3 | `detect` | 11–17s | 原图 → 真实检测框逐个弹出 | tv 0.94 / bottle 0.72 / bowl 0.46 / book 0.25 | `assets/detections.json`（框坐标 + 置信度） | 检测数据 JSON、图片 |
| 4 | `decision` | 17–22s | 投放建议四行 + 非垃圾对象 | YOLO 管「是什么、在哪」，规则表管「投哪个桶」 | `src/yolo_starter/trash_rules.py`、`detect_07_people.jpg` | 规则表条目 |
| 5 | `demo` | 22–27s | 真实网页界面 + 四模式标签 | 4 种模式：垃圾分类 / 全物体提取 / 密集书本 / 坚果识别 | `assets/webapp.png` | 界面截图、模式名 |
| 6 | `ab-vocab` | 27–32s | 左右对比条 | 固定 80 类 → 3 类；开放词汇 → 10 类 20 件 | `test_images/README.md` | 两侧数值 |
| 7 | `ab-dense` | 32–37s | 大数字 + 真实训练曲线 | 0 → 87 → 13 本；微调 40 epoch，mAP@0.50 0.80 | `test_images/README.md`、`assets/training_curve.json` | 曲线数据、数字 |
| 8 | `repro` | 37–41s | 看板截图 + 命令打字机 | 零 GPU · 零联网 · 一条命令 | `assets/dashboard.png`、`README.md` | 命令、徽标 |
| 9 | `cta` | 41–45s | 深底收束，CTA 静止 ≥2s | 解决了什么 / 为什么值得试 / 如何开始 + 仓库地址 | `README.md` | 仓库 URL、三问答案 |

### 15s 版（社媒快剪）

`hook(0–2)` → `detect(2–7)` → `ab-dense(7–11)` → `cta(11–15)`。去掉架构与规则表，只保留「真实检测框 + 真实数字 + 仓库地址」。

### 90s 版（路演 / README 置顶）

在 45s 版基础上插入：`why(11–19)` 课程四项能力「能看/能跑/能对比/能复述」、`webapp(30–40)` 上传照片真实推理逐步演示、`dual-mode(57–64)` demo/full 双模式与数据真实性边界、`tests(64–70)` 103 个 unittest 与离线看板、`faq(70–76)` 无 GPU / 无网络怎么办；`repro(76–82)` 复现入口，`cta(82–90)` CTA 延长至 8 秒，末尾静止停留 ≥3 秒。合计精确 90 秒。

## 可替换素材表

| 素材 | 路径 | 生成方式 | 替换方法 |
| --- | --- | --- | --- |
| 8 张真实原图 | `assets/photo_*.jpg` | 从仓库 `test_images/selected/` 逐字节复制 | 换同名文件 |
| 8 张真实检测图 | `assets/detect_*.jpg` | `yolov8n.pt` 推理 + Ultralytics `result.plot()` | 重跑 `tools/generate_evidence.py` |
| 检测框 JSON | `assets/detections.json` | 同上，含 `xywh_norm` 归一化坐标 | 重跑脚本或改 `SCENES[].data` |
| 训练曲线 | `assets/training_curve.json` | `tools/export_curve.py` 读 `runs/exp_nuts/results.csv` | 换 CSV 重跑 |
| 真实看板 | `assets/dashboard.png` | Chrome headless 截图 `reports/index.html` | 重新截图 |
| 真实网页 | `assets/webapp.png` | Chrome headless 截图本地 Flask 页面 | 重新截图 |

## 导出规格

| 项 | 值 |
| --- | --- |
| 主文件 | `export/yolo-starter-promo-45s.mp4`（H.264 / yuv420p / CRF 18 / 30fps / 1920×1080） |
| 衍生 | 15s、90s 两个版本；封面 GIF（前 6 秒、480px 宽、12fps） |
| 可编辑工程 | `studio/index.html`（时间线 + 场景源码同文件，改 `SCENES` 即改片） |
| 复现入口 | `README.md` 中「一键复现」段落，命令全部可执行 |

## QA 清单（对照课程「最佳实践」）

- [x] 核心镜头优先使用真实录屏与实测证据：8 张原图、8 张真实推理框、2 张真实界面截图、1 条真实训练曲线。
- [x] AI/代码生成仅用于辅助画面：渐变、网格、图标、遮罩由 CSS 生成，不伪造结果。
- [ ] 视频完成后同步更新 README 封面、封面 GIF 与复现入口 → 产物在 `publish/`，**需人工确认后再推送 GitHub**。
- [x] 观众看完能回答：解决了什么、为什么值得试、如何开始（CTA 三问逐条上屏）。
- [x] 全片不出现「界面长什么样」作为卖点，每个镜头都绑定一个开发者可完成的任务。
- [x] 所有数字右下角标注来源文件；demo 数据保留「演示数据」标记。
