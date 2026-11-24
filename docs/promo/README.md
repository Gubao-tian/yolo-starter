# yolo-starter 开源项目宣传视频 · 交付包

**目标仓库**：https://github.com/Gubao-tian/yolo-starter
**课程要求**：使用 rea-video × OpenMotion 制作高质量开源项目宣传视频 —— 先拆解优秀案例，再生成**可编辑**视频，让项目既能展示价值，也能证明真实能力。
**主交付**：45 秒可编辑 HTML 视频工程 + MP4，含 15s / 90s 衍生版、封面、字幕与发布物料。

---

## 一、按课程「推荐工作流」逐步对照

| 步骤 | 课程要求 | 本交付包中的产物 |
| --- | --- | --- |
| 1 选择参考视频 | 找一条欣赏的技术宣传片 | `VIDEO.md` §0–§2：技术产品发布 + 数据故事混合范式，9 镜头结构级拆解（含时间码、字幕、转场、声音落点） |
| 2 用 rea-video 拆解 | 提取镜头、节奏、字幕与转场 | `VIDEO.md` 按 VIDEO.md 格式组织；另附 rea-video 实测命令，拿到参考片后可生成逐帧对照报告 |
| 3 准备真实素材 | README、架构图、运行录屏、实验结果、A/B 结果 | `assets/`：8 张真实原图、8 张真实 YOLOv8 检测图、`detections.json`、真实训练曲线、真实看板截图、真实网页截图 |
| 4 用 OpenMotion 重建 | 替换为自己的品牌、文案和项目内容 | `Motion.md`（Brand Kit / Design / Timeline / 可替换素材）+ `studio/index.html`（可编辑工程与 Studio 时间线） |
| 5 导出并发布 | README 封面、GitHub、LinkedIn、路演展示 | `export/*.mp4`、`publish/cover.png`、`cover.gif`、`README-snippet.md`、`linkedin.md`、`captions-45s.srt` |

**课堂要求核对**：全片不展示「软件长什么样」作为卖点，每个镜头都绑定一个开发者可完成的任务；结尾 CTA 逐条回答「解决了什么 / 为什么值得试 / 如何开始」。

**最佳实践核对**：
- 核心镜头使用真实录屏与实测证据（检测框来自 `yolov8n.pt` 实跑，曲线来自 `runs/exp_nuts/results.csv`）；
- 代码生成只用于几何辅助元素（网格、遮罩、图标），不伪造结果画面；
- `publish/README-snippet.md` 留存 README 封面 / 封面 GIF / 复现入口的同步文本（**已同步到仓库 README**）；
- 每个含数字的镜头右下角标注来源文件；看板截图同屏保留「演示数据」标记。

---

## 二、目录结构

```
yolo-starter-promo/
├── VIDEO.md                    rea-video 格式参考片拆解 + 复刻计划 + 真实性边界
├── Motion.md                   OpenMotion 脚本：Brand Kit / Design / 三版时间线 / 素材表 / QA
├── studio/
│   ├── index.html              可编辑视频工程（Studio 时间线 + 9 类场景渲染器）
│   └── data.js                 实测数据模块（检测框 / 训练曲线 / 仓库信息）
├── assets/
│   ├── photo_*.jpg             8 张仓库自有 COCO 真实照片（逐字节复制）
│   ├── detect_*.jpg            8 张 Ultralytics 真实渲染的检测结果
│   ├── detections.json         真实框坐标 + 置信度（含归一化 xywh）
│   ├── training_curve.json     40 epoch 真实训练曲线
│   ├── dashboard.png           真实离线看板截图（含「演示数据」标记）
│   └── webapp.png              真实 Flask 网页截图（四模式 + 示例图）
├── tools/
│   ├── generate_evidence.py    真实 YOLOv8 推理 → 检测图 + JSON
│   ├── export_curve.py         真实训练日志 → 曲线 JSON
│   ├── build_data.py           JSON → studio/data.js
│   └── verify.py               交付包自检（10 项，含成片规格与数据真实性）
├── export/
│   ├── record.mjs              CDP 逐帧渲染 + WAV 节拍 + FFmpeg 流式合成 MP4
│   ├── yolo-starter-promo-15s.mp4   15.00s
│   ├── yolo-starter-promo-45s.mp4   45.00s（课程推荐标准版）
│   ├── yolo-starter-promo-90s.mp4   90.00s
│   ├── check-45s/ check-90s/  各场景抽查帧（视觉验收用）
│   └── studio_ui.png           Studio 编辑界面截图
└── publish/
    ├── cover.html / cover.png  README 封面
    ├── cover.gif               前 6 秒循环封面
    ├── captions-45s.srt        字幕
    ├── README-snippet.md       待贴回仓库 README 的封面区增补
    └── linkedin.md             LinkedIn / 公众号 / 路演口播稿
```

---

## 三、三个版本，一份工程

| 版本 | 时长 | 用途 | 打开方式 |
| --- | --- | --- | --- |
| 15s | 15s | 社媒快剪：问题 → 真实检测框 → A/B 数字 → CTA | `studio/index.html?v=15` |
| 45s | 45s | 课程推荐标准版（默认） | `studio/index.html?v=45` |
| 90s | 90s | 路演 / README 置顶：增加四项能力、双模式边界、测试与讲稿、FAQ | `studio/index.html?v=90` |

Studio 支持：空格播放/暂停、← → 跳 1 秒、点击时间线或场景列表跳转、`1/2/3` 切版本、字幕开关。
**任意时间点的画面由 `renderAt(t)` 确定性计算** —— 这是逐帧导出 MP4 与「改一行即改一帧」的基础。

---

## 四、从零复现（全部命令可执行）

环境：Windows / Python 3.10+ / Node.js 22+ / Chrome / FFmpeg；YOLO 实跑需项目本地 `.venv`（Ultralytics 8.4.216）。

```bash
# 1) 真实推理素材（在 yolo-starter 仓库根目录执行）
cd "E:\work  buddy\2026-11-23-10-44-32\yolo-starter"
.venv\Scripts\python.exe ..\..\yolo-starter-promo\tools\generate_evidence.py

# 2) 曲线与数据模块
python tools/export_curve.py
python tools/build_data.py

# 3) 预览可编辑工程（双击打开亦可）
start studio/index.html?v=45

# 4) 抽查各场景视觉
node export/record.mjs --v 45 --check

# 5) 导出 MP4（逐帧渲染 1350 帧 + 合成）
node export/record.mjs --v 45
node export/record.mjs --v 15 --v 90

# 6) 交付自检（10 项：文件齐全、真实数据、时间线时长、脚本语法、成片规格、demo 标注）
python tools/verify.py
```

FFmpeg 缺失时：`python -m pip install imageio-ffmpeg`，脚本会自动探测其内置二进制。
Chrome 路径或 FFmpeg 路径可用 `--chrome`、`--ffmpeg`、`--port` 覆盖。

导出采用 **image2pipe 流式送帧**：每帧 PNG 直接写入编码器 stdin，不落盘临时帧文件。
> 注：`export/frames-45s/` 是首版 45s 导出时留下的 1350 张逐帧中间图（约 481 MB），现行脚本已不再生成此类文件；它不影响任何交付物。若需释放空间，可在确认后手动删除该目录（本交付包不会自动批量删除文件）。

---

## 五、真实性声明（重要）

- **真实**：检测框坐标与置信度来自 `weights/yolov8n.pt` 对仓库自有 COCO 照片的实跑输出；训练曲线来自 `runs/exp_nuts/results.csv`；A/B 数字来自 `test_images/README.md` 的实测记录；界面截图来自本地真实运行的 Flask 网页与离线看板。
- **教学示例值，不是真实精度**：`outputs/results.json` 中 `mode=demo` 的 E1~E3 指标（mAP 0.512 / 0.596 / 0.571 等）来自 `samples.py` 内置示例，仓库 README 与看板均标注「演示数据」。本片**未**将其作为实测精度宣传，并在 `repro` 镜头同屏给出警示。
- **未提供参考视频源文件**：`VIDEO.md` 是结构级拆解，不是逐帧实测；文末给出 rea-video 命令，拿到参考片后可补做实测对照。
- **未使用他人素材**：参考片仅用于学习镜头结构，成片不含其画面、音乐、人物形象；音轨为脚本合成的轻量 click，无版权风险。

---

## 六、状态

| 项 | 状态 |
| --- | --- |
| 参考片拆解、Motion.md、可编辑工程、真实素材、封面 PNG/GIF、字幕、发布文案 | 已完成 |
| 三个成片 MP4 | 已完成：15.00s / 45.00s / 90.00s，均为 H.264 1920×1080 30fps yuv420p + AAC 节拍音轨 |
| README 封面 / 封面 GIF / 复现入口同步到 GitHub | **已完成**：物料位于 `docs/promo/`，仓库 README 封面区已上线 |

`publish/linkedin.md` 可作为社交平台发布文案；封面区增补文本留存于 `publish/README-snippet.md`。
