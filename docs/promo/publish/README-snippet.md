# README 封面区增补片段（已同步到仓库 README）

> 对应课程要求「视频完成后同步更新 README / 封面 GIF / 复现入口」。
> **本片段已上线**：物料位于 `docs/promo/`，下方链接已按仓库实际位置调整并插入仓库 `README.md`。

## 物料布局

- 三版成片：`export/yolo-starter-promo-{15,45,90}s.mp4`
- 封面：`publish/cover.png` / `publish/cover.gif`
- 可编辑工程：`studio/index.html`（`?v=15|45|90` 切换版本）
- 脚本与实测素材：`VIDEO.md`、`Motion.md`、`assets/`

---

<!-- ↓↓↓ 从这里开始复制 ↓↓↓ -->

## 45 秒看懂这个项目

[![yolo-starter 宣传片封面](docs/promo/cover.png)](docs/promo/yolo-starter-promo-45s.mp4)

> 点击封面观看 45 秒宣传片（问题 → 方案 → Demo → A/B 结果 → GitHub CTA）。
> 完整可编辑工程、镜头脚本与实测素材见 [`docs/promo/`](docs/promo/)。

| 观众的三问 | 回答 |
| --- | --- |
| **解决了什么** | 一张照片 → YOLOv8 检测框 + 置信度 → 规则表给出投放建议，一条链路讲清楚 |
| **为什么值得试** | 真实实测 A/B（密集书本 `0 → 87 → 13` 本、开放词汇 `3 类 → 10 类 20 件`、微调 40 epoch `mAP@0.50 0.80`）+ 零依赖一键复现 + 5 分钟讲稿与验收清单 |
| **如何开始** | `python scripts/run_all.py --mode demo --open` —— 零 GPU、零联网、零第三方依赖 |

**一键复现**

```bash
git clone https://github.com/Gubao-tian/yolo-starter.git
cd yolo-starter
python scripts/check.py
python -m unittest discover -s tests -v
python scripts/run_all.py --mode demo --open
```

**关于数据真实性**：`outputs/results.json` 的 E1~E3 指标是 `demo` 模式的教学示例值，看板顶部有「演示数据」标记，**不是真实精度**；真实结果请用 `--mode full` 配合自己的数据集。宣传片中的检测框、置信度、训练曲线与 A/B 数字均来自真实推理/训练产物，来源文件在画面右下角逐镜头标注。

**多版本**：15s 社媒快剪 / 45s 标准版 / 90s 路演版共用同一份可编辑时间线，在 `studio/index.html` 中切换 `?v=15|45|90` 即可。

<!-- ↑↑↑ 复制到这里结束 ↑↑↑ -->
