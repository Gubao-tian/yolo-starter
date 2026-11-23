"""一键流水线：把四项能力串成一条可复现的链路。

    configs/experiments.json
        -> demo 示例数据 / full 真实 YOLOv8 验证
        -> metrics 归一化与对比
        -> visualise 图表 + report 看板
        -> outputs/* + reports/index.html

CLI 入口见 scripts/run_all.py。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import platform
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

from . import metrics, report, samples, visualize, yolo_backend
from .experiments import load_config
from .paths import ProjectPaths

CSV_FIELDS = [
    "id", "name", "model", "imgsz", "conf",
    "map50", "map5095", "precision", "recall", "ms_per_image", "params_m",
    "delta_map50", "delta_map5095", "delta_ms_per_image", "is_baseline",
]

COMMANDS = [
    "# 1) 环境自检",
    "python scripts/check.py",
    "# 2) 单元测试",
    "python -m unittest discover -s tests -v",
    "# 3) 零依赖演示（无需 GPU / 无需联网）",
    "python scripts/run_all.py --mode demo --open",
    "# 4) 真实 YOLOv8 实验（准备 datasets/campus_trash 数据集后）",
    "python scripts/run_all.py --mode full --dataset datasets/campus_trash/data.yaml --epochs 20",
    "# 5) 本地预览看板",
    "python -m http.server 8000 --directory reports",
]


def environment_info() -> dict:
    """报告 Python 环境与依赖可用性；demo 模式不导入 torch。"""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ultralytics_available": importlib.util.find_spec("ultralytics") is not None,
        "torch_available": importlib.util.find_spec("torch") is not None,
        "device": "cpu",
    }
    if info["ultralytics_available"]:
        try:
            from importlib.metadata import version
            info["ultralytics_version"] = version("ultralytics")
        except Exception:
            info["ultralytics_version"] = "unknown"
    else:
        info["ultralytics_version"] = "未安装（full 模式需要）"
    if info["torch_available"]:
        try:
            from importlib.metadata import version
            info["torch_version"] = version("torch")
        except Exception:  # 元数据缺失时也不阻断流水线
            info["torch_version"] = "unknown"
    info["device"] = "cpu"
    return info


def full_mode_guard(dataset: str, epochs: int) -> str | None:
    """校验 full 模式数据集并返回未训练权重提示（无提示则返回 None）。"""
    candidate = Path(dataset)
    if candidate.suffix.lower() in (".yaml", ".yml") and not candidate.exists():
        raise FileNotFoundError(
            f"数据集配置不存在: {dataset}；请按 datasets/campus_trash/README.md 准备图片与标签"
        )
    if epochs == 0:
        return (
            "当前 epochs=0，使用的是未针对本数据集训练的权重；指标仅供参考，"
            "建议加 --epochs 20 后再作为实验结论。"
        )
    return None


def _find_dataset_images(paths: ProjectPaths, dataset: str, limit: int) -> list[Path]:
    """在常见位置寻找可用于推理的样本图。"""
    candidates = [
        paths.root / "datasets",
        Path.cwd() / "datasets",
        Path.home() / "datasets",
    ]
    custom = Path(dataset)
    if custom.suffix.lower() in (".yaml", ".yml") and custom.exists():
        candidates.insert(0, custom.parent.parent)
        candidates.insert(0, custom.parent)
    for candidate in candidates:
        found = yolo_backend.collect_image_paths(candidate, limit=limit)
        if found:
            return found
    return []


def _real_visual(paths: ProjectPaths, spec: dict, dataset: str, device: str,
                 limit: int, warnings: list[str]) -> str:
    images = _find_dataset_images(paths, dataset, limit)
    if not images:
        warnings.append("未找到可用于推理的数据集图片，检测框为示意图（非真实推理结果）。")
        return visualize.detection_svg(samples.demo_detections())
    try:
        written = yolo_backend.run_inference(spec, images, paths.outputs / "detections", device=device)
    except Exception as exc:
        warnings.append(f"推理阶段失败：{exc}；检测框退回示意图。")
        written = []
    uris = [visualize.image_data_uri(p) for p in written]
    if not uris:
        warnings.append("推理未产出图片，检测框为示意图（非真实推理结果）。")
        return visualize.detection_svg(samples.demo_detections())
    return "".join(
        f'<figure style="margin:0 0 12px"><img src="{uri}" alt="YOLO 检测结果 {i + 1}"/>'
        + visualize.caption_html(f"样本 {i + 1}：{images[i].name if i < len(images) else ''}") + "</figure>"
        for i, uri in enumerate(uris)
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            m = row["metrics"]
            writer.writerow({
                "id": row["id"], "name": row["name"], "model": row["model"],
                "imgsz": row["imgsz"], "conf": row["conf"],
                "map50": m["map50"], "map5095": m["map5095"],
                "precision": m["precision"], "recall": m["recall"],
                "ms_per_image": m["ms_per_image"], "params_m": m["params_m"],
                "delta_map50": row["delta_map50"],
                "delta_map5095": row["delta_map5095"],
                "delta_ms_per_image": row["delta_ms_per_image"],
                "is_baseline": row["is_baseline"],
            })


def _write_summary(path: Path, context: dict) -> None:
    lines = [
        "# YOLOv8 入门实验摘要",
        "",
        f"- 模式：`{context['mode']}`",
        f"- 数据集：`{context['dataset']}`",
        f"- 生成时间：{context['generated_at']}",
        f"- 数据来源：{context['source_note']}",
        "",
        "## 指标对比",
        "",
        "| 实验 | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | ms/图 | ΔmAP@0.50:0.95 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in context["rows"]:
        m = row["metrics"]
        lines.append(
            f"| {row['id']} {row['name']} | {m['map50']:.3f} | {m['map5095']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['ms_per_image']:.1f} | "
            f"{row['delta_map5095']:+.3f} |"
        )
    lines += ["", "## 自动结论", ""]
    lines += [f"- {item}" for item in context["conclusions"]]
    lines += ["", "## 运行提示", ""]
    lines += [f"- {item}" for item in context["warnings"]] or ["- 无"]
    lines += ["", "## 复现命令", "", "```bash", *context["commands"], "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(mode: str = "demo", experiment_ids: list[str] | None = None,
        dataset: str | None = None, device: str | None = None,
        epochs: int = 0, root: str | Path | None = None,
        open_browser: bool = False, limit: int = 3,
        config_path: str | Path | None = None) -> dict:
    """执行整条流水线，返回产物路径与结构化结果。"""
    if mode not in ("demo", "full"):
        raise ValueError(f"未知模式: {mode}（可选 demo / full）")

    paths = ProjectPaths(root).ensure()
    env = environment_info()
    resolved_device = device or env["device"]
    chosen_config = Path(config_path) if config_path else (
        paths.config_file if paths.config_file.exists() else None
    )
    cfg = load_config(chosen_config)
    specs = list(cfg.select(experiment_ids))
    dataset_name = dataset or cfg.dataset["default"]
    warnings: list[str] = []
    raw_results: dict[str, dict] = {}
    table_specs: list[Any] = list(specs)

    if mode == "demo":
        raw_results = samples.demo_results([s.id for s in specs])
        source_note = samples.demo_note()
        visual_html = visualize.detection_svg(samples.demo_detections())
        classification_html = visualize.classification_panel_html(samples.demo_detections())
        training_info = None
    else:
        if not env["ultralytics_available"]:
            raise RuntimeError(
                "full 模式需要 ultralytics。请先执行：pip install -r requirements.txt"
            )
        guard_warning = full_mode_guard(dataset_name, epochs)
        if guard_warning:
            warnings.append(guard_warning)
        source_note = (
            f"真实 YOLOv8 验证结果：在 {dataset_name} 上按 configs/experiments.json "
            "逐组调用 model.val() 得到。首次运行会下载权重；数据集需按 datasets/campus_trash/README.md 自行准备。"
        )
        for spec in specs:
            raw_results[spec.id] = yolo_backend.run_validation(
                spec.as_dict(), dataset_name, device=resolved_device
            )
        training_info = None
        if epochs > 0:
            try:
                baseline = cfg.baseline_spec()
                saved = yolo_backend.run_training(
                    baseline.as_dict(), dataset_name, epochs=epochs,
                    device=resolved_device, project=str(paths.outputs / "runs"),
                )
                weights = Path(saved) / "weights" / "best.pt"
                trained = {
                    **baseline.as_dict(), "id": f"{baseline.id}-T",
                    "name": f"训练 {epochs} epoch 后", "model": str(weights),
                    "hypothesis": "同一配置下观察训练带来的精度变化",
                }
                raw_results[trained["id"]] = yolo_backend.run_validation(
                    trained, dataset_name, device=resolved_device
                )
                table_specs.append(trained)
                training_info = {"epochs": epochs, "save_dir": str(saved), "weights": str(weights)}
            except Exception as exc:
                warnings.append(f"可选训练步骤未执行：{exc}")
        visual_html = _real_visual(paths, specs[-1].as_dict(), dataset_name,
                                   resolved_device, limit, warnings)
        classification_html = ""
        real_images = _find_dataset_images(paths, dataset_name, limit)
        if not real_images:
            warnings.append("未找到数据集图片，投放建议表为空。")
        else:
            try:
                detected = yolo_backend.collect_detections(
                    specs[-1].as_dict(), real_images, device=resolved_device
                )
                if detected:
                    classification_html = visualize.classification_panel_html(detected)
                else:
                    warnings.append("真实推理未检出垃圾对象，投放建议表为空。")
            except Exception as exc:
                warnings.append(f"投放建议生成失败：{exc}")

    rows = metrics.build_comparison(table_specs, raw_results, cfg.baseline)
    conclusions = metrics.derive_conclusions(rows)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    charts = {
        "mAP@0.50:0.95（越高越好）": visualize.bar_svg(
            [r["metrics"]["map5095"] for r in rows], [r["id"] for r in rows],
            "mAP@0.50:0.95", unit="", color="#2563eb"),
        "mAP@0.50（越高越好）": visualize.bar_svg(
            [r["metrics"]["map50"] for r in rows], [r["id"] for r in rows],
            "mAP@0.50", unit="", color="#7c3aed"),
        "单图推理耗时 ms（越低越好）": visualize.bar_svg(
            [r["metrics"]["ms_per_image"] for r in rows], [r["id"] for r in rows],
            "单图推理耗时", unit="ms", color="#dc2626"),
    }
    context = {
        "mode": mode,
        "generated_at": generated_at,
        "dataset": dataset_name,
        "baseline": cfg.baseline,
        "source_note": source_note,
        "rows": rows,
        "conclusions": conclusions,
        "charts": charts,
        "visual_html": visual_html,
        "classification_html": classification_html,
        "env": env,
        "commands": COMMANDS,
        "warnings": warnings,
    }

    results_path = paths.outputs / "results.json"
    csv_path = paths.outputs / "comparison.csv"
    summary_path = paths.outputs / "summary.md"
    report_path = report.save_report(context, paths.report_file)

    results_path.write_text(json.dumps({
        "schema_version": 1,
        "mode": mode,
        "generated_at": generated_at,
        "dataset": dataset_name,
        "baseline": cfg.baseline,
        "source_note": source_note,
        "environment": env,
        "device": resolved_device,
        "experiments": rows,
        "raw_results": raw_results,
        "conclusions": conclusions,
        "warnings": warnings,
        "training": training_info,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_summary(summary_path, context)

    artifacts = {
        "results": str(results_path),
        "comparison": str(csv_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }
    if open_browser:
        webbrowser.open(report_path.as_uri())

    return {
        "mode": mode,
        "root": str(paths.root),
        "dataset": dataset_name,
        "source_note": source_note,
        "rows": rows,
        "conclusions": conclusions,
        "warnings": warnings,
        "artifacts": artifacts,
    }
