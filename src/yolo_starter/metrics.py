"""指标归一化、对比表构建与自动结论生成。

结论由数据自动推导，而不是手写，避免"结论与表格对不上"。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

METRIC_KEYS = (
    "map50",
    "map5095",
    "precision",
    "recall",
    "ms_per_image",
    "params_m",
)
RATIO_KEYS = frozenset({"map50", "map5095", "precision", "recall"})
DELTA_KEYS = ("map50", "map5095", "ms_per_image")


def _to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if result != result:  # NaN
        return 0.0
    return result


def normalize(raw: Mapping[str, Any] | None) -> dict[str, float]:
    """把任意来源的指标字典规整为六个 float 指标，缺失/非法值记 0。"""
    raw = raw if isinstance(raw, Mapping) else {}
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = _to_float(raw.get(key))
        if key in RATIO_KEYS:
            value = min(1.0, max(0.0, value))
        else:
            value = max(0.0, value)
        out[key] = value
    return out


def _spec_dict(spec: Any) -> dict:
    if hasattr(spec, "as_dict"):
        return spec.as_dict()
    return dict(spec)


def build_comparison(specs: Any, results: Mapping[str, Mapping[str, Any]], baseline_id: str) -> list[dict]:
    """生成对比行：实验参数 + 归一化指标 + 相对基线的差值。"""
    normalized = {_spec_dict(spec)["id"]: normalize(results.get(_spec_dict(spec)["id"]))
                  for spec in specs}
    base = normalized.get(baseline_id, {key: 0.0 for key in METRIC_KEYS})

    rows: list[dict] = []
    for spec in specs:
        info = _spec_dict(spec)
        sid = info["id"]
        current = normalized[sid]
        row = {
            "id": sid,
            "name": info.get("name", sid),
            "model": info.get("model", ""),
            "imgsz": info.get("imgsz", 0),
            "conf": info.get("conf", 0.0),
            "hypothesis": info.get("hypothesis", ""),
            "metrics": current,
            "is_baseline": sid == baseline_id,
        }
        for key in DELTA_KEYS:
            row[f"delta_{key}"] = round(current[key] - base[key], 6)
        rows.append(row)
    return rows


def format_metric(value: float, digits: int = 3) -> str:
    return f"{_to_float(value):.{digits}f}"


def format_delta(value: float, digits: int = 3) -> str:
    return f"{_to_float(value):+.{digits}f}"


def derive_conclusions(rows: list[dict]) -> list[str]:
    """从对比行推导可复述的结论，包含精度、速度与权衡。"""
    if not rows:
        return ["暂无可用实验结果，请先运行流水线生成数据。"]

    best = max(rows, key=lambda r: r["metrics"]["map5095"])
    fastest = min(
        rows,
        key=lambda r: r["metrics"]["ms_per_image"] if r["metrics"]["ms_per_image"] > 0 else float("inf"),
    )
    baseline = next((r for r in rows if r.get("is_baseline")), rows[0])

    conclusions = [
        f"{best['id']}（{best['name']}）的 mAP@0.50:0.95 最高，为 "
        f"{format_metric(best['metrics']['map5095'])}；mAP@0.50 为 "
        f"{format_metric(best['metrics']['map50'])}。",
        f"{fastest['id']} 的单图推理耗时最低（{fastest['metrics']['ms_per_image']:.1f} ms/图），"
        "是速度最优配置。",
    ]

    if best["id"] != fastest["id"] and best["metrics"]["ms_per_image"] > 0:
        extra = best["metrics"]["ms_per_image"] - fastest["metrics"]["ms_per_image"]
        pct = 100.0 * extra / fastest["metrics"]["ms_per_image"]
        conclusions.append(
            f"精度最优的 {best['id']} 比最快的 {fastest['id']} 多耗时 "
            f"{extra:.1f} ms/图（+{pct:.0f}%），体现精度-速度权衡。"
        )

    if best["id"] != baseline["id"]:
        conclusions.append(
            f"相对基线 {baseline['id']}，{best['id']} 的 mAP@0.50:0.95 变化 "
            f"{format_delta(best['metrics']['map5095'] - baseline['metrics']['map5095'])}，"
            f"mAP@0.50 变化 "
            f"{format_delta(best['metrics']['map50'] - baseline['metrics']['map50'])}。"
        )

    high = [r for r in rows if r["conf"] >= 0.5]
    low = [r for r in rows if r["conf"] < 0.5]
    if high and low:
        high_p = sum(r["metrics"]["precision"] for r in high) / len(high)
        low_p = sum(r["metrics"]["precision"] for r in low) / len(low)
        high_r = sum(r["metrics"]["recall"] for r in high) / len(high)
        low_r = sum(r["metrics"]["recall"] for r in low) / len(low)
        if high_p > low_p and high_r < low_r:
            conclusions.append(
                f"提高 conf 阈值的一组 Precision 均值 {format_metric(high_p)} 高于低阈值组 "
                f"{format_metric(low_p)}，但 Recall 均值 {format_metric(high_r)} 低于 "
                f"{format_metric(low_r)}，符合 P-R 权衡预期。"
            )
    return conclusions
