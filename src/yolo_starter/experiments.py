"""实验矩阵的定义、加载与校验。

对比实验的可信度取决于"受控"：同一数据集、同一指标口径，只改少数变量。
本模块保证配置文件里的实验定义本身是合法且可解析的。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .paths import ProjectPaths


@dataclass(frozen=True)
class Experiment:
    """一组受控实验的完整参数。"""

    id: str
    name: str
    model: str
    imgsz: int
    conf: float
    iou: float
    hypothesis: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: dict
    baseline: str
    fast_set: tuple[str, ...]
    experiments: tuple[Experiment, ...]

    def select(self, ids: Iterable[str] | None = None) -> tuple[Experiment, ...]:
        """按给定顺序返回实验；未指定时使用快速实验集。"""
        wanted = list(ids) if ids is not None else list(self.fast_set)
        by_id = {spec.id: spec for spec in self.experiments}
        unknown = [item for item in wanted if item not in by_id]
        if unknown:
            raise ValueError(f"未知实验编号: {', '.join(unknown)}")
        return tuple(by_id[item] for item in wanted)

    def baseline_spec(self) -> Experiment:
        return self.select([self.baseline])[0]


def _validate(payload: dict) -> dict:
    for key in ("dataset", "baseline", "fast_set", "experiments"):
        if key not in payload:
            raise ValueError(f"实验配置缺少字段: {key}")
    if not isinstance(payload["experiments"], list) or not payload["experiments"]:
        raise ValueError("experiments 必须是非空列表")

    seen: set[str] = set()
    for item in payload["experiments"]:
        spec = Experiment(**item)
        if spec.id in seen:
            raise ValueError(f"实验编号重复: {spec.id}")
        seen.add(spec.id)
        if not isinstance(spec.imgsz, int) or spec.imgsz <= 0:
            raise ValueError(f"{spec.id}: imgsz 必须为正整数")
        for field in ("conf", "iou"):
            value = getattr(spec, field)
            if not 0 < value <= 1:
                raise ValueError(f"{spec.id}: {field} 必须落在 (0, 1]")
        if not spec.model.endswith(".pt"):
            raise ValueError(f"{spec.id}: model 必须是 .pt 权重文件名")
        if not spec.hypothesis.strip():
            raise ValueError(f"{spec.id}: hypothesis 不能为空")

    if payload["baseline"] not in seen:
        raise ValueError(f"baseline 未出现在实验列表中: {payload['baseline']}")
    for item in payload["fast_set"]:
        if item not in seen:
            raise ValueError(f"fast_set 含未知实验编号: {item}")
    return payload


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """从 JSON 加载实验矩阵；默认读取 configs/experiments.json。"""
    file_path = Path(path) if path is not None else ProjectPaths().config_file
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload = _validate(payload)
    specs = tuple(Experiment(**item) for item in payload["experiments"])
    return ExperimentConfig(
        dataset=payload["dataset"],
        baseline=payload["baseline"],
        fast_set=tuple(payload["fast_set"]),
        experiments=specs,
    )
