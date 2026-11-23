"""项目路径解析。

设计要点：所有路径都通过 ProjectPaths 暴露，并允许传入自定义 root。
这样流水线可以在临时目录中完整跑一遍（用于测试），不污染项目本身。
"""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """返回项目根目录（src/yolo_starter/paths.py 的上两级）。"""
    return Path(__file__).resolve().parents[2]


class ProjectPaths:
    """项目内各目录的统一入口。"""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else project_root()

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def config_file(self) -> Path:
        return self.configs / "experiments.json"

    @property
    def report_file(self) -> Path:
        return self.reports / "index.html"

    def ensure(self) -> "ProjectPaths":
        for directory in (self.outputs, self.reports, self.assets):
            directory.mkdir(parents=True, exist_ok=True)
        return self
