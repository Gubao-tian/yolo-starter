"""一键入口：从实验配置到看板的完整流水线。

无需安装包即可运行 demo 模式；full 模式需先安装 requirements.txt。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_starter import pipeline  # noqa: E402  路径注入后才能导入


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 YOLOv8 入门实验，生成看板与对比产物",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/run_all.py --mode demo --open\n"
            "  python scripts/run_all.py --mode full --dataset datasets/campus_trash/data.yaml --epochs 20\n"
        ),
    )
    parser.add_argument("--mode", choices=("demo", "full"), default="demo",
                        help="demo=零依赖示例数据；full=真实 YOLOv8（默认 demo）")
    parser.add_argument("--experiments", default=None,
                        help="实验编号，逗号分隔，如 E1,E2,E3；默认用配置中的 fast_set")
    parser.add_argument("--dataset", default=None,
                        help="数据集 YAML 名称或路径；默认读取实验配置")
    parser.add_argument("--device", default=None,
                        help="推理设备，如 cpu / cuda:0；默认自动检测")
    parser.add_argument("--epochs", type=int, default=0,
                        help="可选训练轮数；0 表示只验证与推理（默认 0）")
    parser.add_argument("--limit", type=int, default=3,
                        help="full 模式保存检测结果图的样本数量（默认 3）")
    parser.add_argument("--config", default=None, help="自定义实验配置 JSON 路径")
    parser.add_argument("--root", default=None,
                        help="自定义项目根目录（产物写入位置），默认使用脚本所在项目根目录")
    parser.add_argument("--open", action="store_true", help="生成后自动打开看板")
    return parser


def _display_path(raw: str) -> str:
    path = Path(raw)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_ids = None
    if args.experiments:
        experiment_ids = [item.strip() for item in args.experiments.split(",") if item.strip()]
        if not experiment_ids:
            print("运行失败: --experiments 未包含有效编号", file=sys.stderr)
            return 2

    try:
        result = pipeline.run(
            mode=args.mode,
            experiment_ids=experiment_ids,
            dataset=args.dataset,
            device=args.device,
            epochs=args.epochs,
            root=args.root,
            open_browser=args.open,
            limit=args.limit,
            config_path=args.config,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 2

    for label, raw_path in result["artifacts"].items():
        print(f"已生成: {_display_path(raw_path)}  [{label}]")
    print(f"实验组数: {len(result['rows'])}")
    print(f"自动结论: {len(result['conclusions'])} 条")
    if result["warnings"]:
        print(f"运行提示: {len(result['warnings'])} 条（详见 outputs/summary.md）")
    source = str(result.get("source_note", ""))
    print(f"数据来源: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
