"""环境自检：确认 demo 模式始终可用，并提示 full 模式所需依赖。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_starter import pipeline, yolo_backend  # noqa: E402


def main() -> int:
    env = pipeline.environment_info()
    status = yolo_backend.backend_status()

    print("=== YOLOv8 入门实验 · 环境自检 ===")
    print(f"Python: {env['python']}")
    print(f"平台: {env['platform']}")
    print(f"Ultralytics: {status.get('ultralytics_version') or '未安装'}")
    print(f"Torch: {env.get('torch_version', '未安装')}")
    print(f"可用设备: {env['device']}")
    print("demo 模式可用（零第三方依赖、无需联网）:")
    print("  python scripts/run_all.py --mode demo --open")
    if not status["ultralytics_available"]:
        print("full 模式需先安装依赖:")
        print("  python -m pip install -r requirements.txt")
    else:
        print("full 模式可用:")
        print("  python scripts/run_all.py --mode full --dataset datasets/campus_trash/data.yaml --epochs 20")
        print("如需在新环境重装依赖:")
        print("  python -m pip install -r requirements.txt")
    print("单元测试:")
    print("  python -m unittest discover -s tests -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
