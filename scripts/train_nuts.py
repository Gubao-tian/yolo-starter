"""用人工标注的坚果框微调 YOLOv8n，让模型能检出饭盒里的坚果。

用法：
  python scripts/train_nuts.py --build-only        # 只生成数据集
  python scripts/train_nuts.py --epochs 30         # 生成数据集并训练
  python scripts/train_nuts.py --reuse-data --epochs 30  # 复用已生成数据集

这是小样本演示：标注来自一张饭盒图 + 数据增强，用于课堂演示
"标注→训练→识别"完整链路；要通用需更多不同场景的标注图。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_starter import nut_dataset, yolo_backend  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=str(ROOT / "datasets" / "nuts_real" / "annotations.json"))
    ap.add_argument("--data-out", default=str(ROOT / "datasets" / "nuts_real" / "generated"))
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolov8n.pt"))
    ap.add_argument("--target", default=str(ROOT / "weights" / "yolov8n-nuts.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-crops", type=int, default=10)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--reuse-data", action="store_true")
    return ap.parse_args()


def count_nuts(model, image: str, imgsz: int, conf: float = 0.25) -> int:
    """统计模型在图片上检出的 nut 类框数。"""
    result = model.predict(source=image, imgsz=imgsz, conf=conf,
                           device="cpu", verbose=False)[0]
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None) or []
    return sum(1 for b in boxes
               if str(names.get(int(b.cls[0]), "")).lower() == "nut")


def main() -> None:
    args = parse_args()
    data_yaml = str(Path(args.data_out) / "data.yaml")

    if args.reuse_data and Path(data_yaml).exists():
        print(f"[1/3] 复用数据集：{data_yaml}")
    else:
        ann = nut_dataset.load_annotations(args.annotations)
        src = ROOT / ann["source"]
        if Path(args.data_out).exists():
            shutil.rmtree(args.data_out)
        print(f"[1/3] 从标注生成数据集：{ann['source']}，{len(ann['boxes'])} 个框")
        stats = nut_dataset.build([(src, ann["boxes"])], args.data_out,
                                  n_crops=args.n_crops)
        print(f"      train={stats['train']} val={stats['val']} boxes={stats['boxes']}")

    if args.build_only:
        return

    print(f"[2/3] 微调 {args.epochs} 轮（CPU，可能需要几分钟）")
    spec = {"id": "nuts", "model": args.weights, "imgsz": args.imgsz,
            "conf": 0.25, "iou": 0.7}
    yolo_backend.run_training(
        spec, data_yaml, epochs=args.epochs,
        project=str(ROOT / "runs"), device="cpu")
    weights_dir = ROOT / "runs" / f"exp_{spec['id']}" / "weights"
    candidates = [weights_dir / "best.pt", weights_dir / "last.pt"]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise RuntimeError(f"训练未产出权重：{weights_dir}")
    shutil.copy2(candidates[0], args.target)
    print(f"      权重已保存：{args.target}")

    from ultralytics import YOLO
    base = YOLO(args.weights)
    tuned = YOLO(args.target)
    test_img = ROOT / "test_images" / "selected" / "02_厨余_蔬菜食物.jpg"
    print("[3/3] 对比（饭盒图 conf=0.25）：")
    print(f"      原模型   ：{count_nuts(base, str(test_img), args.imgsz)} 颗")
    print(f"      微调模型 ：{count_nuts(tuned, str(test_img), args.imgsz)} 颗")


if __name__ == "__main__":
    main()
