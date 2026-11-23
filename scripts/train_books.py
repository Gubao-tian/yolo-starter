"""用密集推理的伪标签微调 YOLOv8，让单张图也能认出密集书本。

用法：
  python scripts/train_books.py --build-only      # 只生成数据集
  python scripts/train_books.py --epochs 10       # 生成数据集并训练

注意：这是自训练演示。伪标签来自预先训练好的检测器和同一张书架图，
适合课堂演示"数据→标注→训练→对比"的完整链路，不等于通用书籍检测模型。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolo_starter import book_dataset, detector, yolo_backend  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "test_images" / "09_密集书架_书本.png"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "books"))
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolov8n.pt"))
    ap.add_argument("--target", default=str(ROOT / "weights" / "yolov8n-books.pt"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--n-crops", type=int, default=4)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--real-dataset", default=None,
                    help="已有真实标注的 YOLO data.yaml 路径；提供后跳过伪标签步骤")
    return ap.parse_args()


def count_books(model, image: str, imgsz: int, conf: float = 0.25) -> int:
    result = model.predict(source=image, imgsz=imgsz, conf=conf,
                           device="cpu", verbose=False)[0]
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None) or []
    return sum(1 for b in boxes
               if str(names.get(int(b.cls[0]), "")).lower() == "book")


def main() -> None:
    args = parse_args()
    teacher = detector.load_model(args.weights, "cpu")

    if args.real_dataset:
        print(f"[1/3] 使用真实标注数据集：{args.real_dataset}")
        stats = {"yaml": args.real_dataset}
    else:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        sources = sorted(p for p in (ROOT / "test_images").rglob("*")
                         if p.is_file() and p.suffix.lower() in exts)
        print(f"[1/3] 伪标签推理：{len(sources)} 张图")
        pairs = [(p, detector.detect_dense_books(teacher, p, conf=0.15))
                 for p in sources]
        print(f"      共 {sum(len(d) for _, d in pairs)} 个框；"
              f"无书本图 {sum(1 for _, d in pairs if not d)} 张（作为负样本）")

        print(f"[2/3] 生成数据集：{args.out}")
        if Path(args.out).exists():
            shutil.rmtree(args.out)
        stats = book_dataset.build_dataset_multi(
            pairs, args.out, n_crops=args.n_crops, min_score=args.min_score)
        print(f"      train={stats['train']} val={stats['val']} "
              f"boxes={stats['boxes']} 负样本源={stats['negative_sources']}")
        if args.build_only:
            return

    print(f"[3/3] 微调 {args.epochs} 轮（CPU，可能需要几分钟）")
    spec = {"id": "books", "model": args.weights, "imgsz": args.imgsz,
            "conf": 0.25, "iou": 0.7}
    yolo_backend.run_training(
        spec, stats["yaml"], epochs=args.epochs,
        project=str(ROOT / "runs"), device="cpu")
    # 直接用约定的输出目录拼路径，避免 ultralytics 返回的长路径在 Windows 上 glob 失败
    weights_dir = ROOT / "runs" / f"exp_{spec['id']}" / "weights"
    candidates = [weights_dir / "best.pt", weights_dir / "last.pt"]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise RuntimeError(f"训练未产出权重：{weights_dir}")
    shutil.copy2(candidates[0], args.target)
    print(f"      权重已保存：{args.target}")

    print("对比（整图推理，conf=0.25）：")
    print(f"      原模型   ：{count_books(teacher, args.source, args.imgsz)} 本")
    tuned = detector.load_model(args.target, "cpu")
    print(f"      微调模型 ：{count_books(tuned, args.source, args.imgsz)} 本")


if __name__ == "__main__":
    main()
