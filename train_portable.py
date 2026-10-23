"""Portable RGB-D training entry point for the lightweight project bundle."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from rgbd_stacking_metrics_plan.yolo11_stacking_metrics import attach_stacking_metrics


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs/20261019_s8_finetune/afss25/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune the RGB-D model with business metrics enabled.")
    parser.add_argument("--data", required=True, help="Dataset YAML containing train/val/test RGB image paths.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_WEIGHTS),
        help="Checkpoint for fine-tuning, or a model YAML after adding/removing modules.",
    )
    parser.add_argument("--project", default=str(ROOT / "runs"), help="Output root directory.")
    parser.add_argument("--name", default="experiment", help="Run name under --project.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="", help="Ultralytics device value, e.g. 0, 0,1, cpu or empty for auto.")
    parser.add_argument("--train-sample-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=28975)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    attach_stacking_metrics(model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        train_sample_fraction=args.train_sample_fraction,
        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0001,
        warmup_epochs=3.0,
        amp=True,
        seed=args.seed,
        deterministic=True,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
