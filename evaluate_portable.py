"""Portable val/test deployment check with native and business metrics."""

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

from rgbd_stacking_metrics_plan.yolo11_stacking_metrics import attach_stacking_metrics


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs/20261019_s8_finetune/afss25/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on val or test with business metrics.")
    parser.add_argument("--data", required=True, help="Dataset YAML with RGB-D val/test paths.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--project", default=str(ROOT / "runs/deployment_tests"))
    parser.add_argument("--name", default="test")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.08, help="Business confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dirs = []
    model = YOLO(args.weights)
    attach_stacking_metrics(model, conf=args.conf, iou=0.5)
    model.add_callback("on_val_end", lambda validator: output_dirs.append(Path(validator.save_dir)))
    result = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        plots=True,
        project=args.project,
        name=args.name,
        exist_ok=False,
    )
    save_dir = output_dirs[-1]
    summary = {"weights": str(Path(args.weights).resolve()), "split": args.split, "native_metrics": {}}
    summary["native_metrics"] = {key: float(value) for key, value in result.results_dict.items()}
    for filename in ("stacking_metrics.csv", "stacking_metrics_v3.csv"):
        path = save_dir / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                summary[path.stem] = list(csv.DictReader(stream))[-1]
    summary_path = save_dir / "deployment_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DEPLOYMENT_TEST_COMPLETE: {summary_path}")


if __name__ == "__main__":
    main()
