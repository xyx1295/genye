"""Dependency-free structural verification for this lightweight bundle."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "pyproject.toml",
    "ultralytics/__init__.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/models/yolo/detect/train.py",
    "ultralytics/engine/trainer.py",
    "ultralytics/cfg/models/11/yolo11-seg.yaml",
    "configs/data_63_7_30_server.yaml",
    "rgbd_stacking_metrics_plan/yolo11_stacking_metrics.py",
    "train_portable.py",
    "evaluate_portable.py",
    "runs/20261019_s8_finetune/afss25/weights/best.pt",
    "runs/20261019_s8_finetune/afss25/results.csv",
    "runs/20261019_s8_finetune/completed_summary.json",
)
EXPECTED_BEST_SHA256 = "11111af195b4ef5af9e84836335b355b1916659abe42f40565878bfb8b561ef4"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))
    best = ROOT / "runs/20261019_s8_finetune/afss25/weights/best.pt"
    actual = sha256(best)
    if actual != EXPECTED_BEST_SHA256:
        raise SystemExit(f"best.pt SHA-256 mismatch: {actual}")
    for path in ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            compile(path.read_bytes(), str(path), "exec")
    print(f"BUNDLE_OK: {len(REQUIRED)} required files; best.pt SHA-256={actual}")


if __name__ == "__main__":
    main()
