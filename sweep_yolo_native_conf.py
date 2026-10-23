import csv
from pathlib import Path
from ultralytics import YOLO

RUN = Path('/workspace/runs/baseline_genyes8/20261019_s8_finetune')
DATA = '/workspace/datasets/genyes8_20261019/split_63_7_30_grouped/data.yaml'
CKPT = RUN / 'train_ddp/weights/best.pt'
rows = []
for conf in [0.1, 0.2, 0.3, 0.4]:
    model = YOLO(str(CKPT))
    metrics = model.val(data=DATA, split='test', device='', imgsz=640, batch=16,
                        workers=4, conf=conf, iou=0.7, max_det=300, plots=False,
                        project=str(RUN), name=f'native_conf_{conf:g}')
    row = {'conf': conf}
    row.update({k: float(v) for k, v in metrics.results_dict.items()})
    rows.append(row)
    print(row, flush=True)
out = RUN / 'native_confidence_sweep.csv'
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print('WROTE', out, flush=True)
