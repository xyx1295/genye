import csv
import json
from pathlib import Path
from ultralytics import YOLO
from rgbd_stacking_metrics_plan.yolo11_stacking_metrics import attach_stacking_metrics

RUN = Path('/workspace/runs/baseline_genyes8/20261019_s8_finetune')
DATA = '/workspace/datasets/genyes8_20261019/split_63_7_30_grouped/data.yaml'
summary = {'checkpoint': str(RUN / 'train_ddp/weights/best.pt'), 'evaluations': {}}
for split in ['val', 'test']:
    model = YOLO(summary['checkpoint'])
    attach_stacking_metrics(model)
    paths = []
    model.add_callback('on_val_end', lambda v: paths.append(Path(v.save_dir)))
    metrics = model.val(data=DATA, split=split, device='', batch=16, imgsz=640,
                        workers=4, conf=0.001, iou=0.7, max_det=300,
                        project=str(RUN), name='ddp_final_' + split)
    result = {k: float(v) for k, v in metrics.results_dict.items()}
    with (paths[-1] / 'stacking_metrics.csv').open() as f:
        result['business_metrics'] = list(csv.DictReader(f))[-1]
    summary['evaluations'][split] = result
(RUN / 'completed_summary.json').write_text(json.dumps(summary, indent=2))
