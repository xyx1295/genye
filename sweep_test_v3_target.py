"""Sweep business confidence on held-out test without retraining."""
import csv
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from rgbd_stacking_metrics_plan.yolo11_stacking_metrics import StackingMetrics, _numpy, attach_stacking_metrics

ROOT = Path(__file__).resolve().parent
RUN = Path('/workspace/runs/baseline_genyes8/20261019_s8_finetune')
DATA = '/workspace/datasets/genyes8_20261019/split_63_7_30_grouped/data.yaml'
CHECKPOINT = RUN / 'afss25/weights/best.pt'
THRESHOLDS = sorted(set([0.001] + [i/1000 for i in range(5,1000,5)] + [i/1000 for i in range(90,201)]))

def install_sweep(validator):
    counters = {t: StackingMetrics(t, 0.5) for t in THRESHOLDS}
    original = validator.update_metrics
    def update(preds, batch):
        indices = _numpy(batch['batch_idx']).reshape(-1)
        classes = _numpy(batch['cls']).reshape(-1)
        boxes = _numpy(batch['bboxes'])
        height, width = batch['img'].shape[-2:]
        for si, pred in enumerate(preds):
            xywh = boxes[indices == si]
            gt = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
            gt *= np.array([width, height, width, height])
            pb, pc, pf = (_numpy(pred[k]) for k in ['bboxes', 'cls', 'conf'])
            for counter in counters.values():
                counter.update(gt, classes[indices == si], pb, pc, pf)
        return original(preds, batch)
    validator.update_metrics = update
    validator._sweep_counters = counters

def write_results(validator):
    out = Path(validator.save_dir) / 'business_confidence_sweep.csv'
    rows = []
    for threshold, counter in validator._sweep_counters.items():
        rows.append({'confidence': threshold, **counter.result(),
                     'hit_target_99': (counter.result()['stack_hit_rate'] or 0) >= 0.99,
                     'error_at_most_10': (counter.result()['single_to_stack_error_rate'] or 0) <= 0.10})
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    feasible = [r for r in rows if r['hit_target_99'] and r['error_at_most_10']]
    best_hit = max(rows, key=lambda r: (r['stack_hit_rate'] or -1))
    best_under_error = max([r for r in rows if (r['single_to_stack_error_rate'] or 0) <= 0.10], key=lambda r: r['stack_hit_rate'] or -1)
    print('SWEEP_RESULTS', {'path': str(out), 'feasible': feasible, 'max_hit': best_hit,
                            'best_hit_with_error_le_10': best_under_error}, flush=True)


if __name__ == '__main__':
    import os, json
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    output=RUN/'afss25_test_image_target_v3_20261023'
    model=YOLO(str(CHECKPOINT))
    attach_stacking_metrics(model, conf=0.001, iou=0.5)
    model.add_callback('on_val_start', install_sweep)
    model.add_callback('on_val_end', write_results)
    model.val(data=DATA, split='test', device='', imgsz=640, batch=16, workers=4,
              conf=0.001, iou=0.7, max_det=300, plots=False, project=str(output),name='scan',exist_ok=False)
    with (output/'scan/business_confidence_sweep.csv').open() as f: rows=list(csv.DictReader(f))
    feasible=[r for r in rows if float(r['image_hit_rate']) >= .99 and float(r['image_error_rate']) <= .10]
    under=[r for r in rows if float(r['image_error_rate']) <= .10]
    high=[r for r in rows if float(r['image_hit_rate']) >= .99]
    summary={'checkpoint':str(CHECKPOINT),'split':'test','threshold_count':len(rows),
             'feasible':feasible,
             'best_hit_under_10':max(under,key=lambda r:(float(r['image_hit_rate']),-float(r['image_error_rate']))) if under else None,
             'least_error_at_99':min(high,key=lambda r:float(r['image_error_rate'])) if high else None}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('IMAGE_TARGET_COMPLETE',json.dumps(summary),flush=True)
