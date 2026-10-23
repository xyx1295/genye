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
THRESHOLDS = [0.05, 0.06, 0.07, 0.08, 0.09]

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
    import json, os
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    output = RUN / 'afss25_all_images_dual_v3_20261023'
    output.mkdir(exist_ok=False)
    totals = {t: StackingMetrics(t, 0.5) for t in THRESHOLDS}
    def aggregate(v):
        for t, counter in v._sweep_counters.items():
            totals[t].matrix += counter.matrix
            totals[t].image_matrix += counter.image_matrix
            totals[t].images += counter.images
            totals[t].unmatched_stacks += counter.unmatched_stacks
    for split in ['train', 'val', 'test']:
        model = YOLO(str(CHECKPOINT))
        attach_stacking_metrics(model, conf=0.001, iou=0.5)
        model.add_callback('on_val_start', install_sweep)
        model.add_callback('on_val_end', write_results)
        model.add_callback('on_val_end', aggregate)
        model.val(data=DATA, split=split, device='', imgsz=640, batch=16, workers=4,
                  conf=0.001, iou=0.7, max_det=300, plots=False,
                  project=str(output), name=split, exist_ok=False)
    rows = [{'confidence': t, **c.result()} for t,c in totals.items()]
    assert all(r['images'] == 18491 for r in rows), rows
    with (output/'all_images.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'summary.json').write_text(json.dumps({'checkpoint':str(CHECKPOINT), 'scope':'train+val+test', 'nms_iou':0.7, 'matching_iou':0.5, 'rows':rows},indent=2))
    print('ALL_IMAGES_COMPLETE', json.dumps(rows), flush=True)
