"""One S8-initialized run, validation callbacks, then held-out test evaluation."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Segment
from rgbd_stacking_metrics_plan.yolo11_stacking_metrics import StackingMetrics, attach_stacking_metrics

ROOT = Path(__file__).resolve().parent
DATA = '/workspace/datasets/genyes8_20261019/split_63_7_30_grouped/data.yaml'
RUNS = Path('/workspace/runs/baseline_genyes8/20261019_s8_finetune')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--train-sample-fraction', type=float, default=0.25,
                        help='Fraction of training images sampled per epoch (AFSS-inspired).')
    parser.add_argument('--run-name', default='afss25', help='Output run name.')
    args = parser.parse_args()
    allocation = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert allocation and len(allocation.split(',')) == 1, 'Requires exactly one scheduler-assigned GPU'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    print('Scheduler allocation:', allocation, 'logical device: cuda:0', flush=True)
    # Verify the specified business denominator, class merging and unmatched count.
    counter = StackingMetrics()
    counter.update([[0,0,10,10],[20,20,30,30],[40,40,50,50]], [1,2,0],
                   [[0,0,10,10],[40,40,50,50],[60,60,70,70]], [2,1,2], [0.9]*3)
    r = counter.result()
    assert r['stack_hit_rate'] == 0.5 and r['single_to_stack_error_rate'] == 1/3 and r['unmatched_stack_prediction_count'] == 1
    counter = StackingMetrics()
    counter.update(np.empty((0,4)), [], np.empty((0,4)), [], [])
    assert counter.result()['stack_hit_rate'] is None
    counter = StackingMetrics()
    counter.update([[0,0,10,10]], [1], [[10,10,0,0]], [1], [0.9])
    assert counter.result()['stack_hit_count'] == 0 and counter.result()['unmatched_stack_prediction_count'] == 1

    checkpoint = ROOT / 'YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt'
    model = YOLO(str(checkpoint))
    attach_stacking_metrics(model)
    def audit_training(trainer):
        heads = [m for m in trainer.model.modules() if isinstance(m, Segment)]
        assert heads and all(h.nc == 4 for h in heads)
        for loader in [trainer.train_loader, trainer.test_loader]:
            dataset = loader.dataset
            assert len(dataset.depth_files) == len(dataset.im_files) > 0
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == allocation
        print('AUDIT_OK: four classes; RGB-D training and validation pairs; allocation preserved', flush=True)
    model.add_callback('on_pretrain_routine_end', audit_training)
    model.train(data=DATA, epochs=1 if args.smoke else 300, patience=100,
                fraction=0.002 if args.smoke else 1.0,
                train_sample_fraction=args.train_sample_fraction,
                imgsz=640, batch=16, workers=4, device='', optimizer='MuSGD',
                lr0=0.01, lrf=0.001, momentum=0.937, weight_decay=0.0001,
                warmup_epochs=3.0, amp=True, seed=28975, deterministic=True,
                mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
                conf=0.001, iou=0.7, max_det=300, plots=True,
                project=str(RUNS), name='smoke' if args.smoke else args.run_name, exist_ok=False)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == allocation
    heads = [module for module in model.model.modules() if isinstance(module, Segment)]
    assert heads and all(head.nc == 4 for head in heads)
    if args.smoke:
        print('SMOKE_OK', flush=True)
        return
    best = model.trainer.best
    summary = {'checkpoint': str(best), 'data': DATA, 'allocation': allocation, 'evaluations': {}}
    for split in ['val', 'test']:
        evaluated = YOLO(str(best))
        attach_stacking_metrics(evaluated)
        evaluation_path = []
        evaluated.add_callback('on_val_end', lambda validator: evaluation_path.append(Path(validator.save_dir)))
        result = evaluated.val(data=DATA, split=split, device='', imgsz=640, batch=16,
                               workers=4, conf=0.001, iou=0.7, max_det=300,
                               project=str(RUNS), name=f'final_{split}', plots=True)
        summary['evaluations'][split] = {k: float(v) for k, v in result.results_dict.items()}
        with (evaluation_path[-1] / 'stacking_metrics.csv').open() as stream:
            summary['evaluations'][split]['business_metrics'] = list(csv.DictReader(stream))[-1]
    (RUNS / 'completed_summary.json').write_text(json.dumps(summary, indent=2))
    print('TRAIN_AND_TEST_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
