"""Explicit torchrun entry; rank zero validates the complete validation set."""
import argparse
import os
from pathlib import Path
import torch
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import RANK
from rgbd_stacking_metrics_plan.yolo11_stacking_metrics import attach_stacking_metrics

ROOT = Path(__file__).resolve().parent
RUN = Path('/workspace/runs/baseline_genyes8/20261019_s8_finetune')
DATA = '/workspace/datasets/genyes8_20261019/split_63_7_30_grouped/data.yaml'

class RankZeroMetricsTrainer(SegmentationTrainer):
    # BaseTrainer builds and runs the full validation loader on rank 0 only.
    _stacking_rank0_full_validation = True

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            self.args.project = str(RUN)
            self.args.name = 'train_ddp'
            self.args.exist_ok = True
            self.args.save_dir = str(RUN / 'train_ddp')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert len(mask.split(',')) == 2 and int(os.environ['WORLD_SIZE']) == 2
    assert torch.cuda.device_count() == 2
    if args.smoke:
        cfg = dict(model=str(ROOT / 'YOLOv11-RGB-D-coord_attv2-genye/coord_attv2-s8/weights/best.pt'),
                   data=DATA, epochs=1, fraction=0.002, imgsz=640, batch=16, workers=2,
                   device='', optimizer='MuSGD', lr0=0.01, lrf=0.001, momentum=0.937,
                   weight_decay=0.0001, amp=True, seed=28975, deterministic=True,
                   conf=0.001, iou=0.7, max_det=300, plots=False,
                   project=str(RUN), name='ddp_smoke', exist_ok=True)
    else:
        cfg = dict(model=str(RUN / 'train_ddp/weights/last.pt'),
                   resume=str(RUN / 'train_ddp/weights/last.pt'), device='', batch=16)
    trainer = RankZeroMetricsTrainer(overrides=cfg)
    attach_stacking_metrics(trainer)
    def audit(t):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == mask
        assert len(t.train_loader.dataset.depth_files) == len(t.train_loader.dataset.im_files)
        if RANK == 0:
            assert len(t.test_loader.dataset) == 1296
            assert len(t.test_loader.dataset.depth_files) == 1296
            print('DDP_AUDIT_OK: full validation on rank zero; paired RGB-D; original GPU mask', flush=True)
    trainer.add_callback('on_pretrain_routine_end', audit)
    trainer.train()
    if RANK == 0:
        if args.smoke:
            (RUN / 'ddp_smoke_ok').write_text('Passed two-rank training and full validation')
        else:
            (RUN / 'ddp_training_ok').write_text(str(trainer.best))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
