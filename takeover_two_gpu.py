"""Run inside the allocated container. Preserve the single-GPU job until DDP smoke passes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

RUN = Path('/workspace/runs/baseline_genyes8/20261019_s8_finetune')
STATE = RUN / 'two_gpu_state.json'

def state(phase, **extra):
    STATE.write_text(json.dumps({'phase': phase, 'gpu_mask': os.environ.get('CUDA_VISIBLE_DEVICES'), **extra}, indent=2))
    print(phase, extra, flush=True)

def command(args, logfile):
    with (RUN / logfile).open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True)

def main():
    assert len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == 2
    # Do not interrupt a completed single-GPU run or start a duplicate experiment.
    if (RUN / 'completed_summary.json').exists():
        state('already_complete'); return
    state('checking_memory')
    while True:
        free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(2)]
        if free[0] >= 11 and free[1] >= 6:
            break
        state('waiting_for_memory', free_gib=free)
        if (RUN / 'completed_summary.json').exists():
            state('already_complete'); return
        time.sleep(60)
    state('ddp_smoke')
    ddp = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', 'train_server_ddp.py']
    command(ddp + ['--smoke'], 'ddp_smoke.log')
    assert (RUN / 'ddp_smoke_ok').exists()
    if (RUN / 'completed_summary.json').exists():
        state('already_complete'); return
    # Copy a fully deserializable checkpoint before stopping only our known job.
    source = RUN / 'train/weights/last.pt'
    destination = RUN / 'train_ddp'
    destination.mkdir(exist_ok=True)
    (destination / 'weights').mkdir(exist_ok=True)
    for _ in range(30):
        try:
            shutil.copy2(source, destination / 'weights/last.pt')
            ckpt = torch.load(destination / 'weights/last.pt', map_location='cpu', weights_only=False)
            assert ckpt['epoch'] >= 0 and ckpt.get('optimizer') is not None
            epoch = int(ckpt['epoch']) + 1
            del ckpt
            break
        except (OSError, RuntimeError, EOFError, AssertionError):
            time.sleep(2)
    else:
        raise RuntimeError('Cannot obtain resumable checkpoint; single-GPU job left running')
    info = subprocess.check_output(['scontrol', 'show', 'job', '141'], text=True)
    assert 'JobId=141 ' in info and 'JobState=RUNNING' in info
    state('switching', completed_epoch=epoch)
    subprocess.run(['scancel', '141'], check=True)
    # Original job is now stopped; preserve original best and metric history separately.
    for name in ['best.pt']:
        if (RUN / 'train/weights' / name).exists():
            shutil.copy2(RUN / 'train/weights' / name, destination / 'weights' / name)
    for name in ['results.csv', 'stacking_metrics.csv']:
        if (RUN / 'train' / name).exists():
            shutil.copy2(RUN / 'train' / name, destination / ('single_gpu_' + name))
    state('dual_gpu_training', resumed_after_epoch=epoch)
    command(ddp, 'train_ddp.log')
    state('final_evaluation')
    command([sys.executable, 'evaluate_server_ddp.py'], 'ddp_final_evaluation.log')
    state('complete')

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed', error=str(error))
        raise
