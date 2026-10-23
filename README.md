# baseline_genyes8

轻量化的 RGB-D YOLO11 分割训练与部署测试项目。仓库保留完整可修改源码、AFSS 风格轮换采样、叠件业务指标、最新最佳权重和关键实验结果，可用于继续训练、增减模型模块及独立测试集验收。

## 主要内容

- `ultralytics/`：定制 RGB-D Ultralytics 源码。
- `ultralytics/cfg/models/11/yolo11-seg.yaml`：当前 CoordAttV2、P3 wavelet 和 Depth FPN 模型配置。
- `rgbd_stacking_metrics_plan/`：实例级与图片级叠件业务指标。
- `train_portable.py`：可移植训练、微调和模块消融入口。
- `evaluate_portable.py`：验证集或测试集部署验收入口。
- `runs/20261019_s8_finetune/afss25/weights/best.pt`：已校验的最佳权重。
- `runs/20261019_s8_finetune/`：训练曲线、业务指标、最终摘要和阈值扫描结果。
- `LIGHTWEIGHT_MANIFEST.csv`：公开包内文件的 SHA-256 清单。

## 安装与检查

建议使用 Python 3.10 和与硬件匹配的 PyTorch/CUDA：

```bash
python -m pip install -e .
python verify_bundle.py
```

## 部署测试

```bash
python evaluate_portable.py --data /path/to/data.yaml --split test --device 0
```

## 继续训练或修改模块

使用最佳权重微调：

```bash
python train_portable.py --data /path/to/data.yaml --device 0 --name finetune_v1
```

修改 `ultralytics/nn/tasks.py` 或模型 YAML 后，从配置重新建模：

```bash
python train_portable.py --data /path/to/data.yaml \
  --model ultralytics/cfg/models/11/yolo11-seg.yaml \
  --device 0 --name module_ablation_v1
```

更完整的目录说明、业务阈值和迁移注意事项见 [README_LIGHTWEIGHT.md](README_LIGHTWEIGHT.md)，验证记录见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)。数据集不包含在仓库中。

## License

AGPL-3.0，详见 [LICENSE](LICENSE)。
