# baseline_genyes8 轻量开发与部署测试包

本目录保留了完整的定制 Ultralytics 源码、RGB-D 模型配置、AFSS 风格训练采样、业务指标插件、训练入口、最新最佳权重及关键训练/测试结果。它适合继续增减模型模块、修改训练流程、重新训练，以及在 `val`/`test` 数据集上做部署前验收。

## 目录说明

- `ultralytics/`：完整框架源码。RGB-D 前向、融合模块和模型解析主要位于 `ultralytics/nn/tasks.py`。
- `ultralytics/cfg/models/11/yolo11-seg.yaml`：当前 RGB-D 分割模型配置，可切换或增减融合模块。
- `rgbd_stacking_metrics_plan/`：叠件实例级和图片级业务指标插件。
- `configs/data_63_7_30_server.yaml`：本次训练实际使用的四分类、63%/7%/30% 数据配置；迁移机器时需修改其中的 `path`。
- `train_portable.py`：无服务器绝对路径的训练/微调入口。
- `evaluate_portable.py`：验证集或测试集部署验收入口，同时输出原生指标和业务指标。
- `runs/20261019_s8_finetune/afss25/weights/best.pt`：最新 AFSS 25% 轮换采样最佳权重。
- `runs/20261019_s8_finetune/`：训练 CSV、最终摘要、双指标重统计和置信度扫描结果。
- `runs/legacy_coord_attv2_s8/`：旧实验关键结果，仅供对照，不含重复权重。

## 环境与完整性检查

建议使用项目原训练环境；新环境可在项目根目录执行：

```powershell
python -m pip install -e .
python verify_bundle.py
```

`verify_bundle.py` 会检查训练、指标、模型配置和结果文件，校验最佳权重 SHA-256，并编译全部 Python 源码。

## 部署前测试

数据集 YAML 必须正确指向 RGB 图像，并符合本项目的 RGB-D 配对约定。测试集验收示例：

```powershell
python evaluate_portable.py --data D:\datasets\genyes8\data.yaml --split test --device 0
```

根目录旧文件 `data-ubuntu-genye.yaml` 是历史两分类配置，仅为兼容旧实验保留，不能用于当前四分类权重验收。

默认使用 `conf=0.08`、NMS `iou=0.7`、匹配 IoU `0.5`。输出目录中会生成原生 Box/Mask 指标、`stacking_metrics.csv`、`stacking_metrics_v3.csv` 和 `deployment_test_summary.json`。

## 修改模块或训练流程

直接用最佳权重微调：

```powershell
python train_portable.py --data D:\datasets\genyes8\data.yaml --device 0 --name finetune_v1
```

修改模型模块后，从 YAML 建模训练：

```powershell
python train_portable.py --data D:\datasets\genyes8\data.yaml --model ultralytics\cfg\models\11\yolo11-seg.yaml --device 0 --name module_ablation_v1
```

新增模块时，除了实现模块类，还需在模型解析位置注册，并同步修改 YAML。建议每次先用小数据/少量 epoch 做 smoke test，再进行完整训练与独立测试集验收。

## 精简内容

本包未保留 Python 缓存、重复的 `last.pt`、基础预训练权重、批次可视化图片和大体积控制台日志。服务器原始项目与完整运行结果未删除。
