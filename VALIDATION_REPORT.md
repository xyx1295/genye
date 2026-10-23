# 轻量包验证报告

验证日期：2026-10-23

## 已通过

- 轻量包结构检查：训练框架、RGB-D 模型 YAML、业务指标插件、可移植训练/评估入口、最佳权重和关键结果齐全。
- Python 静态编译：全部 `.py` 文件通过语法编译检查。
- 最佳权重完整性：本地与服务器 SHA-256 均为 `11111af195b4ef5af9e84836335b355b1916659abe42f40565878bfb8b561ef4`。
- 服务器 `yolo11_genye` 环境运行检查：`best.pt` 成功加载为 `SegmentationModel`。
- 模型可修改性检查：当前 `ultralytics/cfg/models/11/yolo11-seg.yaml` 成功从源码构建为 `SegmentationModel`。
- 轻量化检查：仅保留一个 `.pt`，无 `__pycache__`，未保留重复 `last.pt`、基础预训练权重或大体积日志。

## 环境边界

当前本机已有的两个 Conda 环境都没有安装 PyTorch，因此没有在本机执行 GPU/CPU 推理。项目提供的 `pyproject.toml` 可用于建立环境；安装完成后运行 `python verify_bundle.py`，再使用 `evaluate_portable.py` 对实际 `val` 或 `test` 数据做部署验收。

数据集本身未包含在轻量包中。四分类数据配置保存在 `configs/data_63_7_30_server.yaml`，迁移到其他机器时必须修改 `path`，并保持 RGB-D 配对目录约定。
