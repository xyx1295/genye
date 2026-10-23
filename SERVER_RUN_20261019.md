# S8 服务器微调记录

- 容器：`xingyuxuan`；连接 `root@192.168.3.64:11022`。
- 代码：`/workspace/projects/baseline_genyes8`。
- 环境：`/root/miniconda3/envs/yolo11_genye`，PyTorch 2.5.1+cu121。
- 数据副本及解压目录：`/workspace/datasets/genyes8_20261019`。
- 数据划分：`/workspace/datasets/genyes8_20261019/split_63_7_30_grouped/data.yaml`。
- 结果：`/workspace/runs/baseline_genyes8/20261019_s8_finetune`。

两个 ZIP 均保留独立副本，并校验归档 SHA256 和 manifest 中每个 RGB、depth、label 文件的 SHA256。lite20 是 full 的子集，仅复制、解压，不与 full 合并。正式训练使用 full，保持各场地原 test 数量，将相同 RGB SHA256 的样本作为一组；修正原始 12 组跨 train/test 的重复内容，通过确定性单样本交换维持 test 数量，再从剩余样本按组划分 val（SHA256 排序，seed=2027），约为总量 63%/7%/30%。由于整数取整，比例有微小偏差。数据集原始划分以图片为单位，不是以视频序列分组，近邻帧相关性仍是评估限制。

训练初始化使用原 S8 `coord_attv2-s8/weights/best.pt`，保留 RGB-D 架构并适配四分类头。单卡，batch=16，imgsz=640，MuSGD，lr0=0.01，最多 300 轮，patience=100，训练 seed=28975。此次仅一次微调，不执行旧计划的多种子、scratch 或超参数搜索。

通过 `srun-gpu -g 1 -p gpu -c xingyuxuan -e yolo11_genye -w /workspace/projects/baseline_genyes8 -- bash run_server_s8.sh` 申请 GPU。使用调度器分配的可见设备和逻辑 cuda:0。

`run_server_s8.sh` 先做短训练和验证，成功后正式训练，随后最佳权重分别验证 val/test。`smoke.log`、`train.log` 和 `launcher.log` 记录进度。每轮业务指标写入 `train/stacking_metrics.csv`，最终业务指标写入 `final_val/stacking_metrics.csv`、`final_test/stacking_metrics.csv`，全部成功才生成 `completed_summary.json`。

业务口径：A={0,3}，B={1,2}，conf>=0.55，Box IoU>=0.5，类别无关一对一匹配；命中率为 TP_B/N_B，误判率为 E_A_to_B/(N_B+E_A_to_B)，未匹配叠件预测单独统计。test 不参与早停、选模或调参。脚本中的合成数据断言用于核对这些定义。

本文件描述执行方案，不表示训练已经结束；应以服务器日志和 `completed_summary.json` 为准。

数据检查发现 30 组相同 RGB 内容、12 组原始跨训练测试重复。实际训练只使用带 grouped 后缀的划分；不带 grouped 的目录是检查未通过的中间产物，禁止使用。


已提交 Slurm JobID=140（2026-10-19，北京时间约 16:42），单卡。实际划分：train=11649、val=1296、test=5546。原始 test 成员调整 24 张，解决 12 组重复 RGB 跨分区问题。


Job 140 的短训练暴露指标脚本对反向预测框的异常处理，未进入正式训练。已修正：无效预测框不允许匹配 GT，仍保留为未匹配预测，不交换坐标；新增合成样本断言。重试作业 JobID=141，日志 launcher_retry.log。


重试短训练 smoke2 已通过（SMOKE_OK），完整验证及业务 CSV 写入成功。Job 141 已进入正式 train 任务，最终测试结果待训练完成后生成。


## 双卡排队与切换

按用户要求提交双卡 Slurm 作业 143，目前 PENDING(Resources)。现有单卡作业 141 继续训练，不提前取消。

入口为 takeover_two_gpu.py：获批后检查可用显存，执行双卡短训练；只有短训练成功且取得可恢复的检查点，才取消已确认属于本任务的 141，并用 torchrun 两进程从检查点恢复。总 batch 保持 16（每卡 8），不从头训练。切换可能回退当前尚未保存的轮内进度。指标适配仅允许已确认采用 rank0 完整验证集的 RankZeroMetricsTrainer；普通多卡调用仍拒绝。

队列日志：two_gpu_queue.log；状态：two_gpu_state.json；双卡短训练：ddp_smoke.log；正式训练：train_ddp.log；最终测试：ddp_final_test/stacking_metrics.csv。原单卡结果保留，双卡输出位于 train_ddp。

双卡代码已通过语法和导入检查；实际双卡验证将在资源获批后执行。Slurm 获批不代表宿主机没有未登记的 GPU 占用。该方案是排队后从检查点重启，不是在现有进程中热添加显卡。
