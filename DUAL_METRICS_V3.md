# 双指标输出

本项目四类 detect/segment 验证器自动挂载双指标；显式 attach_stacking_metrics 的已有调用优先，避免重复计数。其他类别数量不自动套用本业务定义。

- stacking_metrics.csv：保留旧列名、实例级匹配规则及历史兼容性。
- stacking_metrics_v3.csv：新增图片级 TP/FN/FP/TN、命中率、误判率，同时记录实例级结果。
- 图片触发类别固定为 {1,2}；空GT图为单件图；预测经过相同业务置信度过滤后，只要出现类别1或2即预测叠件图，无需IoU匹配。
- 图片命中率 = TP / (TP+FN)。图片误判率 = FP / (TP+FN+FP)。零分母为None/CSV空白。
- 实例规则和无效预测框处理沿用旧实现。未匹配预测仍单独统计。
- 自动挂载业务conf默认0.55；若验证器输入conf更高则使用输入conf。显式插件可指定业务conf，验证器输入conf不得高于业务conf。
- 扫描脚本结果同时带旧指标键及v3新键。跨数据集汇总累加实例矩阵和图片矩阵，再计算比率。

本次重统计：afss25/weights/best.pt，train+val+test共18491张；conf=0.05/0.06/0.07/0.08/0.09，NMS IoU=0.7，实例匹配IoU=0.5。
包含训练集的全量结果不是独立测试集泛化指标。

核验：200组随机合法框数据与旧版及zip内v3参考逐字段一致；空集合零分母、无GT未匹配叠件预测、双CSV输出和SegmentationValidator自动注册均通过。

归档旧实现：rgbd_stacking_metrics_plan/yolo11_stacking_metrics.legacy.py
参考实现：rgbd_stacking_metrics_plan/v3_reference.py
压缩包内额外训练计划未执行。
