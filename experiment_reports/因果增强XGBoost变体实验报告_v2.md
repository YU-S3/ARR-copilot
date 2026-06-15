# 因果增强 XGBoost 变体实验报告 v2

## 1. 本轮目标

本轮工作直接承接上一版报告的“后续建议”，聚焦两件事：

1. 保留 `ordinal_emd_fw_gpl_xgboost` 作为当前因果增强 XGBoost 主线，并补做真正的后处理概率校准，缓解其“分类强、概率偏激进”的问题。
2. 再补一轮“去除确诊后变量”的严格前瞻性版本，评估在更接近真实筛查场景下，模型性能会发生怎样的变化。

与上一版不同的是：

- 本轮新增了 `causal_xgboost_outputs_v2` 结果目录；
- 不再处理 `test.xlsx` 外部评测；
- 不改写旧版报告，而是单独输出本报告 `因果增强XGBoost变体实验报告_v2.md`。

## 2. 数据与统一口径

### 2.1 数据来源

- 主数据：`c:\Users\YCY\Desktop\ARR-model\数据表格测试.xlsx`
- 主脚本：`c:\Users\YCY\Desktop\ARR-model\causal_xgboost_variants_experiment.py`
- 结果目录：`c:\Users\YCY\Desktop\ARR-model\causal_xgboost_outputs_v2`

### 2.2 任务定义

- 目标列：`确诊（0为排除；1为确诊；2为灰色区域）`
- 统一三分类任务：`0 / 1 / 2`
- 有序主线内部 rank 仍按：`0 < 2 < 1`

### 2.3 数据规模

根据 [experiment_summary.json](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/experiment_summary.json)：

- 总样本数：`374`
- 训练集：`299`
- 测试集：`75`

## 3. 主线版本结果复核

### 3.1 当前完整特征版本的最佳方案

根据 [experiment_summary.json](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/experiment_summary.json) 与 [variant_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/tables/variant_metrics.csv)：

- 当前完整特征版本的最佳方案仍然是 `ordinal_emd_fw_gpl_xgboost`
- 测试集结果：
  - Accuracy：`0.9600`
  - Balanced Accuracy：`0.8935`
  - Macro F1：`0.8935`
  - Weighted F1：`0.9600`
  - OVR ROC AUC Macro：`0.9614`

相比原始 `baseline_xgboost`：

- Accuracy：`0.9467 -> 0.9600`
- Balanced Accuracy：`0.8869 -> 0.8935`
- Macro F1：`0.8623 -> 0.8935`

因此，本轮仍保留 `ordinal_emd_fw_gpl_xgboost` 作为因果增强 XGBoost 的主线方案。

### 3.2 为什么还要继续做概率校准

虽然主线方案分类结果最好，但它的原始校准较差：

- `ordinal_emd_fw_gpl_xgboost`
  - ECE：`0.2343`
  - Brier：`0.1584`
- `baseline_xgboost`
  - ECE：`0.0315`
  - Brier：`0.0858`

这说明主线模型更擅长给出正确分类，但其概率输出明显更“激进”，不利于后续做阈值解释、灰区定位或概率型临床决策。

## 4. 主线概率校准实验

### 4.1 实验设计

本轮新增了针对主线模型的后处理校准实验，重点对象是：

- `ordinal_emd_fw_gpl_xgboost`

实现策略：

1. 从原训练集 `299` 例中进一步切分出：
   - 拟合集：`224`
   - 校准集：`75`
2. 先用拟合集训练主线模型；
3. 再在校准集上学习后处理校准器；
4. 最后统一在同一测试集 `75` 例上比较校准前后表现。

需要注意：

- 这一校准实验的“uncalibrated”结果，使用的是 **224 例拟合集训练出的主线模型**，因此它与上面“299 例完整训练集训练出的主线主结果”不完全相同；
- 这是为了保证校准过程没有使用测试集信息。

### 4.2 比较方法

本轮实际比较了 4 个版本：

1. `uncalibrated`
2. `temperature_scaling`
3. `ovr_isotonic`
4. `ovr_sigmoid`

对应结果表见 [mainline_calibration_comparison.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/tables/mainline_calibration_comparison.csv)。

### 4.3 校准结果

| 方法 | Accuracy | Balanced Accuracy | Macro F1 | AUC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| temperature_scaling | 0.9467 | 0.8869 | 0.8623 | 0.9457 | 0.0466 | 0.1028 |
| ovr_isotonic | 0.9600 | 0.8935 | 0.8935 | 0.9155 | 0.0495 | 0.1042 |
| uncalibrated | 0.9467 | 0.8869 | 0.8623 | 0.9509 | 0.2280 | 0.1836 |
| ovr_sigmoid | 0.9200 | 0.6435 | 0.6311 | 0.9436 | 0.2518 | 0.2149 |

### 4.4 结果解读

可以看到：

1. `temperature_scaling` 明显改善了概率质量。
   - ECE：`0.2280 -> 0.0466`
   - Brier：`0.1836 -> 0.1028`
   - 同时保持了与未校准版本一致的 Accuracy / Balanced Accuracy / Macro F1

2. `ovr_isotonic` 也显著改善了校准。
   - ECE：`0.2280 -> 0.0495`
   - Brier：`0.1836 -> 0.1042`
   - 并且在当前测试集上保留了更高的分类指标：
     - Accuracy：`0.9600`
     - Macro F1：`0.8935`

3. `ovr_sigmoid` 明显不适合当前主线模型。
   - 分类性能和校准同时恶化
   - 说明在当前小样本表格任务中，逐类 sigmoid 校准引入了额外不稳定性

### 4.5 推荐校准方案

根据本轮设定的优先级，本次将 `temperature_scaling` 记为推荐校准方法，原因是：

- 它给出了最低的 `ECE`；
- `Brier Score` 也最低；
- 与未校准版本相比，不牺牲分类主指标。

但同时要强调：

- 如果优先目标是 **在保住当前分类最强结果的同时改善校准**，那么 `ovr_isotonic` 也是一个很强的候选；
- 如果优先目标是 **更稳、更轻量、后续部署更简单**，`temperature_scaling` 更合适。

因此，本轮结论可以表述为：

- **部署推荐校准主线：`ordinal_emd_fw_gpl_xgboost + temperature_scaling`**
- **性能保持型备选：`ordinal_emd_fw_gpl_xgboost + ovr_isotonic`**

## 5. 严格前瞻性版本实验

### 5.1 设计动机

旧版最佳结果大量利用了“确诊后”变量，例如：

- `确诊后_卧位醛固酮醛固酮`
- `肾素.1`
- `确诊后_卧位醛固酮醛固酮.1`
- `肾素.2`
- `确诊后_卧位醛固酮醛固酮.2`
- `肾素.3`

这些变量在真实筛查时点并不可用，因此本轮额外构造了严格前瞻性视图，将上述 6 列全部移除，再重新比较核心方案。

### 5.2 比较范围

按本轮约定，仅保留两个核心方案：

1. `baseline_xgboost_strict_prospective`
2. `ordinal_emd_fw_gpl_xgboost_strict_prospective`

结果表见 [strict_prospective_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/tables/strict_prospective_metrics.csv)。

### 5.3 严格前瞻性结果

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | AUC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| ordinal_emd_fw_gpl_xgboost_strict_prospective | 0.7333 | 0.4810 | 0.4681 | 0.7166 | 0.1451 | 0.4160 |
| baseline_xgboost_strict_prospective | 0.6533 | 0.4824 | 0.4411 | 0.7122 | 0.0991 | 0.4305 |

### 5.4 结果解读

严格前瞻性版本的结果有两个非常清晰的信号：

1. 去除确诊后变量后，整体性能明显下降。
   - 主线方案 Macro F1 从完整特征版本的 `0.8935` 降到 `0.4681`
   - Accuracy 从 `0.9600` 降到 `0.7333`

2. 即使在严格前瞻性条件下，主线方案仍略优于基线。
   - Accuracy：`0.7333 > 0.6533`
   - Macro F1：`0.4681 > 0.4411`
   - AUC：`0.7166 > 0.7122`

3. 但两者的 `Balanced Accuracy` 都只有约 `0.48`，说明：
   - 在只保留筛查时点变量后，当前模型对少数类和边界状态的辨别能力明显不足；
   - 先前最强结果中，“确诊后”变量贡献极大。

### 5.5 方法学含义

这轮严格前瞻性实验的价值，不在于追求更高结果，而在于界定“当前最好成绩的上界来自哪里”。

本轮结论表明：

- 当前完整特征版本的高性能，确实部分依赖确诊流程中的后续变量；
- 如果目标是构建更接近真实筛查场景的模型，必须接受性能明显下降这一现实；
- 因而后续如果要继续推进前瞻性主线，应重点补充：
  - 更强的筛查前变量；
  - 更高质量的药物暴露信息；
  - 更稳定的少数类建模策略。

## 6. 本轮综合结论

本轮 v2 实验可以归纳为 4 条：

1. 在完整特征版本下，`ordinal_emd_fw_gpl_xgboost` 仍然是当前因果增强 XGBoost 主线，分类性能继续优于原始 `baseline_xgboost`。
2. 主线模型原始概率确实偏激进，补做后处理校准是必要的。
3. 在本轮多方法比较中：
   - `temperature_scaling` 是更稳的推荐校准方案；
   - `ovr_isotonic` 是兼顾分类性能的强备选；
   - `ovr_sigmoid` 不建议继续保留。
4. 严格前瞻性版本性能显著下滑，说明当前项目若坚持真实筛查口径，仍需要更多前诊断阶段的信息支持。

## 7. 当前建议输出

基于本轮结果，建议形成两套并行结论：

- **论文/方法学主线**
  - `ordinal_emd_fw_gpl_xgboost`
  - 配套推荐校准：`temperature_scaling`

- **更接近真实筛查场景的保守结论**
  - `ordinal_emd_fw_gpl_xgboost_strict_prospective`
  - 但需明确说明其性能明显低于完整特征版本，目前更适合作为“严格前瞻性补充实验”，而不是最终部署版本

## 8. 结果文件索引

### 8.1 主结果

- 摘要：[experiment_summary.json](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/experiment_summary.json)
- 全变体指标：[variant_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/tables/variant_metrics.csv)
- 全变体校准：[variant_calibration_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/tables/variant_calibration_metrics.csv)

### 8.2 主线校准实验

- 对比表：[mainline_calibration_comparison.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/tables/mainline_calibration_comparison.csv)
- 预测表：`causal_xgboost_outputs_v2/mainline_calibration_predictions.xlsx`
- 图：
  - `figures/mainline_calibration_metrics.png`
  - `figures/mainline_calibration_comparison.png`
  - `figures/mainline_best_calibrated_confusion_heatmap.png`
  - `figures/mainline_best_calibrated_multiclass_roc.png`

### 8.3 严格前瞻性实验

- 指标表：[strict_prospective_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs_v2/tables/strict_prospective_metrics.csv)
- 摘要：`artifacts/strict_prospective_summary.json`
- 预测表：`causal_xgboost_outputs_v2/strict_prospective_predictions.xlsx`
- 图：
  - `figures/strict_prospective_metrics.png`
  - `figures/strict_prospective_calibration.png`
  - `figures/strict_prospective_roc_overview.png`
  - `figures/strict_prospective_mainline_confusion_heatmap.png`
  - `figures/strict_prospective_mainline_multiclass_roc.png`

## 9. 下一步建议

如果继续沿这条线推进，优先级建议如下：

1. 保留 `ordinal_emd_fw_gpl_xgboost + temperature_scaling` 作为当前主结果展示版本。
2. 在论文中同时报告 `ovr_isotonic`，说明“校准与分类性能之间存在取舍”。
3. 将严格前瞻性版本作为补充结果，而不是替代主结果。
4. 若要真正推进前瞻性主线，应优先补充更强的筛查前变量，而不是继续叠加更复杂的损失结构。
