# 基于 ARR 真实世界数据的三分类诊断模型综合实验报告

版本：论文整理版  
数据目录：`C:\Users\YCY\Desktop\ARR-model`  
主要任务：预测 `0=非确诊`、`1=确诊`、`2=灰色区域`

## 摘要

本研究围绕 ARR 真实世界临床表格数据建立三分类诊断模型。数据清洗后保留 374 例有效样本，其中 `0=非确诊` 仅 22 例，`1=确诊` 255 例，`2=灰色区域` 97 例，任务核心难点是极小样本少数类与灰区边界识别。实验从 Random Forest 基线出发，依次比较 XGBoost、异构集成、TabPFN、因果增强 XGBoost、前沿表格增广、SCM-v2、SCM-v3、V3 泛化诊断和 V3.1 概率校准/TabDDPM 追加实验。

综合所有结果，最稳健的正式主线建议为 `xgb_scm_v2_best + temperature calibration`。在主实验 5 个随机种子 80/20 划分下，`xgb_scm_v2_best` 达到 `Accuracy=0.9840`、`Balanced Accuracy=0.9727`、`Macro F1=0.9591`、`0类 Recall=0.95`。`SCM_v3` 的最优配置 `scm_v3_res_norule_single_flat_nocf` 在同一主实验口径下略高，`Balanced Accuracy=0.9768`、`Macro F1=0.9631`，说明 V3 结构增广有效；但在 V3.1 的 train/valid/test 三段诊断中，V2 主线的测试稳健性优于 V3，因此 V3 当前更适合作为候选增强版，而不是替代 V2 的正式保守主线。

## 1. 数据与任务定义

### 1.1 数据来源

- 主数据文件：`数据表格测试.xlsx`
- 主要输出目录：`outputs/`、`ensemble_outputs_v2/`、`causal_xgboost_outputs_v2/`、`frontier_scm_v2_outputs/`、`frontier_scm_v3_outputs/`、`frontier_scm_v31_outputs/`
- 目标列：确诊状态三分类

### 1.2 标签定义

| 标签 | 临床含义 | 建模难点 |
|---:|---|---|
| 0 | 非确诊 | 样本极少，最容易被模型并入确诊或灰区 |
| 1 | 确诊 | 多数类，整体准确率主要由该类贡献 |
| 2 | 灰色区域 | 与 0/1 均存在边界相邻性，适合观察模型排序和校准能力 |

### 1.3 样本分布

清洗后共 374 例：

| 类别 | 样本数 | 占比 |
|---:|---:|---:|
| 0 | 22 | 5.9% |
| 1 | 255 | 68.2% |
| 2 | 97 | 25.9% |

该分布决定了单纯 Accuracy 不足以作为主指标。本研究主要使用 `Balanced Accuracy`、`Macro F1`、`0类 Recall`、`Brier score` 和 `ECE` 评价模型。

## 2. 实验路线总览

| 阶段 | 脚本 | 输出目录 | 核心问题 |
|---|---|---|---|
| RF 基线 | `train_random_forest.py` | `outputs/` | 常规树模型能否识别三分类 |
| 异构集成 | `multiclass_ensemble_experiment.py` | `ensemble_outputs_v2/` | XGBoost/LightGBM/CatBoost 集成是否优于单模型 |
| TabPFN | `tabpfn_only_experiment.py`、`tabpfn_cost_sensitive_experiment.py` | `tabpfn_*_outputs/` | 预训练表格模型是否能直接解决小样本 |
| 因果 XGBoost | `causal_xgboost_variants_experiment.py` | `causal_xgboost_outputs_v2/` | 医学先验、顺序损失、DML 是否提升边界 |
| 前沿增广 | `frontier_parallel_experiment.py`、`frontier_augmentation_ablation_experiment.py` | `frontier_*_outputs/` | 结构化增广是否比黑箱重采样更有效 |
| SCM-v2 | `frontier_scm_v2_experiment.py` | `frontier_scm_v2_outputs/` | 基于结构因果混合的稳定少数类增广 |
| SCM-v3 | `frontier_scm_v3_experiment.py` | `frontier_scm_v3_outputs/` | 更复杂的节点采样、规则、教师、课程式策略是否进一步提升 |
| V3 泛化诊断 | `scm_v3_generalization_experiment.py` | `frontier_scm_v3_outputs/generalization_diagnostics/` | train/valid/test gap、校准、bootstrap CI |
| V3.1 | `scm_v31_experiment.py` | `frontier_scm_v31_outputs/` | V2/V3 校准与 TabDDPM 小规模增广网格 |

## 3. 通用预处理与评价协议

### 3.1 数据预处理

药物等效剂量相关字段的缺失具有明确业务含义，缺失按 0 处理；其他数值变量保留缺失指示并用中位数填补；类别变量做编码。受控 ADASYN 或 SCM 生成的数值样本会被裁剪回训练集观测范围，离散型数值变量会做合法值修复，避免生成不可能的临床取值。

### 3.2 主要评价指标

| 指标 | 用途 |
|---|---|
| Accuracy | 整体正确率，但会受多数类支配 |
| Balanced Accuracy | 三类 recall 的平均值，是少数类敏感主指标 |
| Macro F1 | 三类 F1 的平均值，兼顾 precision 与 recall |
| 0类 Recall | 观察非确诊少数类是否被漏检 |
| Brier score | 概率预测质量，越低越好 |
| ECE | 置信度校准误差，越低越好 |
| class0 AP | 0 类 precision-recall 曲线下面积，观察少数类排序质量 |

### 3.3 两类实验协议

本项目存在两种主要实验协议，解释结果时必须区分：

1. 主实验协议：5 个随机种子、80/20 train/test 分层划分，重点比较模型最终测试性能。
2. 泛化诊断协议：5 个随机种子、60/20/20 train/valid/test 三段划分，重点比较 train-test gap、概率校准和模型稳健性。

因此，`SCM_v3` 在主实验协议下略高于 `SCM-v2`，但在 V3.1 三段诊断协议下不如 `SCM-v2` 稳健，这不是矛盾，而是评价问题不同。

## 4. 方案技术原理与详细参数

### 4.1 Random Forest 基线

Random Forest 通过多棵决策树的 bootstrap 集成降低方差，是最早的可解释基线。该模型没有显式处理强类别不平衡，因此主要用于确认任务难度。

关键参数：

| 参数 | 值 |
|---|---:|
| `n_estimators` | 500 |
| `test_size` | 0.2 |
| `random_state` | 42 |

结果显示 RF 的 `0类 Recall=0`，说明传统基线几乎完全忽略非确诊少数类。

### 4.2 XGBoost + 受控 ADASYN 主线

XGBoost 是后续大部分实验的共同分类器骨架。它通过梯度提升树拟合非线性表格关系；受控 ADASYN 用于为少数类生成邻域合成样本，但设置上限以避免过度把少数类扩到不合理规模。

XGBoost 主要参数来自 `causal_xgboost_variants_experiment.py` 的 `XGB_BEST_PARAMS`：

| 参数 | 值 |
|---|---:|
| `objective` | `multi:softprob` |
| `num_class` | 3 |
| `eval_metric` | `mlogloss` |
| `tree_method` | `hist` |
| `n_estimators` | 276 |
| `max_depth` | 5 |
| `learning_rate` | 0.010573 |
| `subsample` | 0.990973 |
| `colsample_bytree` | 0.949733 |
| `min_child_weight` | 2 |
| `reg_alpha` | 0.003511 |
| `reg_lambda` | 0.035499 |

受控 ADASYN 规则：

- 多数类不扩增。
- `0类` 目标上限约为多数类的 65%。
- 其他少数类目标上限约为多数类的 80%。
- 若少数类样本过少或 ADASYN 无法生成有效样本，则跳过。
- 合成后的数值变量裁剪到训练集范围内。

该主线把早期 RF 的少数类失败显著修复，是后续 SCM 系列的强基线。

### 4.3 异构集成与 Stacking

异构集成比较了 XGBoost、LightGBM、CatBoost、等权平均、全局加权、类别感知加权、OOF Stacking、两阶段分层模型和叶节点注入。其思想是利用不同树模型的归纳偏差互补，提高泛化能力。

实验结果显示，最优仍是单模型 `main_xgboost`：

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall |
|---|---:|---:|---:|---:|
| `main_xgboost` | 0.9467 | 0.8869 | 0.8623 | 0.75 |
| `main_catboost` | 0.9467 | 0.8869 | 0.8482 | 0.75 |
| `main_stacking` | 0.9467 | 0.8869 | 0.8482 | 0.75 |

结论是：在当前小样本数据上，复杂集成没有稳定超过调参后的 XGBoost，可能因为可学习信号有限，额外集成带来的方差抵消了互补收益。

### 4.4 TabPFN 系列

TabPFN 是预训练表格分类器，优势是小样本场景下无需大量调参即可获得强概率排序能力。实验包括原始 TabPFN、ADASYN 后 TabPFN 和 cost-sensitive 变体。

代表结果：

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | AUC Macro |
|---|---:|---:|---:|---:|
| `tabpfn_adasyn_base` | 0.9600 | 0.8167 | 0.8708 | 0.9817 |
| `tabpfn_base_original` | 0.9333 | 0.8905 | 0.8417 | 0.9951 |
| `xgboost_baseline_reference` | 0.9467 | 0.8869 | 0.8623 | 0.9702 |

TabPFN 的排序能力较强，AUC 很高，但少数类 recall 和稳定性不总是优于 XGBoost。后续 V3 中的 `TabPFN + SCM_v3` 还出现过总体 Accuracy 提升但 0 类 recall 下降的问题，因此 TabPFN 不建议直接替换主线，而适合做教师、候选概率或外部稳健性对照。

### 4.5 因果增强 XGBoost

因果增强阶段引入了三类思想：

1. 医学变量分组：将药物、筛查、后处理等变量按临床含义分组。
2. 顺序/灰区结构：把 `0 -> 2 -> 1` 的临床邻近性纳入 ordinal 或 EMD 风格目标。
3. DML 与特征权重：用去混杂或特征权重强调临床更相关的变量。

代表结果：

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall |
|---|---:|---:|---:|---:|
| `ordinal_emd_fw_gpl_xgboost` | 0.9600 | 0.8935 | 0.8935 | 0.75 |
| `baseline_xgboost` | 0.9467 | 0.8869 | 0.8623 | 0.75 |
| `ordinal_xgboost` | 0.9333 | 0.8804 | 0.8367 | 0.75 |

结论是因果/顺序约束能改善边界表达，尤其 `ordinal_emd_fw_gpl_xgboost` 在单次测试中表现较好；但这些方案仍未达到 SCM-v2 的跨种子稳健水平。

### 4.6 前沿代理增广：TAP 与 SCM Proxy

前沿并行阶段比较了 TabPFN 参考、ADASYN 参考、TAP proxy 和 SCM proxy。TAP proxy 偏向表格模式生成；SCM proxy 偏向按临床结构生成可解释样本。

代表结果：

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall |
|---|---:|---:|---:|---:|
| `tabpfn_reference` | 0.9733 | 0.8333 | 0.8825 | 0.50 |
| `xgb_scm_proxy` | 0.9467 | 0.8971 | 0.8674 | 0.75 |
| `xgb_reference` | 0.9467 | 0.8869 | 0.8623 | 0.75 |

该阶段的关键启发是：SCM proxy 虽未显著提高 Accuracy，但提高了 Balanced Accuracy，说明结构化增广比纯黑箱重采样更适合少数类边界。

### 4.7 SCM-v2：正式强主线

SCM-v2 的核心是“结构因果混合增广”。它不直接在全特征空间插值，而是按变量角色生成样本：

- 先识别 hard cases，即当前模型容易出错或低置信样本。
- 以 0 类为主要目标类，优先补充非确诊边界。
- 对治疗变量按 `treat_mix_prob` 混合。
- 对后处理或结果相关变量加 residual perturbation。
- 通过 XGBoost 教师过滤，保留更像目标类的样本。
- 最后与原训练集拼接，再进入 XGBoost + 受控 ADASYN + balanced sample weight。

最优配置：

| 参数 | 值 |
|---|---|
| `seed_strategy` | `hard_case_seed` |
| `target_classes` | `class0_only` |
| `treat_mix_prob` | 0.4 |
| `residual_scale` | 0.5 |
| `teacher_mode` | `single` |
| `seeds` | 42, 2024, 2025, 2026, 2027 |

SCM-v2 结果：

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | 增广样本均值 |
|---|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.9840 | 0.9727 | 0.9591 | 0.95 | 72 |

SCM-v2 是第一个在跨种子结果中同时做到高 Accuracy、高 Macro F1 和高 0 类 recall 的方案。

### 4.8 SCM-v3：复杂机制矩阵与最简最优结论

SCM-v3 试图在 SCM-v2 上继续扩展四类机制：

| 机制 | 可选值 | 目的 |
|---|---|---|
| 节点采样 | `residual` / `conditional_kde` | 从残差扰动升级到条件分布采样 |
| 医学规则过滤 | `off` / `medical_rules` | 删除临床上不合理的合成样本 |
| 教师过滤 | `single` / `hetero_consensus` | 从单一 XGBoost 教师升级到异构教师共识 |
| 课程式生成 | `off` / `two_stage` | 先生成保守样本，再生成更强扰动样本 |
| 反事实 | `off` / `treatment_do` | 模拟干预后的样本变化 |

主实验 16 配置矩阵中，最优不是最复杂方案，而是：

`scm_v3_res_norule_single_flat_nocf`

详细参数：

| 参数 | 值 |
|---|---|
| `seed_strategy` | `hard_case_seed` |
| `target_classes` | `class0_only` |
| `treat_mix_prob` | 0.4 |
| `sampler_strength` | 0.5 |
| `node_sampler` | `residual` |
| `rule_filter_mode` | `off` |
| `teacher_filter_mode` | `single` |
| `curriculum_mode` | `off` |
| `counterfactual_mode` | `off` |

主实验结果：

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | 2类 Recall |
|---|---:|---:|---:|---:|---:|
| `xgb_reference_adasyn` | 0.9733 | 0.9081 | 0.9183 | 0.75 | 0.99 |
| `xgb_scm_v2_best` | 0.9840 | 0.9727 | 0.9591 | 0.95 | 0.98 |
| `scm_v3_res_norule_single_flat_nocf` | 0.9840 | 0.9768 | 0.9631 | 0.95 | 1.00 |

解释：V3 的确在主实验口径下小幅超过 V2，但提升来自保守残差结构增广，而不是 KDE、医学规则、异构教师或课程式复杂机制。这说明当前瓶颈更像是小样本方差和信号上限，而不是机制不够复杂。

### 4.9 反事实、TabDDPM 与 GReaT 原型

三类前沿原型承担探索作用：

| 原型 | 原理 | 结论 |
|---|---|---|
| 反事实增广 | 对治疗或临床状态做 do-like 改动，生成边界样本 | 当前弱于 SCM 主线，不建议扩大 |
| TabDDPM | 扩散模型学习 0 类数值特征分布，生成少数类样本 | 有正向信号，值得继续质量控制 |
| GReaT | 用语言模型式表格生成思路模拟外部知识样本 | 有探索价值，但当前不足以替代主线 |

代表结果：

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall |
|---|---:|---:|---:|---:|
| `xgb_tabddpm_proto` | 0.9680 | 0.9229 | 0.9056 | 0.80 |
| `xgb_great_proto` | 0.9653 | 0.9175 | 0.9018 | 0.80 |
| `xgb_scm_v3_reference` | 0.9760 | 0.9422 | 0.9334 | 0.85 |

TabDDPM 与 GReaT 均优于纯早期基线，但没有超过 SCM 主线，因此更适合作为 V3.2 以后的样本质量控制方向。

### 4.10 V3.1：泛化诊断、校准与 TabDDPM 网格

V3.1 实际完成的输出集中在两条线上：

1. 对 `xgb_scm_v2_best` 与 `scm_v3_res_norule_single_flat_nocf` 做未校准、温度校准、OVR sigmoid、OVR isotonic 对比。
2. 对 TabDDPM 做样本数与训练轮数网格：`n=12/24/48`，`epochs=60/120`。

V3.1 使用 5 个随机种子和 60/20/20 三段划分。代表结果：

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best__temperature` | 0.9707 | 0.9048 | 0.8992 | 0.75 | 0.0520 | 0.0427 |
| `xgb_scm_v2_best__uncalibrated` | 0.9707 | 0.9048 | 0.8992 | 0.75 | 0.0564 | 0.0695 |
| `scm_v3_res_norule_single_flat_nocf__temperature` | 0.9653 | 0.8848 | 0.8884 | 0.70 | 0.0598 | 0.0344 |
| `scm_v3_res_norule_single_flat_nocf__uncalibrated` | 0.9653 | 0.8848 | 0.8884 | 0.70 | 0.0631 | 0.0701 |
| `xgb_tabddpm_n24_e60` | 0.9547 | 0.8856 | 0.8672 | 0.70 | 0.0708 | 0.0690 |
| `xgb_tabddpm_n24_e120` | 0.9520 | 0.8803 | 0.8690 | 0.70 | 0.0710 | 0.0741 |

V3.1 结论：

- 温度校准不改变分类标签，但明显改善 Brier 和 ECE，是最稳妥的概率层改进。
- OVR sigmoid 在该小样本条件下伤害 0 类 recall，不建议采用。
- V2 在三段诊断中比 V3 更稳，应作为保守主线。
- TabDDPM 样本有一定少数类排序价值，但还没有达到替代 SCM 的水平。

## 5. 结果综合：从基线到最终主线

| 阶段 | 代表方案 | 协议 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | 解释 |
|---|---|---|---:|---:|---:|---:|---|
| RF | `RandomForestClassifier` | 单次 80/20 | 0.7600 | 0.4840 | 0.4795 | 0.00 | 无法识别 0 类 |
| XGBoost | `main_xgboost` | 单次 80/20 | 0.9467 | 0.8869 | 0.8623 | 0.75 | 强传统主线 |
| TabPFN | `tabpfn_adasyn_base` | 单次 80/20 | 0.9600 | 0.8167 | 0.8708 | 0.50 | AUC 强，但少数类不足 |
| 因果 XGB | `ordinal_emd_fw_gpl_xgboost` | 单次 80/20 | 0.9600 | 0.8935 | 0.8935 | 0.75 | 顺序/因果约束有效但有限 |
| SCM Proxy | `xgb_scm_proxy` | 单次 80/20 | 0.9467 | 0.8971 | 0.8674 | 0.75 | 结构增广方向成立 |
| SCM-v2 | `xgb_scm_v2_best` | 5 种子 80/20 | 0.9840 | 0.9727 | 0.9591 | 0.95 | 当前保守强主线 |
| SCM-v3 | `scm_v3_res_norule_single_flat_nocf` | 5 种子 80/20 | 0.9840 | 0.9768 | 0.9631 | 0.95 | 主实验略优于 V2 |
| V3.1 校准 | `xgb_scm_v2_best__temperature` | 5 种子 60/20/20 | 0.9707 | 0.9048 | 0.8992 | 0.75 | 三段诊断下最稳 |
| V3.1 TabDDPM | `xgb_tabddpm_n24_e60` | 5 种子 60/20/20 | 0.9547 | 0.8856 | 0.8672 | 0.70 | 探索方向，未超过主线 |

## 6. 过拟合与校准解释

### 6.1 是否缓解过拟合

相较早期 XGBoost 和 RF，SCM-v2/V3 明显缓解了少数类边界过拟合。证据包括：

- 0 类 recall 从 RF 的 0.00、XGBoost 的 0.75 提升到 SCM-v2/V3 主实验的 0.95。
- `SCM_v3_best` 在主实验中 `balanced_accuracy_std=0.0402`，低于 V2 的 `0.0513`，说明主实验随机划分下方差有所下降。
- 2 类 recall 在 V3 最优配置中达到 1.00，没有通过牺牲灰区来换取 0 类 recall。

但不能说过拟合已经根本解决：

- V3.1 三段诊断中，`SCM_v3_best` 的测试 Balanced Accuracy 和 Macro F1 低于 V2。
- V3 复杂机制不稳定，KDE、医学规则、异构教师、课程式并未系统获胜。
- 少数类测试支持数很低，0 类 recall 对单个样本非常敏感。

因此，更严谨的表述是：SCM 系列部分缓解了小样本少数类边界过拟合，但最终稳健性仍依赖简单、保守、可控的增广策略。

### 6.2 概率校准结论

医疗任务中，高分类分数不等于可靠概率。V3.1 显示：

- 温度校准可在不改变 Accuracy/Macro F1 的情况下改善 Brier 和 ECE。
- `xgb_scm_v2_best` 温度校准后 Brier 从 0.0564 降至 0.0520，ECE 从 0.0695 降至 0.0427。
- `SCM_v3_best` 温度校准后 ECE 从 0.0701 降至 0.0344，但分类性能仍低于 V2。
- OVR sigmoid 对小样本少数类不稳定，不建议用于主报告。

## 7. 带注释的关键实验代码

以下代码片段摘自当前目录中的实验脚本，并补充中文注释，便于论文方法部分说明。

### 7.1 受控 ADASYN：避免少数类过度合成

来源：`multiclass_ensemble_experiment.py`

```python
def build_sampling_strategy(y_train: pd.Series) -> dict[int, int]:
    counts = Counter(y_train)
    majority_count = max(counts.values())
    strategy: dict[int, int] = {}

    for cls, count in counts.items():
        if count == majority_count:
            continue

        # 0 类是最小少数类，但不能无限扩增，否则会制造大量虚假边界。
        # 因此 0 类最多扩到多数类的 65%，其他少数类最多扩到 80%。
        max_ratio = 0.65 if int(cls) == 0 else 0.80
        desired = int(min(count * 4, majority_count * max_ratio))

        # 至少需要有实际新增样本，才启用 ADASYN。
        if desired > count + 2:
            strategy[int(cls)] = desired

    return strategy
```

```python
sampler = ADASYN(
    sampling_strategy=strategy,
    n_neighbors=min(5, minority_count - 1),
    random_state=random_state,
)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

# 合成样本仍必须落在训练集观测范围内，避免生成临床上极端异常值。
for col in X_train.columns:
    if col.startswith("num__") and not col.startswith("num__missingindicator"):
        X_resampled[col] = X_resampled[col].clip(
            lower=X_train[col].min(),
            upper=X_train[col].max(),
        )
```

### 7.2 XGBoost 主分类器：统一后续实验骨架

来源：`frontier_scm_v2_experiment.py`、`causal_xgboost_variants_experiment.py`

```python
XGB_BEST_PARAMS = {
    "n_estimators": 276,
    "max_depth": 5,
    "learning_rate": 0.010573268083515799,
    "subsample": 0.9909729556485982,
    "colsample_bytree": 0.9497327922401265,
    "min_child_weight": 2,
    "reg_alpha": 0.0035113563139704067,
    "reg_lambda": 0.03549878832196503,
}

def make_xgb_classifier(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",   # 输出三分类概率
        num_class=3,
        eval_metric="mlogloss",       # 与概率质量相关
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )
```

```python
# balanced sample weight 与受控 ADASYN 同时使用：
# ADASYN 改变训练样本几何分布，sample_weight 继续提醒模型类别不均衡。
sample_weight = compute_sample_weight(class_weight="balanced", y=y_model)
model.fit(X_model_df, y_model, sample_weight=sample_weight, verbose=False)
```

### 7.3 SCM-v3 最优配置：简单残差增广反而最好

来源：`scm_v3_generalization_experiment.py`

```python
SCM_V3_BEST_CONFIG = {
    "seed_strategy": "hard_case_seed",   # 从难例附近生成，聚焦真实边界
    "target_classes": "class0_only",     # 只补最稀缺的非确诊类
    "treat_mix_prob": 0.4,               # 治疗变量混合概率
    "sampler_strength": 0.5,             # 残差扰动强度
    "node_sampler": "residual",          # 不使用 KDE，降低小样本分布估计误差
    "rule_filter_mode": "off",           # 不启用额外医学规则过滤
    "teacher_filter_mode": "single",     # 单教师过滤，减少异构教师不一致
    "counterfactual_mode": "off",        # 不启用反事实干预
}
```

```python
hard_case_map = build_hard_case_index_map(
    split_data.X_train_raw,
    split_data.y_train,
    seed,
)

augmentor = SCMMixV3Augmentor(
    random_state=seed,
    project_dir=PROJECT_DIR,
    use_remote_teacher=False,
    **SCM_V3_BEST_CONFIG,
)

aug_result = augmentor.generate(
    split_data.X_train_raw,
    split_data.y_train,
    hard_case_map,
)

# 生成样本与原始训练集拼接，然后进入统一 XGBoost 训练管线。
X_fit = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
y_fit = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)
```

这段代码体现了 V3 最重要的经验：在样本很少时，保守残差增广优于更复杂的 KDE、规则、异构教师和课程式堆叠。

### 7.4 V3.1 温度校准：改善概率但不改分类边界

来源：`scm_v31_experiment.py`

```python
def apply_probability_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    # 将概率转为 logit-like 空间，再除以温度。
    # T > 1 会软化过度自信概率，T < 1 会增强置信度。
    logits = np.log(np.clip(proba, 1e-8, 1.0))
    scaled = np.exp(logits / max(float(temperature), 1e-8))
    return normalize_probability_rows(scaled)


def fit_temperature(valid_proba: np.ndarray, y_valid: pd.Series) -> dict[str, float]:
    candidates = np.unique(
        np.concatenate([
            np.linspace(0.5, 2.0, 16),
            np.linspace(2.25, 4.0, 8),
        ])
    )

    best_temperature = 1.0
    best_loss = float("inf")

    # 在 valid 集上选择使 multiclass log loss 最低的温度。
    for temperature in candidates:
        calibrated = apply_probability_temperature(valid_proba, float(temperature))
        loss = multiclass_log_loss(y_valid, calibrated)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)

    return {"temperature": best_temperature, "valid_log_loss": best_loss}
```

温度校准只调整概率分布，不改变 argmax 分类结果，因此适合医疗报告中补充“风险概率是否可信”的分析。

### 7.5 TabDDPM 原型：只对 0 类学习生成分布

来源：`tabddpm_prototype_experiment.py`

```python
class Denoiser(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden = max(32, input_dim * 2)
        self.time_embed = nn.Embedding(64, 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_feat = self.time_embed(t)
        return self.net(torch.cat([x, time_feat], dim=1))
```

```python
for _ in range(epochs):
    batch_indices = rng.integers(0, len(X_tensor), size=min(32, len(X_tensor)))
    batch = X_tensor[batch_indices]
    t = torch.randint(0, n_steps, (len(batch),), device=DEVICE)
    noise = torch.randn_like(batch)

    # 将真实 0 类样本加噪，训练网络预测噪声。
    alpha = 0.90 + 0.09 * (t.float() / max(n_steps - 1, 1))
    noisy = alpha.unsqueeze(1) * batch + (1 - alpha.unsqueeze(1)) * noise
    noise_pred = model(noisy, t)

    loss = F.mse_loss(noise_pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

TabDDPM 当前原型只学习 0 类数值特征分布，默认 `epochs=120`，V3.1 网格补充了 `epochs=60/120` 与 `n_samples=12/24/48`。结果显示它有探索价值，但生成样本质量仍不足以超过 SCM 主线。

## 8. 论文写作建议

### 8.1 推荐主结论

建议论文主结论写为：

> 在 ARR 小样本三分类任务中，单纯 Random Forest 无法识别极少数非确诊类；调参 XGBoost 与受控 ADASYN 可显著提升少数类 recall；进一步引入结构因果混合增广后，SCM-v2 在跨随机种子实验中获得最佳稳健性。SCM-v3 在主实验协议下略优于 SCM-v2，但 V3.1 三段泛化诊断显示其稳定性不足以替代 V2。温度校准可在不改变分类性能的前提下改善概率可信度，因此最终建议采用 `xgb_scm_v2_best + temperature calibration` 作为保守主线，`SCM_v3_best` 和 TabDDPM 作为后续增强研究方向。

### 8.2 模型推荐

| 用途 | 推荐方案 | 理由 |
|---|---|---|
| 正式主线 | `xgb_scm_v2_best__temperature` | V3.1 三段诊断最稳，概率更可靠 |
| 主实验增强候选 | `scm_v3_res_norule_single_flat_nocf__temperature` | 主实验指标最高，结构简单 |
| 前沿探索 | `xgb_tabddpm_n24_e60` 或 `xgb_tabddpm_n24_e120` | 有少数类排序信号，但需质量过滤 |
| 不建议作为主线 | `TabPFN + SCM_v3`、OVR sigmoid 校准、反事实扩矩阵 | 已显示少数类 recall 或稳定性风险 |

## 9. 局限性

1. 数据规模较小，尤其 0 类只有 22 例，任何测试集中的 0 类指标都对单个样本极敏感。
2. 当前结果仍主要来自内部随机划分，尚缺外部验证队列。
3. SCM 增广依赖当前特征分组和教师模型，若数据采集口径改变，需要重新审计。
4. TabDDPM/GReaT 原型还未形成严格样本质量控制闭环。
5. V3.1 当前目录结果聚焦校准与 TabDDPM 网格，未在输出中看到完整 `no_post_adasyn`、`capped_aug60`、`hybrid_tabddpm25` 独立命名结果，因此这些计划项不纳入已完成实验证据。

## 10. 最终结论

从全部实验链条看，项目已经从“多数类驱动的高 Accuracy”推进到“少数类边界可控、跨种子稳健、概率可校准”的阶段。最关键的技术贡献不是堆叠更复杂模型，而是证明了结构因果混合增广在极小少数类三分类诊断任务中的价值。

当前最稳妥的论文版结论是：

- `SCM-v2 + XGBoost + temperature calibration` 是正式主线。
- `SCM_v3_best` 是有价值的候选增强，但暂不替代 V2。
- 后续真正值得投入的是外部验证、概率校准、TabDDPM 样本质量控制，而不是继续扩大 SCM-v3 复杂机制矩阵。
