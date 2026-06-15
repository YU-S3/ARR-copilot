# 基于 ARR 临床真实世界数据的 SCM_v3 结构化增广实验报告

## 1. 研究背景与目标

本轮实验是在前期 `Random Forest`、`XGBoost`、多策略集成、`TabPFN-3`、因果增强 XGBoost 以及 `SCM-v2` 的基础上，继续围绕 ARR 小样本三分类任务推进结构化数据增广。

核心问题有三个：

1. `SCM_v3` 是否能在强基线 `SCM-v2` 之上继续提升总体性能和少数类稳健性。
2. 更复杂的增广控制机制，例如条件 KDE、医学规则、异构教师、课程式增广，是否真正带来收益。
3. `TabDDPM`、`GReaT`、反事实生成等前沿原型是否已经具备替代或扩展主线的价值。

本报告目的不是只给最终排行榜，而是完整说明每个方案的原理、对应脚本、实验设置、输出文件、结果和解释，为后续论文撰写、项目汇报和下一轮实验设计提供依据。

## 2. 数据来源与任务定义

### 2.1 数据文件

- 主数据文件：`C:\Users\YCY\Desktop\ARR-model\数据表格测试.xlsx`
- 任务类型：三分类临床表格预测
- 目标变量：确诊结果三分类标签

### 2.2 标签定义

- `0`：非确诊
- `1`：确诊
- `2`：灰色区域

### 2.3 数据规模与类别不均衡

清洗后有效三分类样本共 `374` 例：

- `0 = 非确诊`：`22`
- `1 = 确诊`：`255`
- `2 = 灰色区域`：`97`

该分布显示任务存在明显类别不均衡，尤其是 `0` 类样本非常少。因此，本轮实验除 Accuracy 外，更关注：

- `Balanced Accuracy`
- `Macro F1`
- `0类 Recall`
- `0类 precision-recall 曲线`
- 泛化 gap
- Brier score 与 ECE

## 3. 共同预处理与基础建模口径

### 3.1 共同预处理

本轮实验沿用既有三分类流水线：

- 数值变量：中位数填补，并保留缺失指示变量。
- 类别变量：缺失值填充为 `Missing`，再进行编码。
- 药物相关指定列：按业务含义将缺失值视为 `0`。
- 目标标签：仅保留合法的 `0 / 1 / 2` 样本。

相关实现主要来自：

- `multiclass_ensemble_experiment.py`
- `frontier_augmentation.py`
- `frontier_scm_v2_experiment.py`
- `scm_v3_augmentation.py`

### 3.2 基础分类器

本轮主线的最终判别模型仍以 `XGBoost` 为核心，使用：

- 多分类 softprob 输出
- 受控 `ADASYN`
- `class_weight="balanced"` 对应的样本权重
- 与既有最佳 XGBoost 参数保持一致

这样做的目的是保证 `SCM-v2`、`SCM_v3` 和各类前沿原型的主要差异来自“训练数据如何增广”，而不是来自分类器本身更换。

## 4. 实验总览

| 实验 | 方案 | 主脚本 | 输出目录 | 实验目的 |
|---|---|---|---|---|
| 实验一 | `xgb_reference_adasyn` | `frontier_scm_v3_experiment.py` | `frontier_scm_v3_outputs/` | 提供统一 XGBoost + ADASYN 基线 |
| 实验二 | `xgb_scm_v2_best` | `frontier_scm_v3_experiment.py` 调用 `frontier_scm_v2_experiment.py` | `frontier_scm_v3_outputs/` | 保留强 `SCM-v2` 对照 |
| 实验三 | `SCM_v3` 16 配置消融 | `frontier_scm_v3_experiment.py` | `frontier_scm_v3_outputs/` | 比较 V3 新机制是否有效 |
| 实验四 | `TabPFN + SCM_v3` | `frontier_scm_v3_experiment.py` | `frontier_scm_v3_outputs/tables/tabpfn_*` | 验证 SCM 增广是否适合 TabPFN |
| 实验五 | 反事实增广原型 | `counterfactual_augmentation_experiment.py` | `counterfactual_aug_outputs/` | 探索 do-like 反事实生成 |
| 实验六 | `TabDDPM` 原型 | `tabddpm_prototype_experiment.py` | `tabddpm_proto_outputs/` | 探索扩散式表格生成 |
| 实验七 | `GReaT` 原型 | `great_prototype_experiment.py` | `great_proto_outputs/` | 探索语言模型风格表格生成 |
| 实验八 | 泛化与校准诊断 | `scm_v3_generalization_experiment.py` | `frontier_scm_v3_outputs/generalization_diagnostics/` | 评估过拟合、校准和置信度质量 |

正式主实验使用 5 个随机种子：

- `42`
- `2024`
- `2025`
- `2026`
- `2027`

## 5. 实验一：XGBoost + 受控 ADASYN 基线

### 5.1 实验目的

该实验作为本轮所有结构化增广方法的统一传统机器学习基线。它回答的问题是：在不引入 SCM 生成样本的情况下，仅使用当前最稳的 XGBoost 分类器、类别权重和受控 `ADASYN`，模型能达到什么水平。

### 5.2 方法原理

`xgb_reference_adasyn` 使用真实训练集进行预处理，然后在特征空间中对少数类执行受控 `ADASYN`。与普通过采样相比，受控版本会限制目标采样比例，避免把极少数类机械扩增到过高比例，从而降低噪声样本对决策边界的污染。

该方案没有引入医学结构，也不修改变量之间的因果顺序。它是一个“强但不带结构先验”的基线。

### 5.3 对应实验与输出

- 主脚本：`frontier_scm_v3_experiment.py`
- 关键函数：`fit_xgb_with_safe_adasyn`
- 输出表：
  - `frontier_scm_v3_outputs/tables/metrics_by_seed.csv`
  - `frontier_scm_v3_outputs/tables/metrics_mean_std.csv`

### 5.4 实验结果

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.9733 |
| Balanced Accuracy | 0.9081 |
| Macro F1 | 0.9183 |
| Weighted F1 | 0.9729 |
| OVR ROC AUC Macro | 0.9907 |
| 0类 Recall | 0.75 |
| 1类 Recall | 0.9843 |
| 2类 Recall | 0.99 |

### 5.5 结果分析

该基线已经很强，尤其 Accuracy 和 AUC 都较高。但 `0` 类 recall 只有 `0.75`，说明少数类边界仍不够稳。后续 SCM 增广的主要价值，应体现在提升 `Balanced Accuracy`、`Macro F1` 和 `0类 Recall`，而不是只提升总体 Accuracy。

## 6. 实验二：SCM-v2 强基线复现

### 6.1 实验目的

`SCM-v2` 是上一阶段已经验证有效的结构化增广方案。本轮保留它作为强对照，用来判断 `SCM_v3` 的改进是否真的超过已有主线。

### 6.2 方法原理

`SCM-v2` 的核心思想是按照临床变量的结构顺序生成少数类样本，而不是在所有特征上做无结构扰动。

其生成逻辑可以概括为：

1. 通过交叉验证找出训练集中更难分类的样本，形成 hard-case index。
2. 优先围绕 `0` 类 hard cases 生成增广样本。
3. 按变量结构顺序生成：
   - 外生变量保持来自原始样本。
   - 治疗/用药变量按一定概率与 donor 样本混合。
   - 筛查变量根据父节点变量拟合回归节点并采样残差。
   - 确诊后变量再基于更下游的父节点生成。
4. 使用教师模型过滤候选样本，保证生成样本仍被教师模型判为目标类。
5. 将生成样本与原训练集拼接，再训练 XGBoost。

该方案的本质是“结构一致的少数类定向增广”。

### 6.3 实验设置

本轮引用的 `SCM-v2` 配置为：

- `seed_strategy`: `hard_case_seed`
- `target_classes`: `class0_only`
- `treat_mix_prob`: `0.4`
- `residual_scale`: `0.5`
- `teacher_mode`: `single`

平均增广样本数：`72`

### 6.4 对应实验与输出

- 主脚本：`frontier_scm_v3_experiment.py`
- 被调用脚本：`frontier_scm_v2_experiment.py`
- 输出表：
  - `frontier_scm_v3_outputs/tables/metrics_mean_std.csv`

### 6.5 实验结果

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.9840 |
| Balanced Accuracy | 0.9727 |
| Macro F1 | 0.9591 |
| 0类 Recall | 0.95 |
| 1类 Recall | 0.9882 |
| 2类 Recall | 0.98 |

### 6.6 结果分析

`SCM-v2` 相比纯 `xgb_reference_adasyn` 有明显提升，尤其 `0类 Recall` 从 `0.75` 提高到 `0.95`。这说明结构化少数类增广确实比单纯 ADASYN 更能稳定少数类边界。

也正因为 `SCM-v2` 已经很强，`SCM_v3` 的任务不是证明 SCM 有效，而是证明 V3 的新增机制还能在强 SCM 基线上继续带来增益。

## 7. 实验三：SCM_v3 主线消融实验

### 7.1 实验目的

`SCM_v3` 不是单一方案，而是一组围绕 `SCM-v2` 的机制升级。该实验的目标是系统比较这些新机制是否有效，并找出当前数据规模下最稳的组合。

### 7.2 SCM_v3 相比 SCM-v2 的新增机制

`SCM_v3` 主要新增四类机制。

第一，节点采样方式升级：

- `residual`：沿用残差采样，在节点回归预测值上叠加训练残差。
- `conditional_kde`：在局部邻域内使用条件 KDE 采样，希望更细致地模拟局部分布。

第二，医学规则过滤：

- `off`：不启用医学规则。
- `medical_rules`：候选样本生成后，使用 `scm_v3_medical_rules.py` 中的规则检查变量组合是否符合基本医学逻辑。

第三，教师过滤方式：

- `single`：单一 XGBoost 教师过滤候选样本。
- `hetero_consensus`：使用 XGBoost、LightGBM 等异构教师形成一致性过滤，降低单一教师误判风险。

第四，课程式增广：

- `flat`：一次性生成目标样本。
- `two_stage`：先生成更保守样本，再生成更强扰动样本，希望降低训练早期噪声。

### 7.3 SCM_v3 生成流程

`SCM_v3` 的生成流程如下：

1. 在训练集上构建 hard-case map，优先选择难分类的 `0` 类样本作为增广锚点。
2. 按变量组划分：
   - 外生变量
   - 治疗变量
   - 筛查变量
   - 确诊后变量
3. 对筛查变量和确诊后变量分别拟合节点模型。
4. 根据配置选择 residual 或 conditional KDE 生成节点值。
5. 对治疗变量按 `treat_mix_prob` 做 donor 混合。
6. 根据配置执行医学规则过滤。
7. 根据配置执行单教师或异构教师过滤。
8. 通过过滤的候选样本进入增广集。
9. 增广集与原训练集合并，再训练 XGBoost。

### 7.4 实验矩阵

本轮 `SCM_v3` 主实验共 `16` 个配置：

- `node_sampler`: `residual` / `conditional_kde`
- `rule_filter_mode`: `off` / `medical_rules`
- `teacher_filter_mode`: `single` / `hetero_consensus`
- `curriculum_mode`: `off` / `two_stage`

统一固定：

- `seed_strategy`: `hard_case_seed`
- `target_classes`: `class0_only`
- `treat_mix_prob`: `0.40`
- `counterfactual_mode`: `off`

### 7.5 对应实验与输出

- 主脚本：`frontier_scm_v3_experiment.py`
- 增广实现：`scm_v3_augmentation.py`
- 医学规则：`scm_v3_medical_rules.py`
- 输出目录：`frontier_scm_v3_outputs/`
- 主要输出：
  - `tables/metrics_by_seed.csv`
  - `tables/metrics_mean_std.csv`
  - `tables/overall_leaderboard.csv`
  - `tables/minority_leaderboard.csv`
  - `tables/augmentation_audit.csv`
  - `figures/balanced_accuracy_boxplot.png`
  - `figures/macro_f1_boxplot.png`
  - `experiment_summary.json`

### 7.6 主实验结果总表

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | 1类 Recall | 2类 Recall | 平均增广样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `xgb_reference_adasyn` | 0.9733 | 0.9081 | 0.9183 | 0.75 | 0.9843 | 0.99 | 0 |
| `xgb_scm_v2_best` | 0.9840 | 0.9727 | 0.9591 | 0.95 | 0.9882 | 0.98 | 72 |
| `scm_v3_res_norule_single_flat_nocf` | 0.9840 | 0.9768 | 0.9631 | 0.95 | 0.9804 | 1.00 | 90 |

### 7.7 最优 SCM_v3 配置

本轮整体榜和少数类稳健性榜的最优方案均为：

`scm_v3_res_norule_single_flat_nocf`

对应配置：

- `node_sampler`: `residual`
- `rule_filter_mode`: `off`
- `teacher_filter_mode`: `single`
- `curriculum_mode`: `off`
- `counterfactual_mode`: `off`
- `sampler_strength`: `0.5`
- `treat_mix_prob`: `0.4`

### 7.8 最优方案命名解释

`scm_v3_res_norule_single_flat_nocf` 这一名称本身就是实验配置的压缩编码，可以拆成六段理解：

| 名称片段 | 含义 | 在本方案中的选择 |
|---|---|---|
| `scm_v3` | 使用第三版结构因果增广框架 | 固定为 `SCM_v3` |
| `res` | 节点生成方式 | 使用 `residual` 残差采样 |
| `norule` | 是否启用医学规则过滤 | 不启用医学规则过滤 |
| `single` | 教师模型过滤方式 | 使用单一 XGBoost 教师 |
| `flat` | 增广流程 | 单阶段生成，不使用两阶段课程式增广 |
| `nocf` | 是否启用反事实干预 | 不启用 `treatment_do` 反事实干预 |

因此，该方案的完整含义是：

优先选择训练集中难分类的 `0类` 样本作为锚点，只针对少数类进行结构化补样；治疗变量按一定概率与 donor 样本混合；筛查变量和确诊后变量按 SCM 父子结构顺序生成；连续节点采用回归预测值加残差的方式采样；候选样本只经过一个 XGBoost 教师模型筛选；最后将通过筛选的增广样本与原训练集拼接，再训练最终 XGBoost 分类器。

它是本轮 `SCM_v3` 中最简单、最少额外约束的版本。实验结果显示，在当前小样本 ARR 数据上，简单结构增广比条件 KDE、医学规则过滤、异构教师一致性和课程式两阶段增广更稳定。

### 7.9 结果分析

从主实验 5 种子均值看，`SCM_v3` 相比纯基线提升明显：

- `Accuracy: 0.9733 -> 0.9840`
- `Balanced Accuracy: 0.9081 -> 0.9768`
- `Macro F1: 0.9183 -> 0.9631`
- `0类 Recall: 0.75 -> 0.95`

相比 `SCM-v2`，`SCM_v3` 提升很小：

- `Balanced Accuracy: 0.9727 -> 0.9768`
- `Macro F1: 0.9591 -> 0.9631`
- `2类 Recall: 0.98 -> 1.00`

更重要的是，最优方案不是复杂机制组合，而是最简单的 residual + single teacher + flat 版本。这说明在当前数据规模下：

1. 条件 KDE 并没有稳定超过残差采样。
2. 医学规则过滤可能过度筛掉可用候选样本。
3. 异构教师虽然更保守，但不一定提升最终泛化。
4. 两阶段课程式增广增加样本和流程复杂度，却没有稳定收益。

因此，`SCM_v3` 的实际收益主要来自更稳的简单结构增广，而不是复杂机制堆叠。

## 8. 实验四：TabPFN + SCM_v3 扩展实验

### 8.1 实验目的

该实验用于验证：`SCM_v3` 生成样本是否也能提升表格基础模型 `TabPFN` 的表现。

### 8.2 方法原理

实验包含两个方案：

1. `tabpfn_reference_v3`：直接使用 `TabPFN` 在原始训练集上推理。
2. `tabpfn_scm_v3_best`：使用 `SCM_v3` 最优配置生成增广样本，再训练或调用 TabPFN 进行预测。

其核心假设是：如果 SCM 生成样本真实补充了少数类边界，那么它不只应提升 XGBoost，也可能提升 TabPFN。

### 8.3 对应实验与输出

- 主脚本：`frontier_scm_v3_experiment.py`
- 输出表：
  - `frontier_scm_v3_outputs/tables/tabpfn_metrics_by_seed.csv`
  - `frontier_scm_v3_outputs/tables/tabpfn_metrics_mean_std.csv`

### 8.4 实验结果

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | OVR ROC AUC Macro | 0类 Recall |
|---|---:|---:|---:|---:|---:|
| `tabpfn_reference_v3` | 0.9440 | 0.9418 | 0.8747 | 0.9949 | 0.90 |
| `tabpfn_scm_v3_best` | 0.9680 | 0.8768 | 0.8944 | 0.9895 | 0.65 |

### 8.5 结果分析

`TabPFN + SCM_v3` 提高了 Accuracy 和 Macro F1，但明显伤害少数类稳健性：

- Accuracy：`0.9440 -> 0.9680`
- Macro F1：`0.8747 -> 0.8944`
- Balanced Accuracy：`0.9418 -> 0.8768`
- 0类 Recall：`0.90 -> 0.65`

这说明 `TabPFN` 对增广样本的响应和 XGBoost 不同。SCM 样本对 XGBoost 是边界补充，但对 TabPFN 可能改变了其内部先验或类别概率平衡，导致少数类召回下降。

因此，当前不建议把 `TabPFN + SCM_v3` 作为正式主线，只能作为“总体性能提升但少数类风险较高”的探索结果。

## 9. 实验五：反事实增广原型

### 9.1 实验目的

反事实增广原型用于探索：如果对治疗变量执行类似 `do` 干预的变化，能否生成更有判别价值的少数类样本。

### 9.2 方法原理

该方案不只是沿着原始分布采样，而是尝试修改部分治疗相关变量，生成“如果治疗状态不同，其他变量如何变化”的候选样本。它的目标是模拟因果反事实场景，提升模型对治疗扰动下样本的鲁棒性。

当前实现仍是最小原型，生成逻辑较保守，尚未形成完整临床因果模拟。

### 9.3 对应实验与输出

- 主脚本：`counterfactual_augmentation_experiment.py`
- 输出目录：`counterfactual_aug_outputs/`
- 主要输出：
  - `tables/metrics_by_seed.csv`
  - `tables/metrics_mean_std.csv`
  - `tables/augmentation_audit.csv`
  - `experiment_summary.json`

### 9.4 实验结果

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | OVR ROC AUC Macro | 0类 Recall | 平均增广样本数 |
|---|---:|---:|---:|---:|---:|---:|
| `xgb_reference_adasyn` | 0.9733 | 0.9081 | 0.9183 | 0.9907 | 0.75 | 0 |
| `xgb_scm_v3_reference` | 0.9733 | 0.9255 | 0.9196 | 0.9915 | 0.80 | 90 |
| `xgb_counterfactual_proto` | 0.9653 | 0.8755 | 0.8772 | 0.9904 | 0.65 | 18 |

### 9.5 结果分析

反事实原型当前不成功：

- 总体 Accuracy 低于基线。
- Balanced Accuracy 低于基线。
- Macro F1 低于基线。
- 0类 Recall 从基线的 `0.75` 降至 `0.65`。

这说明当前反事实生成样本很可能没有稳定落在真实临床分布附近，或者干预后变量联动不够可信。该路线不建议直接扩大实验矩阵，应先重构反事实生成逻辑。

## 10. 实验六：TabDDPM 表格扩散原型

### 10.1 实验目的

`TabDDPM` 原型用于探索扩散模型能否学习少数类样本的连续变量分布，并生成比简单重采样更有用的训练样本。

### 10.2 方法原理

当前原型采用轻量级表格扩散思路：

1. 选取 `0` 类样本作为生成目标。
2. 对治疗、筛查和确诊后相关数值变量建模。
3. 使用小型 denoiser 学习加噪后样本的还原方向。
4. 从噪声中逐步反推生成候选数值变量。
5. 对生成值进行范围裁剪和离散变量修正。
6. 将生成样本并入训练集，再训练 XGBoost。

该实现是原型版，不是完整工业级 TabDDPM。

### 10.3 对应实验与输出

- 主脚本：`tabddpm_prototype_experiment.py`
- 输出目录：`tabddpm_proto_outputs/`
- 主要输出：
  - `tables/metrics_by_seed.csv`
  - `tables/metrics_mean_std.csv`
  - `experiment_summary.json`

### 10.4 实验结果

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | OVR ROC AUC Macro | 0类 Recall | 平均增广样本数 |
|---|---:|---:|---:|---:|---:|---:|
| `xgb_reference_adasyn` | 0.9733 | 0.9081 | 0.9183 | 0.9907 | 0.75 | 0 |
| `xgb_scm_v3_reference` | 0.9760 | 0.9422 | 0.9334 | 0.9918 | 0.85 | 90 |
| `xgb_tabddpm_proto` | 0.9680 | 0.9229 | 0.9056 | 0.9909 | 0.80 | 36 |

### 10.5 结果分析

`TabDDPM` 原型有正向信号：

- 0类 Recall 从 `0.75` 提高到 `0.80`。
- Balanced Accuracy 从 `0.9081` 提高到 `0.9229`。

但它仍弱于 `SCM_v3_reference`：

- Macro F1 低于 SCM。
- Balanced Accuracy 低于 SCM。
- 0类 Recall 低于 SCM。

因此，`TabDDPM` 当前不能替代主线，但值得继续作为前沿方向优化。它的价值在于提供了一条不同于 SCM 规则结构的生成路线。

## 11. 实验七：GReaT 风格表格生成原型

### 11.1 实验目的

`GReaT` 原型用于探索语言模型风格的表格生成是否能把外部语义和表格上下文转化为有效训练样本。

### 11.2 方法原理

当前版本更接近“模板化 GReaT 风格生成”：

1. 将表格行转换成文本或结构化描述。
2. 按目标类别生成或扰动少数类样本。
3. 将生成结果映射回表格特征。
4. 进行基础范围检查和类型修正。
5. 与原训练集合并后训练 XGBoost。

由于当前实现不是强本地大模型版本，也没有复杂医学 prompt，因此更适合作为探索性分支。

### 11.3 对应实验与输出

- 主脚本：`great_prototype_experiment.py`
- 输出目录：`great_proto_outputs/`
- 主要输出：
  - `tables/metrics_by_seed.csv`
  - `tables/metrics_mean_std.csv`
  - `experiment_summary.json`

### 11.4 实验结果

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | OVR ROC AUC Macro | 0类 Recall | 平均增广样本数 |
|---|---:|---:|---:|---:|---:|---:|
| `xgb_reference_adasyn` | 0.9733 | 0.9081 | 0.9183 | 0.9907 | 0.75 | 0 |
| `xgb_scm_v3_reference` | 0.9760 | 0.9422 | 0.9334 | 0.9918 | 0.85 | 90 |
| `xgb_great_proto` | 0.9653 | 0.9175 | 0.9018 | 0.9865 | 0.80 | 36 |

### 11.5 结果分析

`GReaT` 原型也有一定正向信号：

- 0类 Recall 从 `0.75` 提高到 `0.80`。
- Balanced Accuracy 从 `0.9081` 提高到 `0.9175`。

但它仍弱于 SCM 主线，并且 AUC 略低。当前版本不适合作为正式主线，更适合后续升级为“强本地模型 + 医学 prompt”的探索方向。

## 12. 实验八：泛化与校准诊断

### 12.1 实验目的

前面主实验主要报告 test 均值，但还不能直接回答“过拟合是否缓解”。因此补充该实验，专门记录 train / valid / test 三套指标，并计算泛化 gap、Brier、ECE 和 0类 PR 曲线。

### 12.2 方法原理

该实验重新使用 5 个随机种子划分数据：

- train：60%
- valid：20%
- test：20%

对以下模型并排评估：

- `xgb_reference_adasyn`
- `xgb_scm_v2_best`
- `scm_v3_res_norule_single_flat_nocf`
- `xgb_tabddpm_proto`

核心新增指标：

- `macro_f1_train_test_gap`
- `balanced_accuracy_train_test_gap`
- Brier score
- ECE
- 0类 average precision
- 0类 precision-recall curve

### 12.3 对应实验与输出

- 主脚本：`scm_v3_generalization_experiment.py`
- 输出目录：`frontier_scm_v3_outputs/generalization_diagnostics/`
- 主要输出：
  - `generalization_report.md`
  - `tables/split_metrics_by_seed.csv`
  - `tables/generalization_gaps_by_seed.csv`
  - `tables/test_bootstrap_ci.csv`
  - `figures/class0_precision_recall_curve.png`
  - `figures/test_calibration_curve.png`
  - `figures/generalization_gap_boxplot.png`

### 12.4 三段划分 test 均值

| 模型 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.9707 | 0.9048 | 0.8992 | 0.75 | 0.0564 | 0.0695 |
| `scm_v3_res_norule_single_flat_nocf` | 0.9653 | 0.8848 | 0.8884 | 0.70 | 0.0631 | 0.0701 |
| `xgb_tabddpm_proto` | 0.9520 | 0.8690 | 0.8536 | 0.65 | 0.0646 | 0.0795 |
| `xgb_reference_adasyn` | 0.9573 | 0.8542 | 0.8442 | 0.60 | 0.0648 | 0.0689 |

### 12.5 泛化 gap

| 模型 | Macro F1 Train-Test Gap | Balanced Accuracy Train-Test Gap | Test Brier | Test ECE |
|---|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.1008 | 0.0952 | 0.0564 | 0.0695 |
| `scm_v3_res_norule_single_flat_nocf` | 0.1116 | 0.1152 | 0.0631 | 0.0701 |
| `xgb_tabddpm_proto` | 0.1464 | 0.1310 | 0.0646 | 0.0795 |
| `xgb_reference_adasyn` | 0.1558 | 0.1458 | 0.0648 | 0.0689 |

### 12.6 Bootstrap test CI

| 模型 | Macro F1 Mean | Macro F1 CI | Balanced Accuracy Mean | Balanced Accuracy CI | 0类 Recall Mean | 0类 Recall CI |
|---|---:|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.8996 | 0.8394 - 0.9511 | 0.9037 | 0.8393 - 0.9617 | 0.7462 | 0.5554 - 0.9231 |
| `scm_v3_res_norule_single_flat_nocf` | 0.8900 | 0.8282 - 0.9421 | 0.8866 | 0.8154 - 0.9504 | 0.7062 | 0.5000 - 0.8847 |
| `xgb_tabddpm_proto` | 0.8582 | 0.7927 - 0.9162 | 0.8703 | 0.7936 - 0.9380 | 0.6546 | 0.4286 - 0.8637 |
| `xgb_reference_adasyn` | 0.8549 | 0.7851 - 0.9174 | 0.8540 | 0.7757 - 0.9268 | 0.5996 | 0.3684 - 0.8182 |

### 12.7 结果分析

该诊断结果修正了主实验的单一结论。

在原始 5 种子主实验口径下，`SCM_v3` 略优于 `SCM-v2`。但在 train / valid / test 三段诊断口径下，`SCM-v2` 在 test 均值、泛化 gap 和 Brier 上略优于 `SCM_v3_best`。

因此更稳妥的判断是：

1. `SCM_v3` 相比纯基线确实缓解了一部分过拟合。
2. `SCM_v3` 相比 `SCM-v2` 的优势还不够稳定。
3. 当前不能说 `SCM_v3` 已经完全替代 `SCM-v2`。
4. 在医疗任务中，后续必须继续报告 Brier、ECE、0类 PR 曲线，不能只看 Accuracy。

## 13. 增广审计结论

主线增广审计文件：

- `frontier_scm_v3_outputs/tables/augmentation_audit.csv`

反事实增广审计文件：

- `counterfactual_aug_outputs/tables/augmentation_audit.csv`

从实验结果和审计方向可以得到以下判断：

1. 当前最优 V3 配置平均生成 `90` 个样本，增广力度高于 `SCM-v2` 的 `72` 个样本。
2. 生成目标主要集中在 `0` 类，符合少数类边界补强目标。
3. 复杂规则与异构教师未稳定带来收益，说明过强过滤会损失可用候选样本。
4. 反事实原型生成样本较少，且指标退化，说明当前反事实样本质量不足。

## 14. 综合结论

结合全部实验，得到以下结论：

1. `SCM_v3` 是有效的结构化增广方向，相比纯 `XGBoost + ADASYN` 明显改善少数类表现。
2. 当前最佳 V3 方案为 `scm_v3_res_norule_single_flat_nocf`。
3. `SCM_v3` 在主实验口径下略优于 `SCM-v2`，但提升幅度很小。
4. 新增泛化诊断显示，`SCM-v2` 在三段划分下略优于 `SCM_v3_best`，说明 V3 优势尚未完全稳定。
5. 复杂机制组合没有赢，当前收益主要来自简单、定向、结构一致的少数类增广。
6. `TabPFN + SCM_v3` 提高总体 Accuracy，但显著损伤 0类 recall，不适合作为当前正式主线。
7. 反事实原型当前失败，不建议直接扩大。
8. `TabDDPM` 和 `GReaT` 均有正向信号，但仍弱于 SCM 主线。
9. 过拟合有一定缓解，但并未根本解决，train-test gap 仍然存在。

一句话总结：

`SCM_v3` 把结构化增广主线继续往前推了一小步，但当前阶段最重要的发现不是“更复杂机制更强”，而是“简单 residual 结构增广最稳；V3 对纯基线有效，但相对 SCM-v2 的优势还需要更强泛化验证”。

## 15. 局限性

### 15.1 数据规模限制

当前总样本只有 `374` 例，其中 `0` 类只有 `22` 例。少数类测试集中单个样本的预测变化会显著影响 recall、Balanced Accuracy 和 Macro F1。

### 15.2 外部验证不足

当前结果仍主要来自内部随机切分，没有独立外部测试集。因此不能直接宣称模型已经具备跨中心泛化能力。

### 15.3 增广策略敏感

`SCM_v3` 的复杂配置并未稳定提升，说明生成策略对小样本分布非常敏感。过度复杂的过滤和采样机制可能引入额外方差。

### 15.4 概率校准仍需加强

`SCM_v3` 的 Brier 和 ECE 没有超过 `SCM-v2`，说明概率层面仍需要专门校准。

## 16. 后续实验建议

### 16.1 固定双主线对照

后续所有新方法都应同时对照：

- `xgb_scm_v2_best`
- `scm_v3_res_norule_single_flat_nocf`

仅超过纯 `xgb_reference_adasyn` 已不足以证明新方法有效。

### 16.2 优先推进 TabDDPM

`TabDDPM` 是当前最值得继续推进的前沿生成方向。建议下一轮重点尝试：

- 类条件扩散生成
- 生成样本医学约束
- 生成样本数量网格
- `TabDDPM + SCM_v3` 混合增广
- 生成样本审计与可视化

### 16.3 重做概率校准实验

建议对 `SCM-v2` 和 `SCM_v3_best` 同时做：

- isotonic calibration
- temperature scaling
- calibration curve
- Brier score
- ECE
- 0类 PR 曲线

### 16.4 引入更稳健验证

建议使用：

- Repeated Stratified CV
- bootstrap test CI
- 留一类敏感性分析
- 少数类样本级错误分析

### 16.5 暂缓反事实路线扩大

反事实原型当前结果偏弱，应先重构生成逻辑，再决定是否重新纳入主线。

## 17. 最终建议模型

当前阶段不建议只给一个绝对最终模型，而建议采用“双主线 + 探索分支”的结论。

### 17.1 主结果展示模型

推荐展示：

- `scm_v3_res_norule_single_flat_nocf`

理由：

1. 在主实验 5 种子口径下综合表现最优。
2. `Balanced Accuracy` 和 `Macro F1` 略优于 `SCM-v2`。
3. `0类 Recall` 达到 `0.95`。
4. 方案结构简单，便于解释和复现。

### 17.2 保守对照模型

必须同时保留：

- `xgb_scm_v2_best`

理由：

1. 它在新增泛化诊断中略优于 `SCM_v3_best`。
2. 它是当前最强稳定对照。
3. 它可以防止过度解释 V3 的边际提升。

### 17.3 前沿探索模型

建议继续投入：

- `xgb_tabddpm_proto`

理由：

1. 它优于纯基线。
2. 它代表不同于 SCM 的生成范式。
3. 仍有通过条件生成和医学约束继续提升的空间。

最终建议表述为：

本阶段正式结论以 `SCM_v3_best` 作为主线候选，以 `SCM-v2` 作为强稳健对照；前沿扩展优先推进 `TabDDPM`，反事实路线暂缓扩大，`GReaT` 保留为外部语义生成探索分支。
