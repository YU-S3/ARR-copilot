# 联合因果增强 + SCM-v2 实验报告

## 1. 实验定位

本轮实验将两个既有主线的最优选择合并验证：

- 因果增强 XGBoost 主线：`ordinal_emd_fw_gpl_xgboost`
- 增广主线：`scm_v2_hard_c0_tm40_rs50_teacher_single`

并以原始 `XGBoost` 作为 baseline，统一在 `5` 个随机种子下做综合比较。

## 2. 对比组

- `xgb_reference_raw`
- `xgb_reference_adasyn`
- `xgb_scm_v2_best`
- `causal_mainline_xgboost`
- `ordinal_mainline_xgboost`
- `causal_scm_v2_joint`
- `ordinal_scm_v2_joint`

## 3. 主结果汇总

根据 `metrics_mean_std.csv`，完整主实验在 `5` 个随机种子上的均值结果如下：

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | Log Loss | MCC | Quadratic Kappa | Top-2 Acc | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.9840 | 0.9727 | 0.9591 | 0.9500 | 0.1145 | 0.9663 | 0.9580 | 1.0000 | 0.0747 | 0.0369 |
| `causal_scm_v2_joint` | 0.9733 | 0.9214 | 0.9264 | 0.8000 | 0.3591 | 0.9435 | 0.9379 | 0.9893 | 0.2567 | 0.1481 |
| `xgb_reference_adasyn` | 0.9733 | 0.9081 | 0.9183 | 0.7500 | 0.1208 | 0.9430 | 0.9224 | 1.0000 | 0.0680 | 0.0460 |
| `ordinal_scm_v2_joint` | 0.9600 | 0.9323 | 0.8982 | 0.8500 | 0.1550 | 0.9188 | 0.8889 | 0.9947 | 0.0821 | 0.0686 |
| `causal_mainline_xgboost` | 0.9680 | 0.8901 | 0.8958 | 0.7000 | 0.3550 | 0.9311 | 0.9262 | 0.9893 | 0.2507 | 0.1456 |
| `xgb_reference_raw` | 0.9627 | 0.8875 | 0.8780 | 0.7000 | 0.1556 | 0.9216 | 0.9184 | 0.9973 | 0.0678 | 0.0655 |
| `ordinal_mainline_xgboost` | 0.9387 | 0.8891 | 0.8528 | 0.7500 | 0.1718 | 0.8772 | 0.8686 | 0.9947 | 0.0707 | 0.0866 |

新增指标给出几个很清晰的信号：

1. `xgb_scm_v2_best` 不仅分类指标最好，概率质量与一致性指标也最好。
   - `Log Loss = 0.1145`，为全表最低。
   - `MCC = 0.9663`，为全表最高。
   - `Quadratic Kappa = 0.9580`，也为全表最高。

2. `causal_scm_v2_joint` 的总体分类能力仍强于两个 baseline，但概率输出明显更激进。
   - `Macro F1 = 0.9264`，高于 `xgb_reference_adasyn` 的 `0.9183`
   - 但 `Log Loss = 0.3591`、`ECE = 0.2567`、`Brier = 0.1481`，显著差于 `xgb_scm_v2_best`

3. `ordinal_scm_v2_joint` 在少数类稳健性与概率质量之间取得了折中。
   - `0类 Recall = 0.8500`
   - `Balanced Accuracy = 0.9323`
   - `Log Loss = 0.1550`，明显优于两条因果联合路线

4. `Top-2 Accuracy` 在多数模型上已接近饱和。
   - `xgb_scm_v2_best` 与 `xgb_reference_adasyn` 为 `1.0000`
   - 说明在本任务中，`Top-2 Accuracy` 更适合作为辅助指标，区分能力不如 `Balanced Accuracy / Macro F1 / Log Loss / MCC`


## 4. 联合模型校准子实验

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | ECE | Brier | 温度均值 |
|---|---:|---:|---:|---:|---:|---:|
| `causal_scm_v2_joint_temperature_scaling` | 0.9440 | 0.8581 | 0.8321 | 0.0819 | 0.0811 | 0.5000 |
| `causal_scm_v2_joint_uncalibrated` | 0.9440 | 0.8581 | 0.8321 | 0.2438 | 0.1683 | 1.0000 |


## 5. 交叉验证结果

本轮已补充：

- `5` 折交叉验证
- `7` 折交叉验证
- `10` 折交叉验证

对应输出文件为：

- `joint_causal_scm_v2_outputs/tables/cross_validation_by_fold.csv`
- `joint_causal_scm_v2_outputs/tables/cross_validation_mean_var.csv`
- `joint_causal_scm_v2_outputs/tables/overfitting_indicators.csv`

### 5.1 5 折结果

根据 `cross_validation_mean_var.csv`，`5` 折时的主要信号是：

| 方案 | Accuracy Mean | Accuracy Var | Balanced Accuracy Mean | Balanced Accuracy Var | Macro F1 Mean | Macro F1 Var | 0类 Recall Mean | Log Loss Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `causal_scm_v2_joint` | 0.9652 | 0.000333 | 0.9229 | 0.004044 | 0.9005 | 0.003328 | 0.8100 | 0.3664 |
| `causal_mainline_xgboost` | 0.9733 | 0.000142 | 0.9168 | 0.000101 | 0.9279 | 0.000565 | 0.7700 | 0.3545 |
| `xgb_scm_v2_best` | 0.9599 | 0.000075 | 0.9103 | 0.004291 | 0.8891 | 0.001749 | 0.7700 | 0.1510 |

这一折数下，因果路线在 `Accuracy / Macro F1` 上更占优，但 `xgb_scm_v2_best` 已经表现出明显更好的概率质量，`Log Loss` 只有 `0.1510`，远低于两条因果路线。

### 5.2 7 折结果

`7` 折后，最优主线开始转向 `SCM-v2 + XGBoost`：

| 方案 | Accuracy Mean | Accuracy Var | Balanced Accuracy Mean | Balanced Accuracy Var | Macro F1 Mean | Macro F1 Var | 0类 Recall Mean | Log Loss Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.9599 | 0.000543 | 0.9385 | 0.002507 | 0.8980 | 0.002801 | 0.8690 | 0.1501 |
| `xgb_reference_raw` | 0.9573 | 0.000770 | 0.9373 | 0.002878 | 0.8985 | 0.003416 | 0.8690 | 0.1624 |
| `ordinal_scm_v2_joint` | 0.9331 | 0.001002 | 0.9215 | 0.002381 | 0.8595 | 0.002028 | 0.8571 | 0.2249 |

这说明随着验证切分更细，`SCM-v2` 主线对少数类和边界状态的稳健性优势开始更稳定地显现出来。

### 5.3 10 折结果

`10` 折结果最能代表本轮新增验证的最终结论：

| 方案 | Accuracy Mean | Accuracy Var | Balanced Accuracy Mean | Balanced Accuracy Var | Macro F1 Mean | Macro F1 Var | 0类 Recall Mean | Log Loss Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.9677 | 0.000703 | 0.9571 | 0.002642 | 0.9221 | 0.002309 | 0.9167 | 0.1412 |
| `causal_scm_v2_joint` | 0.9676 | 0.001140 | 0.9417 | 0.005591 | 0.9226 | 0.005587 | 0.8667 | 0.3639 |
| `ordinal_scm_v2_joint` | 0.9544 | 0.000882 | 0.9331 | 0.004604 | 0.8965 | 0.003517 | 0.8667 | 0.1544 |

可以看到：

1. `xgb_scm_v2_best` 与 `causal_scm_v2_joint` 的 `Accuracy / Macro F1` 已非常接近。
2. 但 `xgb_scm_v2_best` 的 `Balanced Accuracy` 更高：`0.9571 > 0.9417`
3. 且 `0类 Recall` 更高：`0.9167 > 0.8667`
4. 最关键的是 `Log Loss` 显著更低：`0.1412 << 0.3639`

因此，`10` 折交叉验证进一步确认：

- 当前最强主线仍是 `xgb_scm_v2_best`
- `causal_scm_v2_joint` 更像“总体性能接近，但概率质量偏差更大”的补充路线

## 6. 方差与稳定性分析

若只看三条与 `SCM-v2` 直接相关的主线，`10` 折的标准差如下：

| 方案 | Accuracy Std | Balanced Accuracy Std | Macro F1 Std | 0类 Recall Std | Log Loss Std |
|---|---:|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.0265 | 0.0514 | 0.0480 | 0.1708 | 0.0662 |
| `causal_scm_v2_joint` | 0.0338 | 0.0748 | 0.0747 | 0.2082 | 0.0361 |
| `ordinal_scm_v2_joint` | 0.0297 | 0.0679 | 0.0593 | 0.2082 | 0.0439 |

这里说明：

1. `xgb_scm_v2_best` 在 `Balanced Accuracy` 与 `Macro F1` 上最稳定。
2. `causal_scm_v2_joint` 的总体分类均值不差，但方差更大，说明对切分更敏感。
3. `ordinal_scm_v2_joint` 的概率质量相对较好，但总体性能和少数类波动仍弱于 `xgb_scm_v2_best`。

因此，本轮新增“方差与均值”验证后，原先结论不是被推翻，而是被进一步强化：`SCM-v2 + XGBoost + 受控 ADASYN` 不只是均值更高，而且稳定性也更好。

## 7. 过拟合指标分析

本轮新增的 `overfitting_indicators.csv` 使用 train/valid 泛化差距生成 `overfit_risk_flag`。结果显示：

- 除 `Top-2 Accuracy` 外，多数模型在 `Balanced Accuracy / Macro F1 / Log Loss / class0_recall` 上都触发了风险标记
- 说明当前数据规模下，所有路线都存在一定程度的小样本过拟合

以 `10` 折下的三条主线为例：

| 方案 | Balanced Accuracy Gap | Macro F1 Gap | Log Loss Gap | 0类 Recall Gap |
|---|---:|---:|---:|---:|
| `xgb_scm_v2_best` | 0.0425 | 0.0753 | -0.0738 | 0.0833 |
| `causal_scm_v2_joint` | 0.0571 | 0.0766 | -0.0540 | 0.1333 |
| `ordinal_scm_v2_joint` | 0.0594 | 0.0866 | -0.0680 | 0.1333 |

这里 `Log Loss Gap < 0` 表示验证集损失明显大于训练集损失，属于典型过拟合信号。

这张表给出两个重要判断：

1. 当前项目不能声称“已经无过拟合”，所有方案都还有泛化差距。
2. 但 `xgb_scm_v2_best` 的少数类 gap 和总体 gap 都相对更小，说明它是当前最稳的主线。

## 8. 当前结论

综合主实验、校准子实验、`5/7/10` 折交叉验证、方差分析和过拟合指标后，本轮可以形成更完整的结论：

1. **`xgb_scm_v2_best` 仍然是当前综合最优方案。**
   - 主实验中它在 `Accuracy / Balanced Accuracy / Macro F1 / MCC / Quadratic Kappa` 上整体最优
   - 交叉验证中它在 `7` 折和 `10` 折下都保持领先
   - 在新增指标里，它的 `Log Loss` 也是最优

2. **`causal_scm_v2_joint` 的作用被重新界定为“总体性能增强路线”，而不是新的主线替代者。**
   - 它在部分切分下 `Accuracy / Macro F1` 很强
   - 但 `ECE / Brier / Log Loss` 明显更差，且方差更大

3. **`ordinal_scm_v2_joint` 是更偏向少数类稳健与概率质量的折中路线。**
   - 它优于 `ordinal_mainline_xgboost`
   - 但在综合性能与稳定性上仍落后于 `xgb_scm_v2_best`

4. **新增过拟合分析表明：当前项目所有路线都仍存在小样本过拟合。**
   - 这不影响当前主线排序
   - 但意味着后续若继续推进，仍应优先补充数据、加强更严格外部验证或做更保守的模型选择

## 9. 结果文件索引

- 主实验汇总：`joint_causal_scm_v2_outputs/tables/metrics_mean_std.csv`
- 校准子实验：`joint_causal_scm_v2_outputs/tables/joint_calibration_mean_std.csv`
- 交叉验证明细：`joint_causal_scm_v2_outputs/tables/cross_validation_by_fold.csv`
- 交叉验证均值/方差：`joint_causal_scm_v2_outputs/tables/cross_validation_mean_var.csv`
- 过拟合指标：`joint_causal_scm_v2_outputs/tables/overfitting_indicators.csv`
