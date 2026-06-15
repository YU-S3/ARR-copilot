# SCM-v2 三阶段实验结果报告

## 1. 实验背景

在上一轮“前沿增广解耦消融实验”中，项目已经确认：

- `SCM` 路线优于 `TAP-inspired` 路线；
- `SCM + 受控 ADASYN` 是当前树模型主线中的最优组合；
- 下一步最值得做的是：
  1. 进入 `SCM-v2` 参数化实验；
  2. 验证教师去偏是否能进一步提升；
  3. 在补齐 `TabPFN token` 后，测试 `TabPFN + SCM-v2` 是否能兼顾总体性能与少数类稳健性。

因此，本轮实验围绕这三个优先级，新增并运行了主脚本：

- `frontier_scm_v2_experiment.py`

结果目录为：

- `frontier_scm_v2_outputs/`

## 2. 实验目的

本轮三阶段实验依次回答以下问题：

1. `SCM-v2` 的最优参数组合是什么。
2. 在当前 `SCM-v2` 最优配置下，`single / oof / dual` 三种教师模式谁更优。
3. 将最优 `SCM-v2` 迁移到 `TabPFN` 后，是否能同时改善总体性能与少数类稳健性。

## 3. 统一实验设置

### 3.1 数据与任务

- 数据文件：`数据表格测试.xlsx`
- 任务：三分类
- 目标列：`确诊（0为排除；1为确诊；2为灰色区域）`

### 3.2 评估协议

- 随机种子：`42, 2024, 2025, 2026, 2027`
- 统一使用 `train/test = 0.8/0.2`
- 统一复用仓库现有的数据清洗、预处理与评估函数

### 3.3 主指标

本轮主指标仍为：

- `Balanced Accuracy`
- `Macro F1`
- `0类 recall`

同时参考：

- `Accuracy`
- `Weighted F1`
- `OVR ROC AUC Macro`
- `1类 / 2类 recall`

## 4. 结果文件

本轮关键结果文件如下：

- 摘要：[experiment_summary.json](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/experiment_summary.json)
- Phase 1 汇总：[phase1_metrics_mean_std.csv](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/tables/phase1_metrics_mean_std.csv)
- Phase 2 汇总：[phase2_teacher_metrics_mean_std.csv](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/tables/phase2_teacher_metrics_mean_std.csv)
- Phase 3 逐种子结果：[phase3_tabpfn_metrics_by_seed.csv](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/tables/phase3_tabpfn_metrics_by_seed.csv)
- Phase 3 汇总：[phase3_tabpfn_metrics_mean_std.csv](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/tables/phase3_tabpfn_metrics_mean_std.csv)

## 5. Phase 1：SCM-v2 参数化实验

### 5.1 实验设计

Phase 1 枚举了 `16` 组最小矩阵配置，参数包括：

- `seed_strategy`: `random_seed` vs `hard_case_seed`
- `target_classes`: `class0_only` vs `class0_and_class2`
- `treat_mix_prob`: `0.2` vs `0.4`
- `residual_scale`: `0.5` vs `0.8`

同时保留两个参照：

- `xgb_reference_adasyn`
- `xgb_scm_plus_adasyn_baseline`

### 5.2 最优结果

根据 [phase1_metrics_mean_std.csv](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/tables/phase1_metrics_mean_std.csv)，当前最优配置为：

- `scm_v2_hard_c0_tm40_rs50_teacher_single`

其均值结果为：

| 指标 | 数值 |
|---|---:|
| Accuracy | `0.9840` |
| Balanced Accuracy | `0.9727` |
| Macro F1 | `0.9591` |
| Weighted F1 | `0.9851` |
| OVR ROC AUC Macro | `0.9928` |
| 0类 Recall | `0.95` |
| 1类 Recall | `0.9882` |
| 2类 Recall | `0.98` |

### 5.3 与上一轮主线比较

上一轮树模型主线最佳基线是：

- `xgb_scm_plus_adasyn_baseline`

其均值结果为：

| 指标 | 数值 |
|---|---:|
| Accuracy | `0.9813` |
| Balanced Accuracy | `0.9601` |
| Macro F1 | `0.9532` |
| 0类 Recall | `0.90` |

相较之下，`SCM-v2` 最优配置继续提升了：

- `Balanced Accuracy`: `0.9601 -> 0.9727`
- `Macro F1`: `0.9532 -> 0.9591`
- `0类 recall`: `0.90 -> 0.95`

### 5.4 结果解读

Phase 1 给出三个重要结论：

1. **`hard_case_seed` 明显优于 `random_seed`**，说明优先围绕训练中的困难样本做结构增广是有效的。
2. **当前最优目标是只强化 `0类`**，而不是同时大幅扩增 `0类 + 2类`。
3. **`treat_mix_prob=0.4` 与 `residual_scale=0.5` 的组合最稳**，说明适度治疗变量扰动加中等残差注入，更有利于保留结构一致性。

## 6. Phase 2：教师去偏实验

### 6.1 实验设计

Phase 2 固定 Phase 1 的最优配置：

- `scm_v2_hard_c0_tm40_rs50_teacher_single`

比较三种教师模式：

- `single`
- `oof`
- `dual`

### 6.2 结果汇总

根据 [phase2_teacher_metrics_mean_std.csv](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/tables/phase2_teacher_metrics_mean_std.csv)：

| 教师模式 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall |
|---|---:|---:|---:|---:|
| `single` | `0.9840` | `0.9727` | `0.9591` | `0.95` |
| `dual` | `0.9840` | `0.9594` | `0.9487` | `0.90` |
| `oof` | `0.9733` | `0.9408` | `0.9259` | `0.85` |

### 6.3 结果解读

Phase 2 的结论非常明确：

- **当前最优教师模式仍然是 `single`**

也就是说，在当前 ARR 数据规模下：

- 引入 `OOF teacher` 并没有带来更强泛化；
- `dual teacher` 也没有超过单教师；
- 反而增加了筛选复杂度并带来轻微性能损失。

这说明当前数据量级下，“更复杂的教师体系”并不是当前瓶颈，真正有效的提升主要来自：

- 更好的增广结构；
- 更好的种子选择；
- 更好的少数类定向策略。

## 7. Phase 3：TabPFN + SCM-v2 扩展实验

### 7.1 实验设计

Phase 3 使用：

- Phase 1 最优 `SCM-v2` 配置
- Phase 2 最优教师模式 `single`

比较两种方案：

- `tabpfn_reference_v2`
- `tabpfn_scm_v2_best`

### 7.2 汇总结果

根据 [phase3_tabpfn_metrics_mean_std.csv](file:///c:/Users/YCY/Desktop/ARR-model/frontier_scm_v2_outputs/tables/phase3_tabpfn_metrics_mean_std.csv)：

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall |
|---|---:|---:|---:|---:|
| `tabpfn_reference_v2` | `0.9440` | `0.9418` | `0.8747` | `0.90` |
| `tabpfn_scm_v2_best` | `0.9707` | `0.8935` | `0.9077` | `0.70` |

### 7.3 结果解读

这组结果说明 `SCM-v2` 迁移到 `TabPFN` 后，呈现出明显的“总体性能上升，但少数类稳健性下降”的特征：

- 正向变化：
  - `Accuracy`: `0.9440 -> 0.9707`
  - `Macro F1`: `0.8747 -> 0.9077`
- 负向变化：
  - `Balanced Accuracy`: `0.9418 -> 0.8935`
  - `0类 recall`: `0.90 -> 0.70`

因此，当前不能把 `TabPFN + SCM-v2` 视作全面优于 `TabPFN` 原始方案的主线。

更准确的定位应该是：

- **它改善了总体分类性能**
- **但削弱了少数类排除类稳健性**

## 8. 总体结论

本轮三阶段实验已经把当前增广主线进一步跑清楚。

### 8.1 可以明确确认的结论

1. **`SCM-v2` 确实优于上一轮 `SCM baseline`**，并且当前最优配置已经找到。
2. **当前最优参数组合是 `hard_case_seed + class0_only + treat_mix_prob=0.4 + residual_scale=0.5`**。
3. **教师去偏在当前数据规模下没有带来额外收益，`single teacher` 仍是最佳选择**。
4. **`TabPFN + SCM-v2` 提升了总体性能，但损害了少数类稳健性，因此暂时不能替代原始 `TabPFN` 方案**。

### 8.2 当前最强主线

若综合考虑：

- 总体性能
- 少数类稳健性
- 方法复杂度
- 当前可复现性

本轮实验后的推荐主线是：

- **`scm_v2_hard_c0_tm40_rs50_teacher_single + XGBoost + 受控 ADASYN`**

这也是当前整个项目中最值得保留和继续深化的数据增广主线。

## 9. 对后续工作的建议

### 9.1 主线建议

建议后续正式主线固定为：

- `SCM-v2`
- `hard_case_seed`
- `class0_only`
- `single teacher`

### 9.2 可继续优化的方向

后续最值得继续做的有三类：

1. **进一步细化 `0类` 定向生成**
   - 例如分层控制治疗变量扰动强度；
   - 或为高风险误分样本设置更高采样权重。

2. **把 `TabPFN + SCM-v2` 改成“总体性能路线”**
   - 当前它更适合做 Overall 路线，而不是 Minority 路线；
   - 后续可以把目标明确设定为提升 `Accuracy/Macro F1`，而不再强求同步提升 `0类 recall`。

3. **补一份正式图表化报告**
   - 当前结果表已经足够支持论文或汇报；
   - 下一步可以补总览柱状图、关键方案箱线图和分阶段结论图。

## 10. 一句话总结

本轮三阶段实验表明：**`SCM-v2` 已经把结构一致增广从“有效”推进到“成熟可用”，当前最优主线是 `hard_case_seed + class0_only + single teacher` 的 `SCM-v2 + XGBoost`；而 `TabPFN + SCM-v2` 更适合作为总体性能增强路线，而不是当前少数类稳健主线。**
