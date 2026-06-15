# 因果增强 XGBoost 变体实验报告

## 1. 实验目的

在当前 `ARR` 临床真实世界表格数据上，验证以下问题：

1. 在保持当前 `XGBoost` 高分类性能的前提下，是否能够通过因果推断、生理机制先验、灰区软标签和非对称临床代价，引入更强的方法学创新。
2. 在以 `Ordinal XGBoost` 为主线的前提下，哪一种增强方案最适合作为下一版论文或课题的主方案。
3. 当“灰色区域（2）”被视为介于 `0` 和 `1` 之间的暧昧概率区间时，基于二阈值/有序结构建模是否比普通三分类更有优势。

本次实验在同一数据切分、同一测试集上，与原始 `XGBoost` 做公平对照。

## 2. 数据与实验口径

### 2.1 数据来源

- 数据文件：`c:\Users\YCY\Desktop\ARR-model\数据表格测试.xlsx`
- 清洗脚本基础：`c:\Users\YCY\Desktop\ARR-model\multiclass_ensemble_experiment.py`

### 2.2 目标变量

- `确诊（0为排除；1为确诊；2为灰色区域）`

本次仍按三分类任务建模，但所有有序模型都使用临床顺序：

- `0 = 排除`
- `2 = 灰区`
- `1 = 确诊`

即内部严重度 rank 为：$0 < 2 < 1$。

### 2.3 数据规模

根据统一清洗口径：

- 总样本数：`374`
- 训练集：`299`
- 测试集：`75`

### 2.4 统一实验原则

为保证公平比较，全部方案统一使用：

- 相同训练/测试划分
- 相同基础预处理逻辑
- 相同 `XGBoost` 参数骨架
- 相同三分类评价指标
- 相同受控 `ADASYN`
- 相同 `balanced sample_weight`

新增比较维度：

- `ECE`
- `Brier Score`
- 连续严重程度评分单调性

## 3. 对比方案设计与详细技术解析

本轮共比较 `1` 个原始基线 + `10` 个创新变体。为便于后续论文撰写，这里不仅给出方案名称，还把每个方案的技术目标、输入输出、算法结构、损失函数设计、与当前代码实现的映射关系一并写清楚。

统一记号约定：

- 给定样本 `i` 的输入特征为 $x_i \in \mathbb{R}^d$，标签为 `y_i`；
- 原始三分类标签集合记为 $\mathcal{Y} = \{0, 2, 1\}$，其临床顺序定义为 $0 < 2 < 1$；
- 为了便于有序建模，引入内部 rank 编码 $r_i \in \{0, 1, 2\}$，满足：
  - $r_i = 0$ 对应原标签 `0`
  - $r_i = 1$ 对应原标签 `2`
  - $r_i = 2$ 对应原标签 `1`
- 若模型输出 logits $z_i = [z_{i0}, z_{i1}, z_{i2}]$，则 softmax 概率定义为：
- $p_{ik} = \frac{\exp(z_{ik})}{\sum_{m=0}^{2} \exp(z_{im})}$
- 对 one-hot 目标记为 `e(r_i)`，对软标签目标记为 $q_i = [q_{i0}, q_{i1}, q_{i2}]$。

### 3.1 基线：原始 XGBoost

技术目标：

- 构建一个尽可能强、但不引入额外结构先验的三分类参照模型；
- 用来判断后续所有创新设计到底是在“真正提升建模能力”，还是只是换了一种表述方式。

输入与输出：

- 输入：除目标标签外的全部预处理后特征；
- 输出：标准三分类概率 $P(y=0), P(y=1), P(y=2)$。

算法结构：

- 使用 `objective="multi:softprob"` 的标准 `XGBoost` 多分类器；
- 训练时采用项目内已经验证过的一组固定超参数；
- 在训练集上配合 `balanced sample_weight` 与受控 `ADASYN`，以减少极端类别不均衡导致的树分裂偏置。

为什么它重要：

- 这个模型不显式理解“0、2、1”之间的顺序关系；
- 它把三类看作平行类别，因此是检验“有序建模是否真的有价值”的最直接对照。

当前代码实现：

- 主体对应 `make_xgb_classifier()` 与 `fit_baseline_xgb()`；
- 该实现保留了最传统的树模型训练路径，没有自定义损失，也没有额外特征生成。

公式化表达：

- 该模型直接学习：
- $f_\theta: x_i \mapsto p_i = [p_{i0}, p_{i1}, p_{i2}]$
- 其标准多分类交叉熵损失为：
- $\mathcal{L}_{\text{CE}} = - \sum_{i=1}^{n} \sum_{k \in \{0,1,2\}} \mathbf{1}(y_i = k) \log p_{ik}$
- 若考虑样本权重 `w_i`，则训练目标为：
- $\mathcal{L}_{\text{baseline}} = - \sum_{i=1}^{n} w_i \sum_{k \in \{0,1,2\}} \mathbf{1}(y_i = k) \log p_{ik}$
- 这里的 `w_i` 由 `balanced sample_weight` 提供，本质上是对类别不均衡的频率反比修正。

### 3.2 DML-XGBoost

技术动机：

- 当前数据中的药物等效分数并不是纯预测特征，而更像一种“多重干预”记录；
- 如果直接把药物变量喂给分类器，模型容易利用药物与确诊结果之间的相关性，但这些相关性中混入了医生处方偏好和患者隐性严重程度的共同影响。

核心思想：

- 先估计“药物由基础状态决定的部分”，再把剩余的、相对更接近个体特异性的残差送入最终分类器；
- 用双重机器学习思想，把“解释治疗分配”和“解释结局风险”的任务拆开。

具体技术流程：

1. 从原始特征中剔除药物列、部分确诊后变量与 `ARR比值`，构造协变量集 `W`；
2. 对每一列药物等效分数 `T_j` 做交叉拟合回归，得到 out-of-fold 预测 $\hat{T}_j(W)$；
3. 计算治疗残差 $R_j = T_j - \hat{T}_j(W)$；
4. 对 `ARR比值` 同样做基线回归，得到 $\hat{ARR}(W)$；
5. 再用治疗残差去解释 `ARR` 的剩余部分，形成一个近似“药物剥离后的校正 ARR”；
6. 将 `DML_校正ARR + DML_残差_药物列` 输入最终 `XGBoost` 三分类器。

数学上可以理解为：

- 第一阶段：$T = g(W) + \epsilon_T$
- 第二阶段：$ARR = m(W) + h(\epsilon_T) + \epsilon_Y$
- 最终分类阶段：`Y = f(W, corrected_ARR, residual_treatments)`

工程实现细节：

- 使用 `crossfit_residuals()` 做 KFold 交叉拟合，避免在同一样本上同时“拟合残差又消费残差”，从而减轻泄漏；
- 一阶段与二阶段都使用轻量 `XGBRegressor`，确保去交杂过程仍能拟合非线性关系；
- 最终分类仍使用项目统一的 `XGBoost` 多分类骨架，保证比较公平。

当前代码实现：

- 关键函数包括 `build_dml_covariates()`、`crossfit_residuals()`、`build_dml_features()`；
- 最终分类部分仍复用 `fit_baseline_xgb()`。

公式化表达：

- 设协变量为 `W_i`，药物暴露向量为 $T_i = [T_{i1}, \dots, T_{ip}]$，原始 `ARR` 记为 `A_i`；
- 对每个药物分量做一阶段拟合：
- $\hat{T}_{ij} = g_j(W_i)$
- 治疗残差定义为：
- $R_{ij} = T_{ij} - \hat{T}_{ij}$
- 对 `ARR` 基线项拟合：
- $\hat{A}^{(0)}_i = m(W_i)$
- 令 $\tilde{A}_i = A_i - \hat{A}^{(0)}_i$ 表示剥离基础协变量后的剩余 ARR；
- 再拟合治疗残差对剩余 ARR 的影响：
- $\hat{u}_i = h(R_i)$
- 定义校正 ARR 为：
- $A_i^{\text{corr}} = A_i - \hat{u}_i$
- 最终分类器学习：
- $p_i = f_\theta(W_i, A_i^{\text{corr}}, R_i)$
- 对最终分类器仍使用多分类交叉熵：
- $\mathcal{L}_{\text{DML-XGB}} = - \sum_{i=1}^{n} w_i \sum_{k} \mathbf{1}(y_i = k) \log p_{ik}$
- 因而 DML 的本质不是替换分类损失，而是通过特征重构改变 $f_\theta$ 的输入统计结构。

方法价值与局限：

- 价值在于把“预测效果”和“去交杂解释”部分耦合起来，适合作为论文中的因果增强模块；
- 局限在于小样本下残差估计会放大噪声，因此单独作为最终分类器时，往往不如直接判别模型稳定。

### 3.3 Adaptive-CaRe inspired XGBoost

技术动机：

- 有些特征虽然对分类有帮助，但可能是“相关而非因果”的捷径变量；
- 如果树模型过于偏爱这些捷径，性能可能在当前测试集上不错，但医学解释性和外部稳定性不足。

核心思想：

- 不改 `XGBoost` 的底层分裂准则，而是通过特征权重和列采样策略，软性鼓励模型更多使用“更接近生理父节点”的特征。

具体做法：

- 将特征划分为三类：
  - 生理核心变量：如 `ARR比值`、醛固酮、肾素、电解质、血压、结节信息；
  - 药物暴露变量：各类等效分数；
  - 确诊后变量与缺失指示变量；
- 对生理核心变量给更高 `feature_weight`；
- 对药物列和部分后处理变量给更低 `feature_weight`；
- 同时下调 `colsample_bytree` 与 `colsample_bylevel`，让每棵树在受限候选集中完成分裂，以抑制对单一伪相关特征的过拟合。

这不是严格的因果正则化，但它实现了一个工程上可操作的近似：

- “更重要的因果父节点更容易被抽到并参与分裂”；
- “药物类捷径变量虽然不被禁止，但要付出更高的使用机会成本”。

当前代码实现：

- `build_adaptive_feature_weights()` 负责根据原始列语义生成权重；
- `fit_weighted_xgb_booster()` 把这些权重送入 `xgb.DMatrix(feature_weights=...)`。

公式化表达：

- 设原始特征为 $x_i = [x_{i1}, \dots, x_{id}]$，为每一维指定特征权重 $\alpha_j > 0$；
- 可将该方法理解为在树分裂候选搜索时，对特征采样概率施加偏置：
- $\Pr(\text{feature}=j) \propto \alpha_j$
- 其中：
- $\alpha_j$ 较大：生理核心变量；
- $\alpha_j$ 较小：药物暴露变量与部分后处理变量。
- 最终分类目标仍是标准多分类交叉熵：
- $\mathcal{L}_{\text{Adaptive-CaRe}} = - \sum_{i=1}^{n} w_i \sum_{k} \mathbf{1}(y_i = k) \log p_{ik}$
- 但参数更新的搜索路径已被 $\alpha_j$ 所诱导，因此它是“结构化搜索偏置”而非显式惩罚项优化。

方法价值与局限：

- 价值在于不需要改损失函数，工程代价小，故事性强；
- 局限在于它只是在采样和分裂优先级上“偏置模型”，并没有真正建立因果图或约束反事实一致性，因此提升通常有限。

### 3.4 Ordinal XGBoost

技术动机：

- 当前标签不是普通三分类，而是 `0=排除, 2=灰区, 1=确诊` 的有序状态；
- 普通三分类把 `0` 与 `2` 的关系、`2` 与 `1` 的关系、`0` 与 `1` 的关系全部当成同质错误，这与临床语义不符。

核心思想：

- 不直接学习三个互斥类别，而是学习“是否跨越某个严重度阈值”的累计概率；
- 这类做法等价于将问题重写为两个二分类子任务。

具体概率结构：

- 设内部 rank 顺序为 $0 < 1 < 2$，分别对应原标签 $0 < 2 < 1$；
- 训练两个分类器：
  - $P(y > 0)$：是否进入灰区或确诊；
  - $P(y > 1)$：是否进入确诊；
- 再还原类别概率：
  - $P(0) = 1 - P(y > 0)$
  - $P(2) = P(y > 0) - P(y > 1)$
  - $P(1) = P(y > 1)$

为什么这种结构合理：

- 它显式要求模型先学会“排除 vs 非排除”，再学会“灰区 vs 确诊”；
- 这比直接三分类更接近临床诊断路径，因为临床判断本来就常常先排除明显正常，再讨论是否达到确诊阈值。

工程细节：

- 为避免出现 $P(y > 1) > P(y > 0)$ 的概率违背，预测时强制做 `p_gt1 = min(p_gt1, p_gt0)`；
- 这样可以维持累计链接模型应有的单调性。

当前代码实现：

- `build_ordinal_binary_targets()` 生成两个阈值标签；
- `fit_ordinal_xgboost()` 训练两棵二分类 `XGBClassifier` 并重构三分类概率。

公式化表达：

- 设内部 rank 为 $r_i \in \{0,1,2\}$；
- 构造两个累计阈值标签：
- $t_i^{(0)} = \mathbf{1}(r_i > 0)$
- $t_i^{(1)} = \mathbf{1}(r_i > 1)$
- 分别训练两个二分类器：
- $\pi_i^{(0)} = P(r_i > 0 \mid x_i)$
- $\pi_i^{(1)} = P(r_i > 1 \mid x_i)$
- 由累计概率恢复 rank 概率：
- $P(r_i = 0) = 1 - \pi_i^{(0)}$
- $P(r_i = 1) = \pi_i^{(0)} - \pi_i^{(1)}$
- $P(r_i = 2) = \pi_i^{(1)}$
- 再映射回原标签概率：
- $P(y_i = 0) = P(r_i = 0)$
- $P(y_i = 2) = P(r_i = 1)$
- $P(y_i = 1) = P(r_i = 2)$
- 两个阈值子任务的损失分别为二元交叉熵：
- $\mathcal{L}_{\text{ord-0}} = - \sum_i w_i^{(0)} [ t_i^{(0)} \log \pi_i^{(0)} + (1-t_i^{(0)}) \log (1-\pi_i^{(0)}) ]$
- $\mathcal{L}_{\text{ord-1}} = - \sum_i w_i^{(1)} [ t_i^{(1)} \log \pi_i^{(1)} + (1-t_i^{(1)}) \log (1-\pi_i^{(1)}) ]$
- 总体目标可写为：
- $\mathcal{L}_{\text{Ordinal}} = \mathcal{L}_{\text{ord-0}} + \mathcal{L}_{\text{ord-1}}$

方法价值与局限：

- 价值在于结构简单、临床解释直观、在小样本下通常比直接三分类更稳；
- 局限在于它仍使用普通 logloss，没有把“远距误判比近邻误判更严重”的临床代价真正纳入训练。

### 3.5 Physiology-Informed XGBoost

技术动机：

- 医生并不只看孤立的原始变量，而是会在脑中完成一些药理和生理交互推断；
- 例如 Beta 受体阻滞剂、利尿剂、RASS 相关用药会改变肾素和 ARR 的解释方式。

核心思想：

- 不去改树模型，而是在输入层显式构造“药理生理学交叉特征”；
- 把医生通常会做的隐式校正，转成模型可见的显式变量。

构造的主要特征包括：

- 药物负荷类：
  - `药理_总负荷`
  - `药理_净抑制分数`
- 交互项类：
  - `药理_ARR_Beta交互`
  - `药理_ARR_RASS交互`
  - `药理_醛固酮_RASS交互`
- 比值校正类：
  - `药理_ARR_利尿剂校正`
  - `药理_ARR_CCB校正`
  - `药理_肾素_Beta校正`
  - `药理_肾素_利尿剂刺激`
- 综合校正式：
  - `药理_综合校正ARR`

本质上，这一方案试图让模型学习的对象从“原始化验值”升级为“带药理语义的二阶特征空间”。

当前代码实现：

- 全部特征构造在 `add_physiology_features()` 中完成；
- 随后仍进入标准 `fit_baseline_xgb()` 流程。

公式化表达：

- 设原始关键生理变量为：
- $A_i = ARR_i, R_i = Renin_i, L_i = Aldosterone_i$
- 药物暴露记为：
- $B_i = Beta_i, D_i = Diuretic_i, S_i = RASS_i, C_i = CCB_i$
- 则构造一组机制增强特征 $\phi(x_i)$，例如：
- $\phi_1(x_i) = A_i \cdot B_i$
- $\phi_2(x_i) = A_i \cdot S_i$
- $\phi_3(x_i) = \frac{A_i}{1 + D_i}$
- $\phi_4(x_i) = \frac{R_i}{1 + B_i}$
- $\phi_5(x_i) = L_i \cdot (1 + S_i)$
- 最终输入从 `x_i` 扩展为：
- $x_i^{\text{phys}} = [x_i, \phi(x_i)]$
- 最终优化目标仍然是：
- $\mathcal{L}_{\text{phys}} = - \sum_{i=1}^{n} w_i \sum_{k} \mathbf{1}(y_i = k) \log p_{ik}$
- 因而该方法本质上是显式特征映射 $x_i \mapsto x_i^{\text{phys}}$，而不是损失函数改造。

方法价值与局限：

- 价值在于解释性强，适合对医生说明“模型为什么认为该患者风险升高”；
- 局限在于显式特征工程受先验公式质量限制，很难覆盖全部复杂非线性关系，因此常常更像解释增强而非纯性能增强。

### 3.6 Ordinal XGBoost V2

技术动机：

- `Ordinal XGBoost` 已经引入顺序结构，但仍把各类错误当作同质；
- 为了验证“临床代价”是否值得更深入建模，需要先做一个低成本近似版。

核心思想：

- 保留双阈值框架不变；
- 在训练阶段对两个阈值任务使用不同的样本权重；
- 在预测阶段再用代价矩阵对输出概率进行风险重标定。

训练端细节：

- 对 `gt0` 和 `gt1` 两个阈值任务分别定义不同 multiplier；
- 例如对于 `gt1` 任务，会更强调“确诊 vs 非确诊”的分离，同时把灰区看作邻近而非完全等同于排除类。

预测端细节：

- 给定原始概率向量 `p` 与代价矩阵 `C`；
- 计算每个预测类别的期望代价 `p @ C`；
- 对负代价做 softmax，得到风险导向的替代分布；
- 再和原始概率按比例混合，得到最终概率。

这条路线的本质是：

- 训练端用样本权重近似类别代价；
- 推理端用贝叶斯决策式的 expected cost 近似临床风险最小化；
- 但代价仍然主要在“输出层”而不是“梯度层”发挥作用。

当前代码实现：

- `compute_cost_sensitive_weights()` 负责阈值级权重；
- `apply_ordinal_cost_matrix()` 负责输出概率的风险重标定；
- 主过程由 `fit_ordinal_xgboost_v2()` 完成。

公式化表达：

- 在 `Ordinal XGBoost` 的两个阈值任务上，分别引入与原始标签有关的样本权重修正：
- $w_i^{(0)} = \bar{w}_i \cdot \gamma^{(0)}(y_i)$
- $w_i^{(1)} = \bar{w}_i \cdot \gamma^{(1)}(y_i)$
- 其中 $\bar{w}_i$ 是基础类别平衡权重，$\gamma^{(0)}, \gamma^{(1)}$ 是阈值特异性的临床代价系数。
- 训练损失变为：
- $\mathcal{L}_{\text{Ordinal-V2}} = \mathcal{L}_{\text{ord-0}}^{\text{cost}} + \mathcal{L}_{\text{ord-1}}^{\text{cost}}$
- 其中：
- $\mathcal{L}_{\text{ord-b}}^{\text{cost}} = - \sum_i w_i^{(b)} [ t_i^{(b)} \log \pi_i^{(b)} + (1-t_i^{(b)}) \log (1-\pi_i^{(b)}) ]$
- 预测阶段给定原始三分类概率 `p_i` 和代价矩阵 $C \in \mathbb{R}^{3 \times 3}$；
- 第 `c` 个决策类别的期望风险定义为：
- $R_i(c) = \sum_{k=0}^{2} p_{ik} C_{k,c}$
- 用温度参数 $\tau$ 把负风险转成风险偏好分布：
- $\tilde{p}_{ic} = \frac{\exp(-R_i(c) / \tau)}{\sum_{m=0}^{2} \exp(-R_i(m) / \tau)}$
- 最终混合分布为：
- $p_i^{\star} = (1-\lambda) p_i + \lambda \tilde{p}_i$
- 其中 $\lambda$ 是风险重标定混合系数。

方法价值与局限：

- 价值在于提供了“从普通 ordinal 到真正代价敏感 ordinal”的过渡版本；
- 局限在于它仍不是底层损失级别的代价建模，因此性能提升有限。

### 3.7 DML + Ordinal 联合模型

技术动机：

- `DML-XGBoost` 单独作为最终分类器时偏弱；
- 但其产生的 `DML_校正ARR` 可能恰好适合放入“阈值式有序判别”框架。

核心思想：

- 先通过 DML 生成更接近去交杂后的关键特征；
- 再通过 `Ordinal XGBoost V2` 去完成结构化分类。

为什么这条路线有意义：

- DML 负责“把药物带来的表观偏移剥开”；
- Ordinal 负责“按临床顺序完成最终判别”；
- 两者在职责上是互补而不是重复的。

当前代码实现：

- 先调用 `build_dml_features()` 形成扩展特征；
- 再把结果送入 `fit_ordinal_xgboost_v2()`。

公式化表达：

- 令 DML 扩展特征记为：
- $x_i^{\text{dml}} = [W_i, A_i^{\text{corr}}, R_i]$
- 联合模型学习两个累计概率：
- $\pi_i^{(0)} = P(r_i > 0 \mid x_i^{\text{dml}})$
- $\pi_i^{(1)} = P(r_i > 1 \mid x_i^{\text{dml}})$
- 然后继续采用 `Ordinal XGBoost V2` 的代价敏感目标：
- $\mathcal{L}_{\text{DML-Ordinal}} = \mathcal{L}_{\text{ord-0}}^{\text{cost}}(x_i^{\text{dml}}) + \mathcal{L}_{\text{ord-1}}^{\text{cost}}(x_i^{\text{dml}})$
- 因而该方法可以理解为：
- $x_i \xrightarrow{\text{DML}} x_i^{\text{dml}} \xrightarrow{\text{Ordinal-Cost}} p_i$

方法价值与局限：

- 价值在于它是当前最像“因果推断 + 有序医学诊断”联合论文故事的一条路线；
- 局限在于两层建模叠加后，误差传播更强，对样本量要求也更高。

### 3.8 Ordinal EMD XGBoost

技术动机：

- `Ordinal XGBoost V2` 只是在输出侧近似代价，而没有把“远距误判更严重”的信息推进到训练目标；
- 真正更有理论深度的做法，是把类别分布之间的运输成本直接写进损失函数。

核心思想：

- 将预测概率分布与目标分布都看作定义在有序链上的质量分布；
- 使用 Earth Mover's Distance 的链式松弛形式，度量“把预测分布搬运到真实分布”所需的累计代价。

为何选链状松弛 EMD：

- 真实的最优传输求解在树模型自定义损失里会过重；
- 对于本任务，输出空间本身就是 $0 < 2 < 1$ 的链式有序结构；
- 因而可以用累积分布函数差值的形式，构造一个可微、可闭式求梯度的 surrogate。

具体实现思路：

1. 先将原标签映射到 rank 空间 `[0, 1, 2]`；
2. 对模型 logits 做 softmax 得到 `p`；
3. 计算前两个边界上的累计分布 $CDF(p)$；
4. 将预测 CDF 与真实 CDF 的差值乘以边界权重；
5. 再通过 softmax Jacobian 链式回传到每个 logit；
6. 使用对角 Hessian 上界保证 `xgb.train` 可稳定优化。

边界权重的临床含义：

- 如果真实类别是确诊，则把质量错运到排除侧的代价最大；
- 因此“确诊被错到排除”会在对应边界累计差上获得更大惩罚；
- 这比简单的类别权重更贴近临床漏诊成本。

当前代码实现：

- `clinical_boundary_weights_from_soft_targets()` 负责根据目标分布生成边界权重；
- `make_relaxed_emd_objective()` 负责输出自定义梯度与 Hessian；
- `fit_custom_softprob_booster(objective_mode="emd")` 负责训练。

公式化表达：

- 对于 rank 概率 $p_i = [p_{i0}, p_{i1}, p_{i2}]$ 和目标分布 `q_i`，定义前缀累计分布：
- $F_i^{p}(j) = \sum_{m=0}^{j} p_{im}, \quad j \in \{0,1\}$
- $F_i^{q}(j) = \sum_{m=0}^{j} q_{im}, \quad j \in \{0,1\}$
- 设两个边界上的权重为 $\omega_i = [\omega_{i0}, \omega_{i1}]$；
- 则链式 Relaxed EMD surrogate 可写为：
- $\mathcal{L}_{\text{EMD},i} = \sum_{j=0}^{1} \omega_{ij} \, \Psi(F_i^{p}(j) - F_i^{q}(j))$
- 其中 $\Psi(\cdot)$ 代表当前实现中隐式对应于累计差的平滑一阶近似；
- 若以平方形式表述，可写成更直观的近似版本：
- $\mathcal{L}_{\text{EMD},i}^{\text{sq}} \approx \frac{1}{2} \sum_{j=0}^{1} \omega_{ij} (F_i^{p}(j) - F_i^{q}(j))^2$
- 当前实现并非直接最小化严格平方式，而是利用累计差推导 logits 上的近似梯度；
- 联合 soft CE 稳定项后，单样本目标写为：
- $\mathcal{L}_i = \lambda_{\text{emd}} \mathcal{L}_{\text{EMD},i} + \lambda_{\text{ce}} \mathcal{L}_{\text{CE-soft},i}$
- 其中：
- $\mathcal{L}_{\text{CE-soft},i} = - \sum_{k=0}^{2} q_{ik} \log p_{ik}$
- 当 $q_i = e(r_i)$ 时，就退化为针对 one-hot 目标的 EMD + CE 训练。

方法价值与局限：

- 价值在于首次把“非对称临床代价 + 有序最优传输”推进到训练目标内部；
- 局限在于当前仍是 relax 版 surrogate，而非严格 OT 解，因此更适合工程实用和论文方法学近似，而不是数学上最精确的 Wasserstein 优化。

### 3.9 Ordinal FW-GPL XGBoost

技术动机：

- 灰区标签本身就不是一个绝对确定的状态；
- 如果强迫模型把灰区样本拟合成尖锐 one-hot 标签，模型会对本就边界模糊的样本产生过强信心。

核心思想：

- 把硬标签变成软标签分布；
- 尤其让灰区标签同时向左右相邻类别保留一定概率质量。

当前使用的标签模板：

- `0 -> [0.85, 0.15, 0.00]`
- `2 -> [0.15, 0.70, 0.15]`
- `1 -> [0.00, 0.15, 0.85]`

这组模板的含义是：

- `0` 与 `1` 仍然是相对尖锐的主标签，但允许少量质量向灰区泄漏；
- `2` 的目标分布最宽，因为灰区本身就是相邻状态的混合区。

损失函数设计：

- 不再使用普通 one-hot cross entropy；
- 而是使用 soft-label cross entropy，使模型学习“逼近目标概率分布”，而不是“命中唯一正确类别”。

这样做的直接效果通常包括：

- 降低灰区样本的过度自信；
- 让相邻类别之间的决策边界变得更平滑；
- 提高模型面对重叠特征区域时的鲁棒性。

当前代码实现：

- `FW_GPL_SOFT_LABELS` 定义软标签模板；
- `build_soft_labels_fw_gpl()` 负责按标签生成目标分布；
- `make_soft_ce_objective()` 负责 soft CE 的梯度与 Hessian；
- `fit_custom_softprob_booster(objective_mode="soft_ce", use_fw_gpl=True)` 负责训练。

公式化表达：

- 为每个原始类别定义软标签分布：
- `q(y=0) = [0.85, 0.15, 0.00]`
- `q(y=2) = [0.15, 0.70, 0.15]`
- `q(y=1) = [0.00, 0.15, 0.85]`
- 因而对任一样本 `i`，其目标分布写为：
- $q_i = q(y_i)$
- 训练损失为 soft-label cross entropy：
- $\mathcal{L}_{\text{FW-GPL}} = - \sum_{i=1}^{n} w_i \sum_{k=0}^{2} q_{ik} \log p_{ik}$
- 如果从标签平滑角度理解，这等价于把 one-hot 目标 `e(r_i)` 替换为带邻域泄漏的分布 `q_i`；
- 其梯度对 logits 的一阶形式为：
- $\frac{\partial \mathcal{L}_{\text{FW-GPL}}}{\partial z_{ik}} = w_i (p_{ik} - q_{ik})$
- 因而它与普通交叉熵的差异不在梯度形式，而在目标分布 `q_i` 的定义。

方法价值与局限：

- 价值在于它直接抓住了“灰区不确定性”这个本项目最有临床意义的问题；
- 局限在于软标签模板目前是固定的，并未根据样本本身的可信度、检查类型或实验边界做自适应调整。

### 3.10 Ordinal EMD + FW-GPL XGBoost

技术动机：

- 单独 EMD 能表达远距误判代价，但对灰区不确定性处理不足；
- 单独 FW-GPL 能处理标签模糊，但没有把临床灾难性误判的代价推到足够高；
- 两者天然互补。

核心思想：

- 让 soft target 提供“灰区模糊性”；
- 让 EMD surrogate 提供“非对称临床运输代价”；
- 再用一个轻量 soft CE 项作为数值稳定器。

组合损失写成：

- `Loss = lambda_emd * RelaxedEMD + lambda_ce * SoftCE`

当前实验使用：

- `lambda_emd = 0.7`
- `lambda_ce = 0.3`

为何要混合两项：

- 如果只有 EMD，优化会更偏向全局分布搬运，局部类别概率可能不够稳定；
- 如果只有 SoftCE，则无法体现“确诊错成排除”比“排除错成确诊”更严重；
- 混合后既保留分布运输视角，又保留类概率对齐能力。

从临床语义上看，这个模型同时在学习三件事：

- “灰区不是硬边界类别，而是一个过渡带”；
- “0、2、1 不是平行类别，而是有顺序的病程状态”；
- “把确诊错判为排除，比反方向误判更危险”。

当前代码实现：

- 仍使用 `fit_custom_softprob_booster()`；
- 但设置为 `objective_mode="emd", use_fw_gpl=True`；
- 此时 EMD 梯度是对软目标分布而不是 one-hot 目标分布计算的。

公式化表达：

- 对该方案，目标分布不再是 one-hot，而是 FW-GPL 给出的 `q_i`；
- 单样本联合损失定义为：
- $\mathcal{L}_i^{\text{EMD+FW}} = \lambda_{\text{emd}} \mathcal{L}_{\text{EMD},i}(p_i, q_i) + \lambda_{\text{ce}} \mathcal{L}_{\text{CE-soft},i}(p_i, q_i)$
- 其中：
- $\mathcal{L}_{\text{CE-soft},i}(p_i, q_i) = - \sum_{k=0}^{2} q_{ik} \log p_{ik}$
- $\mathcal{L}_{\text{EMD},i}(p_i, q_i)$ 则按累计分布差定义：
- $\mathcal{L}_{\text{EMD},i}(p_i, q_i) \approx \sum_{j=0}^{1} \omega_{ij} \, \Psi(F_i^{p}(j) - F_i^{q}(j))$
- 总目标为：
- $\mathcal{L}_{\text{EMD+FW}} = \sum_{i=1}^{n} \mathcal{L}_i^{\text{EMD+FW}}$
- 当 $\lambda_{\text{emd}} = 0$ 时，退化为纯 FW-GPL；
- 当 $q_i = e(r_i)$ 时，退化为纯 Ordinal EMD。

方法价值与局限：

- 这是当前最完整、最符合临床真实决策逻辑的一条树模型路线；
- 局限在于它的概率输出更激进，因此虽然分类性能最好，但校准不如标准 `baseline_xgboost`。

### 3.11 DML + Ordinal EMD + FW-GPL 联合模型

技术动机：

- 这一方案尝试把本轮所有创新因素一次性叠加到一起，验证是否存在“全组合收益”。

组合结构：

1. 先通过 DML 路线生成 `DML_校正ARR` 和各药物残差；
2. 再把这些增强特征送入 `Ordinal EMD + FW-GPL` 自定义损失模型；
3. 输出三分类概率与连续严重程度评分。

它代表的研究问题是：

- 如果我们同时引入去交杂特征、顺序结构、灰区软标签、非对称临床代价，性能是否会继续单调提升？

当前代码实现：

- 特征生成仍由 `build_dml_features()` 完成；
- 最终训练仍调用 `fit_custom_softprob_booster(objective_mode="emd", use_fw_gpl=True)`；
- 只是输入矩阵换成了 DML 扩展特征。

公式化表达：

- 令 DML 扩展输入记为：
- $x_i^{\text{full}} = [W_i, A_i^{\text{corr}}, R_i]$
- 则该模型直接学习：
- $p_i = f_\theta(x_i^{\text{full}})$
- 其训练目标与 `Ordinal EMD + FW-GPL` 相同，只是输入换成 DML 扩展特征：
- $\mathcal{L}_{\text{DML-EMD-FW}} = \sum_{i=1}^{n} \left[ \lambda_{\text{emd}} \mathcal{L}_{\text{EMD},i}(p_i, q_i) + \lambda_{\text{ce}} \mathcal{L}_{\text{CE-soft},i}(p_i, q_i) \right]$
- 因而它可视为组合映射：
- $x_i \xrightarrow{\text{DML}} x_i^{\text{full}} \xrightarrow{\text{EMD+FW ordinal booster}} p_i$

方法价值与局限：

- 价值在于它是当前方法学复杂度最高、也最接近论文“终极融合版”的路线；
- 局限在于小样本下模块过多会引入噪声叠加，导致理论更强并不必然意味着测试表现更好。

### 3.12 连续严重程度评分

技术动机：

- 离散标签只能回答“更像哪一类”；
- 但对灰区患者而言，临床更关心的是“更偏向排除还是更偏向确诊”，以及距离两侧有多远。

核心思想：

- 不再只看 `argmax` 分类结果，而是把 ordered rank 的期望作为连续病情评分。

定义方式：

- 设内部 rank 为 `[0, 1, 2]`，对应原标签 `[0, 2, 1]`；
- 给定 rank 概率 `p_r`，定义：
- `severity_score = E[rank] / 2`
- 即：
- `severity_score = (0 * p_0 + 1 * p_1 + 2 * p_2) / 2`

这样得到的评分范围是 `[0, 1]`：

- 越接近 `0`，越偏向排除；
- 越接近 `1`，越偏向确诊；
- 灰区患者通常落在中间区间。

为什么它有用：

- 它把离散分类输出提升为连续风险谱；
- 能帮助医生理解灰区患者的“相对位置”，而不只是给出一个生硬的三分类标签；
- 也为后续阈值筛查、动态追踪和治疗前后变化评估提供了更细粒度的指标。

当前代码实现：

- `compute_continuous_severity_score()` 计算评分；
- `evaluate_severity_score()` 评估不同真实类别上的均值单调性；
- `plot_severity_distributions()` 负责输出小提琴图和箱线图。

公式化表达：

- 设 rank 概率为 $p_i = [p_{i0}, p_{i1}, p_{i2}]$；
- 定义内部 rank 随机变量 $R_i \in \{0,1,2\}$；
- 连续严重程度评分定义为其归一化期望：
- $s_i = \frac{\mathbb{E}[R_i \mid x_i]}{2} = \frac{1}{2} \sum_{k=0}^{2} k \, p_{ik}$
- 展开后即：
- $s_i = \frac{p_{i1} + 2 p_{i2}}{2}$
- 其中这里的 $p_{i0}, p_{i1}, p_{i2}$ 指的是 rank 顺序 `[0,2,1]` 下的概率；
- 评价时，对每个真实类别 `c` 计算条件均值：
- $\mu_c = \frac{1}{n_c} \sum_{i: y_i = c} s_i$
- 若满足：
- $\mu_0 < \mu_2 < \mu_1$
- 则说明模型在连续风险谱上保持了临床单调性。

方法价值与局限：

- 价值在于它把有序分类自然扩展成连续疾病谱建模；
- 局限在于它仍依赖前端概率分布质量，因此如果概率校准较差，连续评分的绝对数值解释也需要更谨慎。

## 4. 实现文件与结果目录

### 4.1 主脚本

- [causal_xgboost_variants_experiment.py](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_variants_experiment.py)

### 4.2 结果目录

- `c:\Users\YCY\Desktop\ARR-model\causal_xgboost_outputs`

### 4.3 关键输出

- 摘要：[experiment_summary.json](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/experiment_summary.json)
- 指标表：[variant_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/tables/variant_metrics.csv)
- 扩展指标表：[variant_calibration_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/tables/variant_calibration_metrics.csv)
- 严重度摘要：[variant_severity_summary.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/tables/variant_severity_summary.csv)
- 测试集逐例预测：`causal_xgboost_outputs/test_set_variant_predictions.xlsx`
- 最优方案特征重要性：[best_variant_feature_importance.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/tables/best_variant_feature_importance.csv)

## 5. 核心结果

### 5.1 总体对比结果

根据 [variant_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/tables/variant_metrics.csv)：

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | OVR ROC AUC |
|---|---:|---:|---:|---:|
| ordinal_emd_fw_gpl_xgboost | 0.9600 | 0.8935 | 0.8935 | 0.9614 |
| baseline_xgboost | 0.9467 | 0.8869 | 0.8623 | 0.9704 |
| ordinal_fw_gpl_xgboost | 0.9467 | 0.8869 | 0.8623 | 0.9601 |
| ordinal_xgboost | 0.9333 | 0.8804 | 0.8367 | 0.9785 |
| dml_ordinal_xgboost | 0.9333 | 0.8804 | 0.8367 | 0.9528 |
| ordinal_xgboost_v2 | 0.9333 | 0.8703 | 0.8361 | 0.9650 |
| adaptive_care_xgboost | 0.9333 | 0.8036 | 0.8243 | 0.9758 |
| ordinal_emd_xgboost | 0.9467 | 0.8101 | 0.8151 | 0.9837 |
| dml_ordinal_emd_fw_gpl_xgboost | 0.9467 | 0.8101 | 0.8151 | 0.9403 |
| physiology_informed_xgboost | 0.9200 | 0.7971 | 0.7816 | 0.9772 |
| dml_xgboost | 0.9200 | 0.7869 | 0.7811 | 0.9826 |

### 5.2 总体最优方案

按当前项目的主评价口径：

- `Accuracy`
- `Balanced Accuracy`
- `Macro F1`

综合判断，**当前总体最优方案已经从原始 `baseline_xgboost` 更新为 `ordinal_emd_fw_gpl_xgboost`**。

其测试集结果为：

- Accuracy：`0.9600`
- Balanced Accuracy：`0.8935`
- Macro F1：`0.8935`
- Weighted F1：`0.9600`
- OVR ROC AUC Macro：`0.9614`

与原始 `baseline_xgboost` 相比：

- Accuracy：`0.9467 -> 0.9600`
- Balanced Accuracy：`0.8869 -> 0.8935`
- Macro F1：`0.8623 -> 0.8935`

这说明“非对称临床代价 + 灰区软标签 + 有序结构”的组合，在当前数据上第一次实质性超过了原始最优 `XGBoost` 基线。

### 5.3 创新变体中的分层结论

如果只在创新变体中选择，当前可以分成三档：

- **第一档：`ordinal_emd_fw_gpl_xgboost`**
- **第二档：`ordinal_fw_gpl_xgboost` 与 `ordinal_xgboost` / `dml_ordinal_xgboost`**
- **第三档：`ordinal_xgboost_v2` 与其余增强路线**

其中：

- `ordinal_emd_fw_gpl_xgboost`
  - Accuracy：`0.9600`
  - Balanced Accuracy：`0.8935`
  - Macro F1：`0.8935`
  - `0=非确诊` recall：`0.75`
  - `1=确诊` recall：`0.9804`
  - `2=灰区` recall：`0.95`

- `ordinal_fw_gpl_xgboost`
  - Accuracy：`0.9467`
  - Balanced Accuracy：`0.8869`
  - Macro F1：`0.8623`

- `ordinal_xgboost`
  - Accuracy：`0.9333`
  - Balanced Accuracy：`0.8804`
  - Macro F1：`0.8367`

- `dml_ordinal_xgboost`
  - Accuracy：`0.9333`
  - Balanced Accuracy：`0.8804`
  - Macro F1：`0.8367`

结论：

- 如果目标是“当前数据上的绝对分类性能”，主推 `ordinal_emd_fw_gpl_xgboost`
- 如果目标是“更保守、更易解释且结构简单”的有序路线，`ordinal_xgboost` 仍然是更简洁的主线
- 如果目标是“因果特征 + 有序结构”的论文补充方案，`dml_ordinal_xgboost` 仍有独立价值

## 6. 各变体结果解读

### 6.1 DML-XGBoost

结果：

- Accuracy：`0.9200`
- Balanced Accuracy：`0.7869`
- Macro F1：`0.7811`
- AUC：`0.9826`

解读：

- 排序能力强，但最终离散分类能力偏弱；
- 说明单独做去交杂特征生成，仍然会损失一部分对小样本分类有用的方差。

结论：

- 更适合作为辅助特征生成模块；
- 不适合直接替代最终判别模型。

### 6.2 Adaptive-CaRe inspired XGBoost

结果：

- Accuracy：`0.9333`
- Balanced Accuracy：`0.8036`
- Macro F1：`0.8243`

解读：

- 保留了一定总体性能；
- 但少数类和整体鲁棒性都不如最优有序方案。

结论：

- 有方法学故事性；
- 但当前近似实现仍不是最佳选择。

### 6.3 Ordinal XGBoost

结果：

- Accuracy：`0.9333`
- Balanced Accuracy：`0.8804`
- Macro F1：`0.8367`
- AUC：`0.9785`

解读：

- 它已经证明“基于二阈值去做三重标签预测”是合理的；
- 少数类性能稳定；
- 同时把 `0-2-1` 的临床顺序结构自然编码到模型中。

结论：

- 它仍然是最稳、最简洁的有序建模基线。

### 6.4 Physiology-Informed XGBoost

结果：

- Accuracy：`0.9200`
- Balanced Accuracy：`0.7971`
- Macro F1：`0.7816`

解读：

- 显式药理特征交叉没有显著提分；
- 但仍保留一定临床解释价值。

结论：

- 更适合作为解释层增强方案；
- 不适合作为主模型。

### 6.5 Ordinal XGBoost V2

结果：

- Accuracy：`0.9333`
- Balanced Accuracy：`0.8703`
- Macro F1：`0.8361`

解读：

- 只做“阈值代价权重 + 风险重标定”的第二版强化，没有带来超越；
- 说明单纯后处理近似还不够，需要把临床代价推进到训练目标内部。

结论：

- 证明了代价矩阵近似可实现；
- 但不足以成为最终主线。

### 6.6 DML + Ordinal 联合模型

结果：

- Accuracy：`0.9333`
- Balanced Accuracy：`0.8804`
- Macro F1：`0.8367`

解读：

- 它把原本偏弱的 `DML` 路线提升到与 `ordinal_xgboost` 持平；
- 说明 `DML_校正ARR` 在进入有序判别结构后，确实能够发挥更大价值。

结论：

- 是当前最值得保留的“因果 + 有序”联合版本。

### 6.7 Ordinal EMD XGBoost

结果：

- Accuracy：`0.9467`
- Balanced Accuracy：`0.8101`
- Macro F1：`0.8151`
- AUC：`0.9837`

解读：

- 单独引入 Relaxed EMD 后，AUC 很高；
- 但平衡准确率和宏平均 F1 明显低于最佳有序方案；
- 说明“只有临床代价、不处理灰区软标签”会让模型更偏向拉开远距类别，而未能充分利用灰区不确定性。

结论：

- 单独 EMD 不够；
- 需要与灰区软标签联合使用。

### 6.8 Ordinal FW-GPL XGBoost

结果：

- Accuracy：`0.9467`
- Balanced Accuracy：`0.8869`
- Macro F1：`0.8623`
- AUC：`0.9601`

解读：

- 仅引入灰区软标签就已经把有序模型提升到了与原始 `baseline_xgboost` 持平；
- 且保留了 `0=非确诊` 的 `0.75` recall；
- 说明把灰区视为“0 和 1 之间的暧昧概率区间”是有效的。

结论：

- `FW-GPL` 是这轮最成功的单独增强之一；
- 证明灰区不确定性建模本身就有明显价值。

### 6.9 Ordinal EMD + FW-GPL XGBoost

结果：

- Accuracy：`0.9600`
- Balanced Accuracy：`0.8935`
- Macro F1：`0.8935`
- AUC：`0.9614`

解读：

- 这是本轮真正突破性的结果；
- 它同时保留了：
  - `0=非确诊` recall：`0.75`
  - `2=灰区` recall：`0.95`
  - `1=确诊` recall：`0.9804`
- 说明把“非对称临床代价”与“灰区软标签”联合起来，确实比单独使用任何一条增强路线更强。

结论：

- `ordinal_emd_fw_gpl_xgboost` 是当前最优的论文主方案与实验主方案。

### 6.10 DML + Ordinal EMD + FW-GPL 联合模型

结果：

- Accuracy：`0.9467`
- Balanced Accuracy：`0.8101`
- Macro F1：`0.8151`

解读：

- 当 `DML_校正ARR` 注入最强的 EMD + FW-GPL 有序结构时，结果并未继续提升；
- 说明在当前数据上，`DML` 特征对简单有序模型有帮助，但对已经足够强的“灰区软标签 + 临床代价”组合反而可能引入额外噪声。

结论：

- `DML + EMD + FW-GPL + Ordinal` 不是当前最优；
- 最合适的联合模型仍是较轻量的 `dml_ordinal_xgboost`。

## 7. 校准与连续严重程度结果

### 7.1 校准结果

根据 [variant_calibration_metrics.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/tables/variant_calibration_metrics.csv)：

- `baseline_xgboost`
  - ECE：`0.0315`
  - Brier：`0.0858`

- `ordinal_emd_fw_gpl_xgboost`
  - ECE：`0.2343`
  - Brier：`0.1584`

解读：

- 当前最优分类器 `ordinal_emd_fw_gpl_xgboost` 的校准明显弱于 `baseline_xgboost`；
- 这说明它是一条“更强分类性能，但更激进的概率输出”路线；
- 因此如果用于概率解释或阈值决策，后续应补做专门概率校准。

### 7.2 连续严重程度评分

根据 [variant_severity_summary.csv](file:///c:/Users/YCY/Desktop/ARR-model/causal_xgboost_outputs/tables/variant_severity_summary.csv)，所有变体都满足：

- `排除 < 灰区 < 确诊`

其中当前最优模型 `ordinal_emd_fw_gpl_xgboost` 的类均值为：

- `0类`：`0.3983`
- `2类`：`0.4820`
- `1类`：`0.8422`

说明：

- 虽然该模型校准更激进，但在连续严重度谱上仍维持合理的临床单调性；
- 这使其不仅能输出离散标签，还能为灰区患者提供连续风险定位。

## 8. 图表说明

结果图位于 `c:\Users\YCY\Desktop\ARR-model\causal_xgboost_outputs\figures`

- `variant_comparison.png`
  - 全部变体在 `Accuracy / Balanced Accuracy / Macro F1 / AUC` 上的并列柱状图
- `variant_recall_heatmap.png`
  - 各方案在 `非确诊 / 确诊 / 灰色区域` 三类上的召回率热力图
- `variant_roc_overview.png`
  - 各变体对“确诊类”的 ROC 对比图
- `variant_calibration_comparison.png`
  - 各变体 `ECE / Brier` 校准对比图
- `severity_score_violin.png`
  - 各变体连续严重程度评分小提琴图
- `severity_score_boxplot.png`
  - 各变体连续严重程度评分箱线图
- `best_variant_confusion_heatmap.png`
  - 当前总体最优方案的混淆矩阵
- `best_variant_multiclass_roc.png`
  - 当前总体最优方案的多分类 ROC 图

## 9. 后续建议

下一步最值得做的是：

1. 保留 `ordinal_emd_fw_gpl_xgboost` 作为主线，补做概率校准，以解决当前更激进的概率输出问题。
2. 在论文中同时保留 `ordinal_xgboost` 作为简洁对照，突出“灰区是暧昧概率区间”这一核心思想。
3. 将 `dml_ordinal_xgboost` 作为因果联合补充方案，不建议继续把 `DML` 硬叠加到最复杂的 EMD + FW-GPL 路线上。
4. 如果目标是更接近真实筛查场景，建议再补一轮“去除确诊后变量”的严格前瞻性版本。
