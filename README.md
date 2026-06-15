# ARR / PA Screening Experiment Code

本仓库整理 ARR / 原发性醛固酮增多症（PA）筛查与诊断相关实验代码。上传内容以代码、运行脚本、依赖说明和整理后的 Markdown 报告为主；临床原始表格、模型权重、生成结果表、图表和日志仍保留在本地并由 `.gitignore` 排除。

## 仓库内容

- 实验脚本：传统机器学习、筛查特征策略、因果 XGBoost、SCM-v2/v3 增广、TabPFN、本地 TableGPT2/QLoRA 原型、PA 筛查 API。
- 运行脚本：PowerShell / Bash 一键复现实验、API 启动脚本、进度查看脚本。
- 依赖文件：`requirements*.txt`、`environment*.yml`。
- 报告归档：`experiment_reports/` 中集中保存已整理的 Markdown 实验报告。

## 未纳入版本控制

- 临床数据表格，例如 `data_0428.xlsx`、`数据表格测试.xlsx`、`data_fixed.xlsx` 和复核工作簿。
- 本地模型权重和缓存，例如 `models/`、`TabPFN/`、`models.zip`、checkpoint、adapter、`.pt/.pth/.bin/.safetensors` 等。
- 生成的实验输出目录、CSV/XLSX 结果表、图片、日志、PDF、Word 报告和压缩包。
- `.env` 密钥文件。需要本地密钥时从 `.env.example` 复制。

## 核心数据约定

- 主目标列：`确诊（0为排除；1为确诊；2为灰色区域）`。
- 主要筛查目标：
  - 三分类：`0=排除/非确诊`、`1=确诊`、`2=灰色区域`。
  - 二分类：通常将 `1/2` 视为需要进一步评估，`0` 视为筛查阴性。
- 重要防泄漏策略：`screening_no_post` 排除确诊试验后的醛固酮、肾素等 post-test 特征；`full_reference` 仅作参考；`post_mask_stress` 用于泄漏/稳健性压力测试。

## 脚本与实验方案索引

### 基线、数据清洗与共享工具

| 脚本 | 方案内容 |
| --- | --- |
| `multiclass_ensemble_experiment.py` | 多分类主基线。负责数据清洗、目标列规范化、零填充药物分数、预处理器构建；训练 XGBoost / LightGBM / CatBoost，比较 ADASYN、校准、加权融合、按类别融合、两阶段策略，并输出 SHAP 解释。 |
| `train_random_forest.py` | 轻量 RandomForest 基线。读取表格、清洗三分类目标、训练随机森林，输出指标、预测表、特征重要性和 `joblib` 模型。 |
| `env_utils.py` | 本地环境变量工具。读取 `.env`，为 TabPFN API token 等实验密钥提供统一加载入口。 |
| `watch_rerun_progress.py` | 复现实验进度查看器。读取 `rerun_manifest.json` 和各步骤 `progress.json/progress.log`，展示 smoke/full 流水线当前状态。 |

### 筛查策略与传统模型实验

| 脚本 | 方案内容 |
| --- | --- |
| `screening_0428_experiment.py` | 0428 筛查主实验。比较 `full_reference`、`screening_no_post`、`post_mask_stress` 三类特征策略；同时跑 binary / three-class 任务；候选包含 XGBoost、CatBoost、因果/ordinal XGBoost、SCM-v2 增广 XGBoost，并支持 5/10 折交叉验证。 |
| `screening_tuning_0428_experiment.py` | 筛查模型调参实验。围绕 `screening_no_post` 做正则强度、树深、类别权重、SCM 增广、二分类阈值和校准策略搜索；二分类候选包括 `xgb_bin_d*_l*`、`xgb_bin_scm_*`，三分类候选包括 CatBoost、`three_xgb_scm_*`、`three_ordinal_scm_*`。 |
| `screening_constrained_0428_experiment.py` | 高敏感度约束筛查实验。构造 ARR、醛固酮/肾素、血钾、血压、用药干扰等 engineered features；比较 XGBoost、CatBoost、Balanced Random Forest、Easy Ensemble、soft voting；阈值策略覆盖 fixed 0.50 以及 sensitivity >= 0.90/0.93/0.95 下最大 specificity / accuracy / balanced accuracy。 |
| `screening_diagnostics_0428_experiment.py` | 筛查诊断分析。对 binary 和 three-class 方案做 OOF 误差分析、概率校准、阈值表现、缺失/特征策略审计、过拟合和失败样本定位。 |

### 因果 XGBoost、SCM 与合成增广

| 脚本 | 方案内容 |
| --- | --- |
| `causal_xgboost_variants_experiment.py` | 因果/机制 XGBoost 变体总实验。比较 baseline XGBoost、DML 残差化、physiology-informed 特征、ordinal EMD、FW-GPL soft labels、DML+ordinal+EMD+FW-GPL 组合；包含主线校准、strict prospective 评估、外部 full-training 评估接口、严重度分数和最佳变体解释。 |
| `frontier_augmentation.py` | 前沿增广基础模块。实现 `TapInspiredInpaintingAugmentor` 和 `SCMMixAugmentor`，提供条件插补、锚点距离、bootstrap、XGBoost teacher、生成样本 audit/metadata 等共享能力。 |
| `frontier_augmentation_ablation_experiment.py` | 前沿增广消融实验。跨多个随机种子比较原始 XGBoost、TAP-style inpainting、SCM mix 等增广方案，输出样本分布漂移、增广审计和指标汇总。 |
| `frontier_parallel_experiment.py` | 小样本前沿并行实验。并行比较 XGBoost baseline、SCM/TAP 增广方案和可选 TabPFN teacher/reference，用于早期筛选可行增广路线。 |
| `frontier_scm_v2_experiment.py` | SCM-v2 主实验。Phase 1 做 baseline 与配置验证；Phase 2 遍历 hard-case seed、目标类别、treatment mix、residual scale、teacher mode；Phase 3 在有 token 时加入 TabPFN 参照/扩展。 |
| `frontier_scm_v3_experiment.py` | SCM-v3 改进实验。基于 SCM-v2 引入 conditional KDE node sampler、medical rules、teacher filter、curriculum、可选 treatment-do counterfactual；比较 XGB baseline、SCM-v2 reference、SCM-v3 配置和可选 TabPFN+SCM-v3。 |
| `scm_v3_augmentation.py` | SCM-v3 增广核心模块。实现 `SCMMixV3Augmentor`、节点级条件采样、规则过滤、teacher 过滤、课程式增广、反事实样本生成和跨 seed 指标汇总。 |
| `scm_v3_medical_rules.py` | 医学规则过滤模块。对合成样本执行非负、生理范围、ARR/醛固酮/肾素关系、血压/电解质等临床合理性检查，并返回规则通过/拒绝原因。 |
| `scm_v3_generalization_experiment.py` | SCM-v3 泛化与校准诊断。使用 train/validation/test 三段切分，比较 `xgb_reference_adasyn`、`xgb_scm_v2_best`、`scm_v3_res_norule_single_flat_nocf` 和可选 `xgb_tabddpm_proto`，生成 PR 曲线、校准、class-0 precision/recall 与 Markdown 诊断报告。 |
| `scm_v31_experiment.py` | SCM V3.1 跟进实验。对 `xgb_scm_v2_best` 和 `scm_v3_res_norule_single_flat_nocf` 做 uncalibrated / temperature / OVR sigmoid / OVR isotonic 校准比较，并加入 TabDDPM 生成样本量与 epoch 网格。 |
| `joint_causal_scm_v2_experiment.py` | 联合因果增强 + SCM-v2 实验。比较因果 XGBoost 主线、`xgb_scm_v2_best`、`causal_scm_v2_joint`、`ordinal_scm_v2_joint`，评估 SCM-v2 增广与因果/ordinal 损失组合是否互补。 |
| `counterfactual_augmentation_experiment.py` | 反事实增广原型。以 SCM-v3 reference 为对照，启用 `counterfactual_mode=treatment_do` 生成治疗/用药干预反事实样本，并比较 XGBoost baseline、SCM-v3 reference、counterfactual prototype。 |
| `great_prototype_experiment.py` | GReaT 风格表格生成原型。将病例特征序列化为文本式表格样本，生成 class-0 方向合成数据，对比 XGBoost baseline、SCM-v3 reference、GReaT-style augmentation。 |
| `tabddpm_prototype_experiment.py` | TabDDPM 原型实验。用轻量 denoising diffusion 原型生成表格样本，对比 XGBoost baseline、SCM-v3 reference、TabDDPM augmentation。 |
| `medical_deconfounder_experiment.py` | 医学去交杂器实验。用 VAE/latent treatment route 建模用药和检查干扰，构造反事实 ARR/治疗表示，并与 XGBoost 预测表现、患者级反事实输出一起评估。 |
| `raas_pinn_tabpfn_experiment.py` | RAAS 机制约束神经网络与 TabPFN 对照。实现 RAAS/PINN 风格分类器，加入生理约束损失，和 TabPFN 预测进行对比或组合分析。 |

### TabPFN 相关实验

| 脚本 | 方案内容 |
| --- | --- |
| `tabpfn_only_experiment.py` | TabPFN 单独实验。调用 TabPFN client，与 XGBoost 参照比较三分类表现、ROC 和预测明细。 |
| `tabpfn_cost_sensitive_experiment.py` | TabPFN 代价敏感实验。比较原始 TabPFN、ADASYN 训练样本、代价敏感/think-mode 等模式，关注确诊类 OVR ROC 和不同训练模式指标。 |
| `tabpfn_screening_no_post_experiment.py` | 本地 TabPFN-3 无 post-test 筛查实验。支持 binary / three / both 任务；默认 `screening_no_post`，可加 `full_reference` 和 `post_mask_stress` 审计；候选包括 multiclass/binary、balanced 和 default checkpoint 变体；支持 5/10 折与多 seed。 |
| `tabpfn_xgb_fusion_paper_experiment.py` | 论文版 TabPFN + XGB 二分类融合。采用 no-post 严格特征策略、nested CV、OOF 阈值选择、sigmoid/isotonic 校准、概率 blend/stacking，围绕 `xgb_binary_scm_v2` 和 TabPFN binary 变体生成论文级筛查指标。 |
| `tabpfn_traditional_fusion_experiment.py` | TabPFN + 传统模型融合扩展。候选集包括 `xgb_binary_scm_v2`、筛查调参候选、约束筛查候选；支持 `first_batch`、`fusion_core`、`constrained`、`all`；评估 OOF selection、stacking、top-2 traditional、cascade gate/rescue 和敏感度资格筛选。 |
| `tabpfn_binary_threshold_fusion.py` | TabPFN 二分类阈值/融合实验。针对 TabPFN binary 变体与 `xgb_binary_scm_v2`，在 validation 上选择 default、max balanced accuracy、sensitivity >= 0.90/0.92/0.95 阈值，比较 sigmoid/isotonic 校准和 TabPFN-XGB blend。 |

### TableGPT2 / QLoRA 相关脚本

| 脚本 | 方案内容 |
| --- | --- |
| `tablegpt2_pa/__init__.py` | TableGPT2 PA 实验包声明。 |
| `tablegpt2_pa/common.py` | TableGPT2 共享数据与 prompt 工具。清洗目标列，构造协议 A/B/C 二分类数据集，拆分 train/val/test/ICL pool，选择医学优先特征，生成 few-shot 表格 prompt 和 JSON response。 |
| `tablegpt2_pa/download_model.py` | TableGPT2-7B 下载脚本。支持 Hugging Face endpoint/mirror、`snapshot_download`、direct resolve、git-lfs 等下载方式，目标目录默认为 `models/TableGPT2-7B`。 |
| `tablegpt2_pa/run_comparison.py` | TableGPT2 方案的传统基线。按协议 A/B/C 构造 PA 二分类任务，训练 RandomForest 和 XGBoost，对照后续 QLoRA 生成式模型。 |
| `tablegpt2_pa/train_qlora.py` | TableGPT2/Qwen checkpoint 的 PA QLoRA 微调。支持协议 A/B/C、top-p 特征数、k-shot prompt、4bit、bf16、LoRA target module 自动识别，输出 adapter、trainer 目录、预测表和实验摘要。 |
| `tablegpt2_pa/package_bundle.py` | TableGPT2 PA 实验打包脚本。将代码、依赖、可选模型目录和可选输出目录打包，便于迁移到服务器运行。 |

### API 与一键运行脚本

| 脚本 | 方案内容 |
| --- | --- |
| `pa_api_backend.py` | 本地 PA 筛查 HTTP API。训练或加载 `xgb_bin_d3_l20_bal_isotonic` 二分类模型，固定 `screening_no_post`，提供 `/health` 和 `POST /pa-diagnosis/predict`，输出 balanced 与 high-sensitivity 两种模式判断。 |
| `start_pa_api_backend.bat` | Windows API 启动脚本。优先使用指定 conda 环境 Python，失败时回退 `conda run -n arr_rf python`，启动 `pa_api_backend.py --host 0.0.0.0 --port 8000`。 |
| `run_arr_rerun_0428.ps1` | Windows 复现总流水线。按 smoke/full 模式依次运行 `multiclass_ensemble_experiment.py`、`frontier_scm_v2_experiment.py`、`frontier_scm_v3_experiment.py`、`scm_v31_experiment.py`，生成 manifest、分步日志和进度文件。 |
| `run_arr_rerun_0428.sh` | Bash 版复现总流水线。功能对应 PowerShell 版，面向 Linux/macOS/服务器环境运行 ensemble、SCM-v2、SCM-v3、SCM-v31 四步。 |
| `run_screening_0428.ps1` | Windows 筛查主实验启动脚本。运行 `screening_0428_experiment.py`，用于 no-post screening 主方案及特征策略审计。 |
| `run_screening_tuning_0428.ps1` | Windows 筛查调参启动脚本。运行 `screening_tuning_0428_experiment.py`，输出调参、阈值和确认候选。 |
| `run_screening_constrained_0428.ps1` | Windows 高敏感度约束实验启动脚本。运行 `screening_constrained_0428_experiment.py`，并在缺省环境下尝试 `conda run -n arr_rf python`。 |
| `run_screening_diagnostics_0428.ps1` | Windows 筛查诊断启动脚本。运行 `screening_diagnostics_0428_experiment.py`，生成误差、校准、缺失和策略审计结果。 |
| `run_tabpfn_screening_no_post.ps1` | Windows 本地 TabPFN 筛查启动脚本。调用 `tabpfn_screening_no_post_experiment.py`，默认使用本地 TabPFN checkpoint 目录和 arr_rf 环境。 |

## 本地运行提示

基础环境：

```powershell
pip install -r requirements.txt
copy .env.example .env
```

TabPFN、TableGPT2 或 QLoRA 实验需要额外下载对应 checkpoint，并保持在 `.gitignore` 已排除的本地目录中。运行后生成的 `*_outputs/`、`rerun_0428_outputs/`、`tablegpt2_pa_outputs/`、`models/` 等目录不要提交到仓库。

常用命令示例：

```powershell
python multiclass_ensemble_experiment.py --input data_0428.xlsx
python screening_0428_experiment.py --feature-policy screening_no_post --task-mode both
python frontier_scm_v3_experiment.py --best-only --skip-tabpfn
python watch_rerun_progress.py full
```
