# ARR / PA Screening Experiment Code

This repository is prepared as a code-only upload for ARR / primary aldosteronism screening experiments. Clinical data files, model weights, generated reports, figures, logs, and experiment result tables are intentionally excluded from version control.

## What Is Included

- Experiment scripts for traditional ML, causal XGBoost variants, SCM-v2/v3 augmentation, TabPFN comparisons, TableGPT2/QLoRA prototypes, and the local PA screening API.
- Dependency files such as `requirements*.txt` and `environment*.yml`.
- Local run scripts such as `run_screening_0428.ps1`, `run_arr_rerun_0428.ps1`, and `run_tabpfn_screening_no_post.ps1`.

## What Is Not Included

- Clinical spreadsheets such as `data_0428.xlsx`, `数据表格测试.xlsx`, and review workbooks.
- Model weights and caches under `models/`, `TabPFN/`, `models.zip`, and checkpoint files.
- Generated experiment outputs, CSV tables, figures, logs, PDFs, Word reports, and rerun folders.
- `.env` secrets. Use `.env.example` as the local template.

## Main Entry Points

- `multiclass_ensemble_experiment.py`: shared data cleaning and multiclass ensemble baseline.
- `screening_0428_experiment.py`: no-post-test screening feature policies and binary/three-class screening experiments.
- `screening_tuning_0428_experiment.py`: regularized and threshold-tuned screening models.
- `screening_constrained_0428_experiment.py`: sensitivity-constrained screening candidate selection.
- `screening_diagnostics_0428_experiment.py`: error analysis and data quality diagnostics.
- `causal_xgboost_variants_experiment.py`: ordinal, cost-aware, and causal XGBoost variants.
- `frontier_scm_v2_experiment.py`, `frontier_scm_v3_experiment.py`, `scm_v3_augmentation.py`: structured augmentation experiments.
- `tabpfn_screening_no_post_experiment.py`, `tabpfn_xgb_fusion_paper_experiment.py`, `tabpfn_traditional_fusion_experiment.py`: local TabPFN screening and fusion experiments.
- `tablegpt2_pa/`: TableGPT2-oriented PA binary experiment utilities.
- `pa_api_backend.py`: local HTTP API for the binary screening model.

## Local Setup Notes

The scripts expect local data and optional model assets to exist beside the code, but those files are not part of the repository. Recreate them locally before running experiments, then keep generated outputs outside Git.

```powershell
pip install -r requirements.txt
copy .env.example .env
```

For TabPFN or TableGPT2 runs, download the required checkpoints locally according to each model license and keep them ignored by Git.
