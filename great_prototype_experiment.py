from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frontier_augmentation import POST_TREATMENT_COLUMNS, SCREENING_COLUMNS, TREATMENT_COLUMNS, clip_numeric_value, infer_discrete_numeric_features
from frontier_scm_v2_experiment import DEFAULT_SEEDS, ProgressTracker, prepare_split_data
from frontier_scm_v3_experiment import fit_scm_v3_config, fit_xgb_with_safe_adasyn
from scm_v3_augmentation import evaluate_model_from_raw, summarize_model_metrics


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "great_proto_outputs"
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "TableGPT2-7B"
SCM_V3_REFERENCE_CONFIG = {
    "seed_strategy": "hard_case_seed",
    "target_classes": "class0_only",
    "treat_mix_prob": 0.40,
    "sampler_strength": 0.65,
    "node_sampler": "conditional_kde",
    "rule_filter_mode": "medical_rules",
    "teacher_filter_mode": "single",
    "curriculum_mode": "off",
    "counterfactual_mode": "off",
}
TEXT_PRIORITY_COLUMNS = [
    "年龄",
    "性别",
    "确诊实验类型",
    "是否有肾上腺结节",
    "是否有增生",
    "结节最大直径",
    *TREATMENT_COLUMNS,
    *SCREENING_COLUMNS,
    *POST_TREATMENT_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GReaT 风格表格生成原型实验。")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--backend", type=str, default="template", choices=["template", "hf_local"])
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    return parser.parse_args()


def row_to_text(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for col in columns:
        if col not in row.index or pd.isna(row[col]):
            continue
        parts.append(f"{col}={row[col]}")
    return "；".join(parts)


def text_to_row(text: str, template_row: pd.Series, columns: list[str], support_df: pd.DataFrame) -> pd.Series:
    candidate = template_row.copy()
    kv_map: dict[str, str] = {}
    for segment in text.replace(";", "；").split("；"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        kv_map[key.strip()] = value.strip()

    discrete_features = set(infer_discrete_numeric_features(support_df))
    for col in columns:
        if col not in kv_map:
            continue
        value = kv_map[col]
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            candidate[col] = value
            continue
        numeric = clip_numeric_value(float(numeric), pd.to_numeric(support_df[col], errors="coerce"))
        if col in discrete_features:
            numeric = float(np.round(numeric))
        candidate[col] = numeric
    return candidate


def generate_template_text(
    class_df: pd.DataFrame,
    *,
    seed: int,
    max_rows: int,
    columns: list[str],
) -> list[str]:
    rng = np.random.default_rng(seed)
    rows: list[str] = []
    numeric_columns = [col for col in columns if col in class_df.columns and pd.api.types.is_numeric_dtype(class_df[col])]
    support_cache = {col: pd.to_numeric(class_df[col], errors="coerce") for col in numeric_columns}

    for _ in range(max_rows):
        base_row = class_df.iloc[int(rng.integers(0, len(class_df)))].copy()
        donor_row = class_df.iloc[int(rng.integers(0, len(class_df)))].copy()
        candidate = base_row.copy()
        for col in numeric_columns:
            base_value = pd.to_numeric(pd.Series([base_row[col]]), errors="coerce").iloc[0]
            donor_value = pd.to_numeric(pd.Series([donor_row[col]]), errors="coerce").iloc[0]
            if pd.isna(base_value) or pd.isna(donor_value):
                continue
            mixed = 0.7 * float(base_value) + 0.3 * float(donor_value)
            noise = rng.normal(0.0, max(float(support_cache[col].std(ddof=0) or 0.0) * 0.05, 1e-4))
            candidate[col] = clip_numeric_value(mixed + noise, support_cache[col])
        rows.append(row_to_text(candidate, columns))
    return rows


def generate_hf_local_text(
    class_df: pd.DataFrame,
    *,
    seed: int,
    max_rows: int,
    columns: list[str],
    model_dir: Path,
) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    rng = np.random.default_rng(seed)
    prompts: list[str] = []
    for _ in range(max_rows):
        examples = class_df.sample(n=min(3, len(class_df)), replace=len(class_df) < 3, random_state=int(rng.integers(0, 1_000_000)))
        prefix = "以下是 0 类真实患者示例：\n"
        for _, row in examples.iterrows():
            prefix += row_to_text(row, columns) + "\n"
        prefix += "请生成一个新的、合理的 0 类患者记录：\n"
        encoded = tokenizer(prefix, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=120,
                do_sample=True,
                top_p=0.92,
                temperature=0.85,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        candidate = text.split("请生成一个新的、合理的 0 类患者记录：")[-1].strip().splitlines()[0]
        prompts.append(candidate)
    return prompts


def generate_great_rows(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int,
    backend: str,
    model_dir: Path,
    smoke_test: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    class_zero = X_train_raw.loc[y_train == 0].copy()
    if class_zero.empty:
        return pd.DataFrame(columns=X_train_raw.columns), pd.Series(dtype=int, name=y_train.name)

    columns = [col for col in TEXT_PRIORITY_COLUMNS if col in X_train_raw.columns]
    max_rows = 12 if smoke_test else min(max(len(class_zero) * 2, 24), 80)
    if backend == "hf_local":
        texts = generate_hf_local_text(class_zero, seed=seed, max_rows=max_rows, columns=columns, model_dir=model_dir)
    else:
        texts = generate_template_text(class_zero, seed=seed, max_rows=max_rows, columns=columns)

    rng = np.random.default_rng(seed + 123)
    rows: list[pd.Series] = []
    for text in texts:
        template_row = class_zero.iloc[int(rng.integers(0, len(class_zero)))].copy()
        rows.append(text_to_row(text, template_row, columns, X_train_raw))

    augmented_df = pd.DataFrame(rows).reset_index(drop=True)
    y_aug = pd.Series(np.zeros(len(augmented_df), dtype=int), name=y_train.name)
    return augmented_df, y_aug


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    tracker = ProgressTracker(OUTPUT_DIR, total_steps=len(seeds) * 3)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        split_data = prepare_split_data(PROJECT_DIR, seed)

        baseline = fit_xgb_with_safe_adasyn(
            split_data.X_train_raw,
            split_data.y_train,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed=seed,
        )
        rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_reference_adasyn",
                y_test=split_data.y_test,
                proba=baseline.proba,
                augmented_size=0,
                train_size=baseline.train_size,
                resampled_train_size=baseline.resampled_train_size,
            )
        )
        tracker.log("完成 baseline", stage="baseline_done", advance=1, context={"seed": seed})

        scm_v3_result, scm_v3_aug_size, _ = fit_scm_v3_config(split_data, seed, SCM_V3_REFERENCE_CONFIG)
        rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_scm_v3_reference",
                y_test=split_data.y_test,
                proba=scm_v3_result.proba,
                augmented_size=scm_v3_aug_size,
                train_size=scm_v3_result.train_size,
                resampled_train_size=scm_v3_result.resampled_train_size,
            )
        )
        tracker.log("完成 SCM_v3 参照", stage="scm_v3_done", advance=1, context={"seed": seed})

        X_aug, y_aug = generate_great_rows(
            split_data.X_train_raw,
            split_data.y_train,
            seed=seed,
            backend="template" if args.smoke_test else args.backend,
            model_dir=Path(args.model_dir),
            smoke_test=args.smoke_test,
        )
        X_train_aug = pd.concat([split_data.X_train_raw, X_aug], axis=0, ignore_index=True)
        y_train_aug = pd.concat([split_data.y_train, y_aug], axis=0, ignore_index=True)
        great_model = fit_xgb_with_safe_adasyn(
            X_train_aug,
            y_train_aug,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed=seed,
        )
        rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_great_proto",
                y_test=split_data.y_test,
                proba=great_model.proba,
                augmented_size=int(len(X_aug)),
                train_size=great_model.train_size,
                resampled_train_size=great_model.resampled_train_size,
            )
        )
        tracker.log("完成 GReaT 原型", stage="great_done", advance=1, context={"seed": seed, "augmented_size": len(X_aug)})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize_model_metrics(metrics_df, metrics_df["model_name"].unique().tolist())
    summary_df.to_csv(tables_dir / "metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    summary = {
        "seeds": seeds,
        "smoke_test": args.smoke_test,
        "backend": "template" if args.smoke_test else args.backend,
        "model_dir": str(args.model_dir),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
