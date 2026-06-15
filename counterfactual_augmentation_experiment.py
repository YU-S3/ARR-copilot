from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from frontier_scm_v2_experiment import DEFAULT_SEEDS, ProgressTracker, prepare_split_data
from frontier_scm_v3_experiment import fit_scm_v3_config, fit_xgb_with_safe_adasyn
from scm_v3_augmentation import evaluate_model_from_raw, generate_counterfactual_rows, summarize_model_metrics


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "counterfactual_aug_outputs"
SCM_V3_REFERENCE_CONFIG = {
    "seed_strategy": "hard_case_seed",
    "target_classes": "class0_only",
    "treat_mix_prob": 0.40,
    "sampler_strength": 0.65,
    "node_sampler": "conditional_kde",
    "rule_filter_mode": "medical_rules",
    "teacher_filter_mode": "single",
    "curriculum_mode": "off",
    "counterfactual_mode": "treatment_do",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="反事实增广原型实验。")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    tracker = ProgressTracker(OUTPUT_DIR, total_steps=len(seeds) * 3)
    rows: list[dict] = []
    audit_frames: list[pd.DataFrame] = []

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

        cf_result = generate_counterfactual_rows(
            split_data.X_train_raw,
            split_data.y_train,
            random_state=seed,
            max_rows=18 if args.smoke_test else None,
        )
        X_train_cf = pd.concat([split_data.X_train_raw, cf_result.X_aug], axis=0, ignore_index=True)
        y_train_cf = pd.concat([split_data.y_train, cf_result.y_aug], axis=0, ignore_index=True)
        cf_model = fit_xgb_with_safe_adasyn(
            X_train_cf,
            y_train_cf,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed=seed,
        )
        rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_counterfactual_proto",
                y_test=split_data.y_test,
                proba=cf_model.proba,
                augmented_size=int(len(cf_result.X_aug)),
                train_size=cf_model.train_size,
                resampled_train_size=cf_model.resampled_train_size,
            )
        )
        audit = cf_result.audit.copy()
        audit["seed"] = seed
        audit_frames.append(audit)
        tracker.log("完成 counterfactual 原型", stage="counterfactual_done", advance=1, context={"seed": seed, "augmented_size": len(cf_result.X_aug)})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize_model_metrics(metrics_df, metrics_df["model_name"].unique().tolist())
    summary_df.to_csv(tables_dir / "metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    if audit_frames:
        pd.concat(audit_frames, axis=0, ignore_index=True).to_csv(
            tables_dir / "augmentation_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary = {
        "seeds": seeds,
        "smoke_test": args.smoke_test,
        "models": metrics_df["model_name"].unique().tolist(),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
