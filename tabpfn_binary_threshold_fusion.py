from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, prepare_data
from screening_0428_experiment import (
    apply_feature_policy,
    fit_bundle_by_name,
    predict_bundle,
    task_target,
    aggregate_metrics,
)
from tabpfn_screening_no_post_experiment import (
    TABPFN_DIR,
    VARIANTS,
    validate_checkpoints,
    fit_predict_tabpfn,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "rerun_0428_outputs" / "tabpfn_screening_no_post" / "binary_threshold_fusion"
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]
SENSITIVITY_TARGETS = [0.90, 0.92, 0.95]
THRESHOLDS = np.round(np.linspace(0.05, 0.95, 181), 4)
BLEND_WEIGHTS = np.round(np.linspace(0.0, 1.0, 21), 4)


@dataclass
class ProbabilitySet:
    valid: np.ndarray
    test: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune TabPFN binary thresholds and fuse with XGB/SCM.")
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tabpfn-dir", type=Path, default=TABPFN_DIR)
    parser.add_argument("--feature-policy", default="screening_no_post")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--valid-size", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fit-mode", default="fit_preprocessors")
    parser.add_argument(
        "--tabpfn-variant",
        choices=["tabpfn3_binary", "tabpfn3_binary_balanced", "tabpfn3_default_binary_balanced"],
        default="tabpfn3_default_binary_balanced",
    )
    return parser.parse_args()


def positive_proba(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        return np.clip(arr, 1e-8, 1.0 - 1e-8)
    return np.clip(arr[:, 1], 1e-8, 1.0 - 1e-8)


def expected_calibration_error(y_true: pd.Series, pos_prob: np.ndarray, n_bins: int = 10) -> float:
    y_arr = y_true.to_numpy(dtype=int)
    p = np.clip(np.asarray(pos_prob, dtype=float), 1e-8, 1.0 - 1e-8)
    pred = (p >= 0.5).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correctness = (pred == y_arr).astype(float)
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(0.1, 1, n_bins)):
        mask = (confidence >= left) & (confidence <= right if right == 1 else confidence < right)
        if np.any(mask):
            ece += abs(float(correctness[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return float(ece)


def binary_metrics(y_true: pd.Series, pos_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_arr = y_true.to_numpy(dtype=int)
    p = np.clip(np.asarray(pos_prob, dtype=float), 1e-8, 1.0 - 1e-8)
    pred = (p >= threshold).astype(int)
    recalls = recall_score(y_arr, pred, labels=[0, 1], average=None, zero_division=0)
    try:
        auc = float(roc_auc_score(y_arr, p))
    except ValueError:
        auc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_arr, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, pred)),
        "macro_f1": float(f1_score(y_arr, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_arr, pred, average="weighted", zero_division=0)),
        "ovr_roc_auc_macro": auc,
        "ece": expected_calibration_error(y_true, p),
        "brier_score": float(brier_score_loss(y_arr, p)),
        "log_loss": float(log_loss(y_arr, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "specificity": float(recalls[0]),
        "sensitivity": float(recalls[1]),
        "class0_recall": float(recalls[0]),
        "class1_recall": float(recalls[1]),
    }


def threshold_candidates(y_valid: pd.Series, pos_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        rows.append(binary_metrics(y_valid, pos_prob, float(threshold)))
    return pd.DataFrame(rows)


def choose_threshold(y_valid: pd.Series, pos_prob: np.ndarray, objective: str) -> dict[str, Any]:
    candidates = threshold_candidates(y_valid, pos_prob)
    if objective == "default_0p50":
        row = binary_metrics(y_valid, pos_prob, 0.5)
        row["objective_met"] = True
        row["objective"] = objective
        return row
    if objective == "max_balanced_accuracy":
        ordered = candidates.sort_values(["balanced_accuracy", "macro_f1", "specificity"], ascending=False)
        row = ordered.iloc[0].to_dict()
        row["objective_met"] = True
        row["objective"] = objective
        return row
    if objective.startswith("sensitivity_ge_"):
        target = float(objective.removeprefix("sensitivity_ge_"))
        feasible = candidates[candidates["sensitivity"] >= target].copy()
        if feasible.empty:
            ordered = candidates.sort_values(["sensitivity", "specificity", "balanced_accuracy"], ascending=False)
            row = ordered.iloc[0].to_dict()
            row["objective_met"] = False
        else:
            ordered = feasible.sort_values(["specificity", "balanced_accuracy", "macro_f1"], ascending=False)
            row = ordered.iloc[0].to_dict()
            row["objective_met"] = True
        row["objective"] = objective
        return row
    raise ValueError(f"Unknown objective: {objective}")


def fit_xgb_scm_probability(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    seed: int,
) -> ProbabilitySet:
    bundle, augmented_size = fit_bundle_by_name(
        "xgb_binary_scm_v2",
        X_fit,
        y_fit,
        task="binary",
        seed=seed,
    )
    valid = positive_proba(predict_bundle(bundle, X_valid))
    test = positive_proba(predict_bundle(bundle, X_test))
    return ProbabilitySet(
        valid=valid,
        test=test,
        metadata={
            "train_size": bundle.train_size,
            "resampled_train_size": bundle.resampled_train_size,
            "augmented_size": augmented_size,
        },
    )


def fit_tabpfn_probability(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    seed: int,
    checkpoint_paths: dict[str, str],
    n_estimators: int,
    device: str,
    fit_mode: str,
    variant_name: str,
) -> ProbabilitySet:
    X_eval = pd.concat([X_valid, X_test], axis=0, ignore_index=True)
    proba, metadata = fit_predict_tabpfn(
        variant_name=variant_name,
        X_train_raw=X_fit,
        y_train=y_fit,
        X_eval_raw=X_eval,
        seed=seed,
        checkpoint_paths=checkpoint_paths,
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
    )
    p = positive_proba(proba)
    valid = p[: len(X_valid)]
    test = p[len(X_valid) :]
    return ProbabilitySet(
        valid=valid,
        test=test,
        metadata={
            "train_size": metadata["train_size"],
            "resampled_train_size": metadata["resampled_train_size"],
            "augmented_size": 0,
        },
    )


def fit_sigmoid_calibrator(y_valid: pd.Series, pos_prob: np.ndarray) -> LogisticRegression:
    p = np.clip(pos_prob, 1e-8, 1.0 - 1e-8)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(logits, y_valid.to_numpy(dtype=int))
    return calibrator


def apply_sigmoid_calibrator(calibrator: LogisticRegression, pos_prob: np.ndarray) -> np.ndarray:
    p = np.clip(pos_prob, 1e-8, 1.0 - 1e-8)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    return np.clip(calibrator.predict_proba(logits)[:, 1], 1e-8, 1.0 - 1e-8)


def fit_isotonic_calibrator(y_valid: pd.Series, pos_prob: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(pos_prob, y_valid.to_numpy(dtype=int))
    return calibrator


def evaluate_source(
    *,
    rows: list[dict[str, Any]],
    seed: int,
    source_name: str,
    valid_y: pd.Series,
    test_y: pd.Series,
    valid_prob: np.ndarray,
    test_prob: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> None:
    objectives = ["default_0p50", "max_balanced_accuracy"] + [
        f"sensitivity_ge_{target:.2f}" for target in SENSITIVITY_TARGETS
    ]
    for objective in objectives:
        selected = choose_threshold(valid_y, valid_prob, objective)
        threshold = float(selected["threshold"])
        metrics = binary_metrics(test_y, test_prob, threshold)
        row = {
            "seed": seed,
            "source_name": source_name,
            "objective": objective,
            "threshold": threshold,
            "objective_met_on_valid": bool(selected["objective_met"]),
            "valid_balanced_accuracy": float(selected["balanced_accuracy"]),
            "valid_sensitivity": float(selected["sensitivity"]),
            "valid_specificity": float(selected["specificity"]),
            **metrics,
        }
        if extra:
            row.update(extra)
        rows.append(row)


def run_seed(
    *,
    prepared_X: pd.DataFrame,
    prepared_y: pd.Series,
    seed: int,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
) -> list[dict[str, Any]]:
    train_idx, test_idx = train_test_split(
        np.arange(len(prepared_y)),
        test_size=0.2,
        random_state=seed,
        stratify=prepared_y,
    )
    X_train_base = prepared_X.iloc[train_idx].reset_index(drop=True)
    X_test_base = prepared_X.iloc[test_idx].reset_index(drop=True)
    y_train_base = prepared_y.iloc[train_idx].reset_index(drop=True)
    y_test = task_target(prepared_y.iloc[test_idx].reset_index(drop=True), "binary")

    inner_idx, valid_idx = train_test_split(
        np.arange(len(y_train_base)),
        test_size=args.valid_size,
        random_state=seed + 7001,
        stratify=task_target(y_train_base, "binary"),
    )
    X_fit_base = X_train_base.iloc[inner_idx].reset_index(drop=True)
    X_valid_base = X_train_base.iloc[valid_idx].reset_index(drop=True)
    y_fit = task_target(y_train_base.iloc[inner_idx].reset_index(drop=True), "binary")
    y_valid = task_target(y_train_base.iloc[valid_idx].reset_index(drop=True), "binary")

    split_view = apply_feature_policy(X_fit_base, X_valid_base, args.feature_policy)
    test_view = apply_feature_policy(X_fit_base, X_test_base, args.feature_policy)
    X_fit = split_view.X_train
    X_valid = split_view.X_eval
    X_test = test_view.X_eval

    xgb = fit_xgb_scm_probability(X_fit, y_fit, X_valid, X_test, seed + 101)
    tab = fit_tabpfn_probability(
        X_fit,
        y_fit,
        X_valid,
        X_test,
        seed=seed + 211,
        checkpoint_paths=checkpoint_paths,
        n_estimators=args.n_estimators,
        device=args.device,
        fit_mode=args.fit_mode,
        variant_name=args.tabpfn_variant,
    )

    rows: list[dict[str, Any]] = []
    evaluate_source(
        rows=rows,
        seed=seed,
        source_name="xgb_binary_scm_v2_inner",
        valid_y=y_valid,
        test_y=y_test,
        valid_prob=xgb.valid,
        test_prob=xgb.test,
        extra=xgb.metadata,
    )
    evaluate_source(
        rows=rows,
        seed=seed,
        source_name=f"{args.tabpfn_variant}_inner",
        valid_y=y_valid,
        test_y=y_test,
        valid_prob=tab.valid,
        test_prob=tab.test,
        extra=tab.metadata,
    )

    sigmoid = fit_sigmoid_calibrator(y_valid, tab.valid)
    tab_valid_sigmoid = apply_sigmoid_calibrator(sigmoid, tab.valid)
    tab_test_sigmoid = apply_sigmoid_calibrator(sigmoid, tab.test)
    evaluate_source(
        rows=rows,
        seed=seed,
        source_name=f"{args.tabpfn_variant}_sigmoid",
        valid_y=y_valid,
        test_y=y_test,
        valid_prob=tab_valid_sigmoid,
        test_prob=tab_test_sigmoid,
        extra={**tab.metadata, "calibration": "sigmoid"},
    )

    isotonic = fit_isotonic_calibrator(y_valid, tab.valid)
    tab_valid_iso = np.clip(isotonic.predict(tab.valid), 1e-8, 1.0 - 1e-8)
    tab_test_iso = np.clip(isotonic.predict(tab.test), 1e-8, 1.0 - 1e-8)
    evaluate_source(
        rows=rows,
        seed=seed,
        source_name=f"{args.tabpfn_variant}_isotonic",
        valid_y=y_valid,
        test_y=y_test,
        valid_prob=tab_valid_iso,
        test_prob=tab_test_iso,
        extra={**tab.metadata, "calibration": "isotonic"},
    )

    for tab_weight in BLEND_WEIGHTS:
        blend_valid = tab_weight * tab.valid + (1.0 - tab_weight) * xgb.valid
        blend_test = tab_weight * tab.test + (1.0 - tab_weight) * xgb.test
        source_name = f"blend_tabpfn_{tab_weight:.2f}_xgb_{1.0 - tab_weight:.2f}"
        evaluate_source(
            rows=rows,
            seed=seed,
            source_name=source_name,
            valid_y=y_valid,
            test_y=y_test,
            valid_prob=blend_valid,
            test_prob=blend_test,
            extra={
                "tabpfn_weight": float(tab_weight),
                "xgb_weight": float(1.0 - tab_weight),
                "train_size": int(len(y_fit)),
                "resampled_train_size": np.nan,
                "augmented_size": xgb.metadata.get("augmented_size", 0),
            },
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = validate_checkpoints(args.tabpfn_dir, [args.tabpfn_variant])
    if VARIANTS[args.tabpfn_variant].task != "binary":
        raise ValueError(f"Expected a binary TabPFN variant, got: {args.tabpfn_variant}")
    prepared = prepare_data(args.input)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"running seed={seed}", flush=True)
        seed_rows = run_seed(
            prepared_X=prepared.X,
            prepared_y=prepared.y,
            seed=seed,
            args=args,
            checkpoint_paths=checkpoint_paths,
        )
        rows.extend(seed_rows)
        pd.DataFrame(rows).to_csv(tables_dir / "binary_threshold_fusion_by_seed.partial.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "binary_threshold_fusion_by_seed.csv", index=False, encoding="utf-8-sig")
    summary = aggregate_metrics(metrics_df, ["source_name", "objective"], seed_base=71000)
    summary = summary.sort_values(
        ["objective", "balanced_accuracy_mean", "macro_f1_mean", "sensitivity_mean"],
        ascending=[True, False, False, False],
    )
    summary.to_csv(tables_dir / "binary_threshold_fusion_summary.csv", index=False, encoding="utf-8-sig")

    top_by_objective = (
        summary.sort_values(["objective", "balanced_accuracy_mean", "macro_f1_mean"], ascending=[True, False, False])
        .groupby("objective", as_index=False)
        .head(8)
    )
    top_by_objective.to_csv(tables_dir / "binary_threshold_fusion_top_by_objective.csv", index=False, encoding="utf-8-sig")
    run_summary = {
        "input_file": str(args.input),
        "output_dir": str(args.output_dir),
        "tabpfn_dir": str(args.tabpfn_dir),
        "feature_policy": args.feature_policy,
        "seeds": args.seeds,
        "valid_size_within_train": args.valid_size,
        "n_estimators": args.n_estimators,
        "device": args.device,
        "fit_mode": args.fit_mode,
        "tabpfn_variant": args.tabpfn_variant,
        "checkpoint_paths": checkpoint_paths,
        "tables": {
            "by_seed": "tables/binary_threshold_fusion_by_seed.csv",
            "summary": "tables/binary_threshold_fusion_summary.csv",
            "top_by_objective": "tables/binary_threshold_fusion_top_by_objective.csv",
        },
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"completed: {args.output_dir}", flush=True)
    print(top_by_objective.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
