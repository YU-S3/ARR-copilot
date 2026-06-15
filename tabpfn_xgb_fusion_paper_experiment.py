from __future__ import annotations

import argparse
import json
import math
import time
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
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, prepare_data
from screening_0428_experiment import aggregate_metrics, fit_bundle_by_name, predict_bundle, task_target
from tabpfn_screening_no_post_experiment import TABPFN_DIR, fit_predict_tabpfn, validate_checkpoints


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "rerun_0428_outputs" / "tabpfn_screening_no_post" / "fusion_paper"
)
TASK = "binary"
FEATURE_POLICY = "screening_no_post_no_missing_indicators"
XGB_MODEL_NAME = "xgb_binary_scm_v2"
TABPFN_VARIANT = "tabpfn3_default_binary_balanced"
POST_TEST_COLUMNS = ["试验后醛固酮", "试验后肾素"]
MISSING_INDICATOR_PATTERNS = [
    "missingindicator",
    "missing_indicator",
    "_missing",
    "是否缺失",
    "缺失指示",
]
NON_MODEL_COLUMNS = ["住院号"]
GAP_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "mcc",
    "quadratic_kappa",
    "class0_recall",
    "class1_recall",
    "sensitivity",
    "specificity",
]
HIGHER_IS_BETTER_WEIGHTS = {
    "balanced_accuracy_mean": 0.20,
    "accuracy_mean": 0.10,
    "macro_f1_mean": 0.10,
    "ovr_roc_auc_macro_mean": 0.05,
    "class0_recall_mean": 0.15,
    "sensitivity_mean": 0.10,
}
LOWER_IS_BETTER_WEIGHTS = {
    "brier_score_mean": 0.08,
    "log_loss_mean": 0.04,
    "ece_mean": 0.03,
    "generalization_gap_balanced_accuracy": 0.10,
    "balanced_accuracy_std": 0.05,
}


@dataclass
class FeatureView:
    X_train: pd.DataFrame
    X_eval: pd.DataFrame
    included_columns: list[str]
    dropped_columns: list[str]


@dataclass
class ProbabilitySet:
    train: np.ndarray
    test: np.ndarray
    metadata: dict[str, Any]


@dataclass
class InnerOOF:
    xgb: np.ndarray
    tabpfn: np.ndarray
    y: pd.Series
    rows: list[dict[str, Any]]


@dataclass
class SourceProbabilities:
    source_name: str
    candidate_family: str
    oof: np.ndarray
    train: np.ndarray
    test: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested no-post TabPFN + XGB binary fusion experiment for paper-grade screening analysis."
    )
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--tabpfn-dir", type=Path, default=TABPFN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--max-outer-folds", type=int, default=0)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fit-mode",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
        default="fit_preprocessors",
    )
    parser.add_argument("--skip-stacking", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def lower_name(column: str) -> str:
    return str(column).strip().lower()


def forbidden_reason(column: str) -> str | None:
    text = str(column)
    low = lower_name(text)
    if text in POST_TEST_COLUMNS:
        return "post_test"
    if any(pattern in low for pattern in MISSING_INDICATOR_PATTERNS[:3]):
        return "missing_indicator"
    if any(pattern in text for pattern in MISSING_INDICATOR_PATTERNS[3:]):
        return "missing_indicator"
    if text in NON_MODEL_COLUMNS:
        return "non_model_id"
    if text.startswith("Unnamed"):
        return "unnamed"
    return None


def forbidden_columns(columns: list[str] | pd.Index) -> list[str]:
    return [str(col) for col in columns if forbidden_reason(str(col)) is not None]


def assert_no_forbidden_columns(X: pd.DataFrame, stage: str) -> None:
    bad = forbidden_columns(X.columns)
    if bad:
        raise AssertionError(f"{stage} contains forbidden columns: {bad}")


def apply_strict_screening_policy(X_train: pd.DataFrame, X_eval: pd.DataFrame) -> FeatureView:
    dropped = [str(col) for col in X_train.columns if forbidden_reason(str(col)) is not None]
    included = [str(col) for col in X_train.columns if str(col) not in dropped]
    X_train_view = X_train[included].copy().reset_index(drop=True)
    X_eval_view = X_eval[included].copy().reset_index(drop=True)
    assert_no_forbidden_columns(X_train_view, "train feature matrix")
    assert_no_forbidden_columns(X_eval_view, "eval feature matrix")
    return FeatureView(
        X_train=X_train_view,
        X_eval=X_eval_view,
        included_columns=included,
        dropped_columns=dropped,
    )


def feature_audit_rows(columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matched_missing_patterns = {pattern: False for pattern in MISSING_INDICATOR_PATTERNS}
    for col in columns:
        reason = forbidden_reason(col)
        if reason == "missing_indicator":
            for pattern in MISSING_INDICATOR_PATTERNS:
                if pattern in lower_name(col) or pattern in col:
                    matched_missing_patterns[pattern] = True
        rows.append(
            {
                "feature_policy": FEATURE_POLICY,
                "column": col,
                "status": "dropped" if reason else "included",
                "reason": reason or "model_input",
            }
        )
    for expected in POST_TEST_COLUMNS:
        if expected not in columns:
            rows.append(
                {
                    "feature_policy": FEATURE_POLICY,
                    "column": expected,
                    "status": "absent_after_prepare_data",
                    "reason": "post_test",
                }
            )
    for pattern, matched in matched_missing_patterns.items():
        if not matched:
            rows.append(
                {
                    "feature_policy": FEATURE_POLICY,
                    "column": f"<pattern:{pattern}>",
                    "status": "absent_no_input",
                    "reason": "missing_indicator",
                }
            )
    return rows


def write_feature_policy_files(tables_dir: Path, columns: list[str]) -> dict[str, Any]:
    rows = feature_audit_rows(columns)
    pd.DataFrame(rows).to_csv(tables_dir / "feature_policy_columns.csv", index=False, encoding="utf-8-sig")
    included = [row["column"] for row in rows if row["status"] == "included"]
    actual_dropped = [row["column"] for row in rows if row["status"] == "dropped"]
    actual_forbidden = [col for col in columns if forbidden_reason(col) is not None]
    retained_forbidden = [col for col in included if forbidden_reason(col) is not None]
    if retained_forbidden:
        raise AssertionError(f"Forbidden columns retained by feature policy: {retained_forbidden}")
    return {
        "input_feature_count_before_policy": len(columns),
        "input_feature_count_after_policy": len([col for col in columns if forbidden_reason(col) is None]),
        "actual_forbidden_columns_detected": actual_forbidden,
        "dropped_columns": actual_dropped,
        "retained_forbidden_columns": retained_forbidden,
        "forbidden_check_passed": len(retained_forbidden) == 0,
    }


def positive_proba(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        p = arr
    else:
        p = arr[:, 1]
    return np.clip(p, 1e-8, 1.0 - 1e-8)


def two_column_proba(pos_prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(pos_prob, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.column_stack([1.0 - p, p])


def logit_prob(pos_prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(pos_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def calibration_ece(y_true: pd.Series, pos_prob: np.ndarray, n_bins: int = 10) -> float:
    y_arr = y_true.to_numpy(dtype=int)
    p = np.clip(np.asarray(pos_prob, dtype=float), 1e-8, 1.0 - 1e-8)
    pred = (p >= 0.5).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correctness = (pred == y_arr).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == n_bins - 1 else confidence < right)
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
    try:
        ll = float(log_loss(y_arr, two_column_proba(p), labels=[0, 1]))
    except ValueError:
        ll = float("nan")
    tn = int(np.sum((y_arr == 0) & (pred == 0)))
    fp = int(np.sum((y_arr == 0) & (pred == 1)))
    fn = int(np.sum((y_arr == 1) & (pred == 0)))
    tp = int(np.sum((y_arr == 1) & (pred == 1)))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_arr, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, pred)),
        "macro_f1": float(f1_score(y_arr, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_arr, pred, average="weighted", zero_division=0)),
        "ovr_roc_auc_macro": auc,
        "ece": calibration_ece(y_true, p),
        "brier_score": float(brier_score_loss(y_arr, p)),
        "log_loss": ll,
        "mcc": float(matthews_corrcoef(y_arr, pred)),
        "quadratic_kappa": float(cohen_kappa_score(y_arr, pred, weights="quadratic")),
        "top2_accuracy": 1.0,
        "class0_recall": float(recalls[0]),
        "class1_recall": float(recalls[1]),
        "class2_recall": float("nan"),
        "sensitivity": float(recalls[1]),
        "specificity": float(recalls[0]),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "predicted_positive_rate": float(pred.mean()),
        "positive_prob_mean": float(p.mean()),
    }


def threshold_grid() -> np.ndarray:
    return np.round(np.linspace(0.05, 0.95, 181), 4)


def selection_score(metrics: dict[str, float]) -> float:
    values = {
        "balanced_accuracy": 0.35,
        "macro_f1": 0.20,
        "class0_recall": 0.20,
        "sensitivity": 0.15,
        "accuracy": 0.10,
    }
    score = 0.0
    for key, weight in values.items():
        value = metrics.get(key, float("nan"))
        if math.isfinite(value):
            score += weight * value
    return float(score)


def choose_threshold(y_valid: pd.Series, pos_prob: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in threshold_grid():
        metrics = binary_metrics(y_valid, pos_prob, float(threshold))
        metrics["selection_score"] = selection_score(metrics)
        if best is None:
            best = metrics
            continue
        if (
            metrics["selection_score"],
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
            metrics["class0_recall"],
        ) > (
            best["selection_score"],
            best["balanced_accuracy"],
            best["macro_f1"],
            best["class0_recall"],
        ):
            best = metrics
    assert best is not None
    return best


def blend_weights(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--blend-step must be in (0, 1].")
    values = list(np.arange(0.0, 1.0 + step / 2.0, step))
    values = sorted({round(min(max(float(value), 0.0), 1.0), 4) for value in values})
    if values[0] != 0.0:
        values.insert(0, 0.0)
    if values[-1] != 1.0:
        values.append(1.0)
    return values


def fit_xgb_probability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    bundle, augmented_size = fit_bundle_by_name(
        XGB_MODEL_NAME,
        X_train,
        y_train,
        task=TASK,
        seed=seed,
    )
    proba = positive_proba(predict_bundle(bundle, X_eval))
    metadata = {
        "train_size": int(bundle.train_size),
        "resampled_train_size": int(bundle.resampled_train_size),
        "augmented_size": int(augmented_size),
    }
    return proba, metadata


def fit_tabpfn_probability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    *,
    seed: int,
    checkpoint_paths: dict[str, str],
    n_estimators: int,
    device: str,
    fit_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    proba, metadata = fit_predict_tabpfn(
        variant_name=TABPFN_VARIANT,
        X_train_raw=X_train,
        y_train=y_train,
        X_eval_raw=X_eval,
        seed=seed,
        checkpoint_paths=checkpoint_paths,
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
    )
    return positive_proba(proba), {
        "train_size": int(metadata["train_size"]),
        "resampled_train_size": int(metadata["resampled_train_size"]),
        "augmented_size": 0,
        "feature_count": int(metadata["feature_count"]),
        "categorical_feature_count": int(metadata["categorical_feature_count"]),
        "checkpoint_path": metadata["checkpoint_path"],
    }


def run_inner_oof(
    X_outer_train_base: pd.DataFrame,
    y_outer_train_three: pd.Series,
    *,
    inner_splits: int,
    outer_fold_index: int,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
) -> InnerOOF:
    y_outer_binary = task_target(y_outer_train_three, TASK)
    xgb_oof = np.zeros(len(y_outer_binary), dtype=float)
    tab_oof = np.zeros(len(y_outer_binary), dtype=float)
    rows: list[dict[str, Any]] = []
    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=args.random_state + outer_fold_index * 1009,
    )
    for inner_index, (fit_idx, valid_idx) in enumerate(inner_cv.split(X_outer_train_base, y_outer_binary), start=1):
        X_fit_base = X_outer_train_base.iloc[fit_idx].reset_index(drop=True)
        X_valid_base = X_outer_train_base.iloc[valid_idx].reset_index(drop=True)
        y_fit = y_outer_binary.iloc[fit_idx].reset_index(drop=True)
        y_valid = y_outer_binary.iloc[valid_idx].reset_index(drop=True)
        view = apply_strict_screening_policy(X_fit_base, X_valid_base)
        seed_base = args.random_state + outer_fold_index * 10000 + inner_index * 101
        xgb_prob, xgb_meta = fit_xgb_probability(view.X_train, y_fit, view.X_eval, seed=seed_base + 11)
        tab_prob, tab_meta = fit_tabpfn_probability(
            view.X_train,
            y_fit,
            view.X_eval,
            seed=seed_base + 23,
            checkpoint_paths=checkpoint_paths,
            n_estimators=args.n_estimators,
            device=args.device,
            fit_mode=args.fit_mode,
        )
        xgb_oof[valid_idx] = xgb_prob
        tab_oof[valid_idx] = tab_prob
        for name, prob, meta in [
            (XGB_MODEL_NAME, xgb_prob, xgb_meta),
            (TABPFN_VARIANT, tab_prob, tab_meta),
        ]:
            m = binary_metrics(y_valid, prob, 0.5)
            rows.append(
                {
                    "outer_fold_index": outer_fold_index,
                    "inner_fold_index": inner_index,
                    "model_name": name,
                    "train_size": int(len(y_fit)),
                    "valid_size": int(len(y_valid)),
                    **meta,
                    **m,
                }
            )
    return InnerOOF(xgb=xgb_oof, tabpfn=tab_oof, y=y_outer_binary, rows=rows)


def fit_outer_probabilities(
    X_outer_train_base: pd.DataFrame,
    y_outer_train_three: pd.Series,
    X_outer_test_base: pd.DataFrame,
    *,
    outer_fold_index: int,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
) -> tuple[ProbabilitySet, ProbabilitySet, FeatureView]:
    y_outer_binary = task_target(y_outer_train_three, TASK)
    view = apply_strict_screening_policy(X_outer_train_base, X_outer_test_base)
    X_eval_both = pd.concat([view.X_train, view.X_eval], axis=0, ignore_index=True)
    seed_base = args.random_state + outer_fold_index * 20000
    xgb_both, xgb_meta = fit_xgb_probability(view.X_train, y_outer_binary, X_eval_both, seed=seed_base + 301)
    tab_both, tab_meta = fit_tabpfn_probability(
        view.X_train,
        y_outer_binary,
        X_eval_both,
        seed=seed_base + 401,
        checkpoint_paths=checkpoint_paths,
        n_estimators=args.n_estimators,
        device=args.device,
        fit_mode=args.fit_mode,
    )
    train_len = len(y_outer_binary)
    return (
        ProbabilitySet(train=xgb_both[:train_len], test=xgb_both[train_len:], metadata=xgb_meta),
        ProbabilitySet(train=tab_both[:train_len], test=tab_both[train_len:], metadata=tab_meta),
        view,
    )


def fit_sigmoid(y: pd.Series, pos_prob: np.ndarray) -> LogisticRegression | None:
    if y.nunique() < 2:
        return None
    model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    model.fit(logit_prob(pos_prob).reshape(-1, 1), y.to_numpy(dtype=int))
    return model


def apply_sigmoid(model: LogisticRegression | None, pos_prob: np.ndarray) -> np.ndarray:
    if model is None:
        return positive_proba(pos_prob)
    return positive_proba(model.predict_proba(logit_prob(pos_prob).reshape(-1, 1))[:, 1])


def fit_isotonic(y: pd.Series, pos_prob: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(y_min=1e-8, y_max=1.0 - 1e-8, out_of_bounds="clip")
    model.fit(np.asarray(pos_prob, dtype=float), y.to_numpy(dtype=int))
    return model


def apply_isotonic(model: IsotonicRegression, pos_prob: np.ndarray) -> np.ndarray:
    return positive_proba(model.predict(np.asarray(pos_prob, dtype=float)))


def stack_features(xgb_prob: np.ndarray, tab_prob: np.ndarray) -> np.ndarray:
    x = positive_proba(xgb_prob)
    t = positive_proba(tab_prob)
    return np.column_stack([logit_prob(x), logit_prob(t), np.abs(x - t)])


def fit_stacking_model(
    y_oof: pd.Series,
    xgb_oof: np.ndarray,
    tab_oof: np.ndarray,
    *,
    seed: int,
) -> tuple[LogisticRegression, float, pd.DataFrame]:
    X_meta = stack_features(xgb_oof, tab_oof)
    y_arr = y_oof.to_numpy(dtype=int)
    class_counts = np.bincount(y_arr, minlength=2)
    n_splits = int(min(5, class_counts.min()))
    candidate_cs = [0.1, 1.0, 10.0]
    rows: list[dict[str, Any]] = []
    if n_splits < 2:
        best_c = 1.0
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best_c = 1.0
        best_score = -np.inf
        for c_value in candidate_cs:
            fold_scores = []
            for train_idx, valid_idx in cv.split(X_meta, y_arr):
                model = LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                )
                model.fit(X_meta[train_idx], y_arr[train_idx])
                p_valid = positive_proba(model.predict_proba(X_meta[valid_idx])[:, 1])
                chosen = choose_threshold(pd.Series(y_arr[valid_idx]), p_valid)
                fold_scores.append(float(chosen["selection_score"]))
            mean_score = float(np.mean(fold_scores))
            rows.append({"C": c_value, "selection_score_mean": mean_score})
            if mean_score > best_score:
                best_score = mean_score
                best_c = c_value
    final_model = LogisticRegression(
        C=best_c,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )
    final_model.fit(X_meta, y_arr)
    return final_model, best_c, pd.DataFrame(rows)


def make_source_probabilities(
    *,
    y_oof: pd.Series,
    oof: InnerOOF,
    xgb_outer: ProbabilitySet,
    tab_outer: ProbabilitySet,
    blend_values: list[float],
    skip_stacking: bool,
    outer_fold_index: int,
    args: argparse.Namespace,
) -> tuple[list[SourceProbabilities], pd.DataFrame]:
    sources: list[SourceProbabilities] = [
        SourceProbabilities(
            source_name=XGB_MODEL_NAME,
            candidate_family="single_model",
            oof=oof.xgb,
            train=xgb_outer.train,
            test=xgb_outer.test,
            metadata=xgb_outer.metadata,
        ),
        SourceProbabilities(
            source_name=TABPFN_VARIANT,
            candidate_family="single_model",
            oof=oof.tabpfn,
            train=tab_outer.train,
            test=tab_outer.test,
            metadata=tab_outer.metadata,
        ),
    ]
    for tab_weight in blend_values:
        if tab_weight in {0.0, 1.0}:
            continue
        xgb_weight = 1.0 - tab_weight
        sources.append(
            SourceProbabilities(
                source_name=f"blend_raw_tabpfn_{tab_weight:.2f}_xgb_{xgb_weight:.2f}",
                candidate_family="fixed_probability_blend",
                oof=tab_weight * oof.tabpfn + xgb_weight * oof.xgb,
                train=tab_weight * tab_outer.train + xgb_weight * xgb_outer.train,
                test=tab_weight * tab_outer.test + xgb_weight * xgb_outer.test,
                metadata={
                    "tabpfn_weight": float(tab_weight),
                    "xgb_weight": float(xgb_weight),
                    "calibration": "none",
                },
            )
        )

    sigmoid_xgb = fit_sigmoid(y_oof, oof.xgb)
    sigmoid_tab = fit_sigmoid(y_oof, oof.tabpfn)
    isotonic_xgb = fit_isotonic(y_oof, oof.xgb)
    isotonic_tab = fit_isotonic(y_oof, oof.tabpfn)
    calibrated = {
        "sigmoid": (
            apply_sigmoid(sigmoid_xgb, oof.xgb),
            apply_sigmoid(sigmoid_tab, oof.tabpfn),
            apply_sigmoid(sigmoid_xgb, xgb_outer.train),
            apply_sigmoid(sigmoid_tab, tab_outer.train),
            apply_sigmoid(sigmoid_xgb, xgb_outer.test),
            apply_sigmoid(sigmoid_tab, tab_outer.test),
        ),
        "isotonic": (
            apply_isotonic(isotonic_xgb, oof.xgb),
            apply_isotonic(isotonic_tab, oof.tabpfn),
            apply_isotonic(isotonic_xgb, xgb_outer.train),
            apply_isotonic(isotonic_tab, tab_outer.train),
            apply_isotonic(isotonic_xgb, xgb_outer.test),
            apply_isotonic(isotonic_tab, tab_outer.test),
        ),
    }
    for method, (xgb_oof, tab_oof, xgb_train, tab_train, xgb_test, tab_test) in calibrated.items():
        sources.append(
            SourceProbabilities(
                source_name=f"{XGB_MODEL_NAME}_{method}",
                candidate_family=f"{method}_calibrated_single",
                oof=xgb_oof,
                train=xgb_train,
                test=xgb_test,
                metadata={"calibration": method, "base_model": XGB_MODEL_NAME},
            )
        )
        sources.append(
            SourceProbabilities(
                source_name=f"{TABPFN_VARIANT}_{method}",
                candidate_family=f"{method}_calibrated_single",
                oof=tab_oof,
                train=tab_train,
                test=tab_test,
                metadata={"calibration": method, "base_model": TABPFN_VARIANT},
            )
        )
        for tab_weight in blend_values:
            if tab_weight in {0.0, 1.0}:
                continue
            xgb_weight = 1.0 - tab_weight
            sources.append(
                SourceProbabilities(
                    source_name=f"blend_{method}_tabpfn_{tab_weight:.2f}_xgb_{xgb_weight:.2f}",
                    candidate_family=f"{method}_calibrated_blend",
                    oof=tab_weight * tab_oof + xgb_weight * xgb_oof,
                    train=tab_weight * tab_train + xgb_weight * xgb_train,
                    test=tab_weight * tab_test + xgb_weight * xgb_test,
                    metadata={
                        "tabpfn_weight": float(tab_weight),
                        "xgb_weight": float(xgb_weight),
                        "calibration": method,
                    },
                )
            )

    stack_rows = pd.DataFrame()
    if not skip_stacking:
        stack_model, best_c, stack_rows = fit_stacking_model(
            y_oof,
            oof.xgb,
            oof.tabpfn,
            seed=args.random_state + outer_fold_index * 3001,
        )
        sources.append(
            SourceProbabilities(
                source_name=f"stack_logreg_c_{best_c:g}",
                candidate_family="stacking_logistic",
                oof=positive_proba(stack_model.predict_proba(stack_features(oof.xgb, oof.tabpfn))[:, 1]),
                train=positive_proba(
                    stack_model.predict_proba(stack_features(xgb_outer.train, tab_outer.train))[:, 1]
                ),
                test=positive_proba(
                    stack_model.predict_proba(stack_features(xgb_outer.test, tab_outer.test))[:, 1]
                ),
                metadata={"stacking_C": float(best_c), "calibration": "stacking_logistic"},
            )
        )
        if not stack_rows.empty:
            stack_rows.insert(0, "outer_fold_index", outer_fold_index)
    return sources, stack_rows


def evaluated_rows_for_source(
    source: SourceProbabilities,
    *,
    y_oof: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    outer_fold_index: int,
    repeat_index: int,
    split_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    threshold_specs = [
        ("default_0p50", {"threshold": 0.5, **binary_metrics(y_oof, source.oof, 0.5)}),
        ("oof_composite_threshold", choose_threshold(y_oof, source.oof)),
    ]
    for threshold_objective, threshold_info in threshold_specs:
        threshold = float(threshold_info["threshold"])
        oof_metrics = binary_metrics(y_oof, source.oof, threshold)
        oof_score = selection_score(oof_metrics)
        train_metrics = binary_metrics(y_train, source.train, threshold)
        test_metrics = binary_metrics(y_test, source.test, threshold)
        model_name = f"{source.source_name}|{threshold_objective}"
        row = {
            "task": TASK,
            "feature_policy": FEATURE_POLICY,
            "outer_fold_index": outer_fold_index,
            "repeat_index": repeat_index,
            "split_index": split_index,
            "model_name": model_name,
            "source_name": source.source_name,
            "candidate_family": source.candidate_family,
            "threshold_objective": threshold_objective,
            "threshold": threshold,
            "oof_selection_score": oof_score,
            "train_size": int(len(y_train)),
            "eval_size": int(len(y_test)),
            "split_type": "outer_test",
            **source.metadata,
            **test_metrics,
        }
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        rows.append(row)
        selection_rows.append(
            {
                "outer_fold_index": outer_fold_index,
                "repeat_index": repeat_index,
                "split_index": split_index,
                "model_name": model_name,
                "source_name": source.source_name,
                "candidate_family": source.candidate_family,
                "threshold_objective": threshold_objective,
                "threshold": threshold,
                "oof_selection_score": oof_score,
                **{f"oof_{key}": value for key, value in oof_metrics.items()},
            }
        )
    return rows, selection_rows


def add_oof_selected_row(
    rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    *,
    outer_fold_index: int,
) -> None:
    if not rows:
        return
    candidates = [row for row in rows if row["outer_fold_index"] == outer_fold_index]
    if not candidates:
        return
    selected = max(
        candidates,
        key=lambda row: (
            float(row.get("oof_selection_score", float("-inf"))),
            float(row.get("balanced_accuracy", float("-inf"))),
            float(row.get("macro_f1", float("-inf"))),
        ),
    )
    selected_copy = selected.copy()
    selected_copy["model_name"] = "oof_selected_composite"
    selected_copy["source_name"] = "oof_selected_composite"
    selected_copy["selected_source_name"] = selected["source_name"]
    selected_copy["selected_model_name"] = selected["model_name"]
    selected_copy["candidate_family"] = "oof_selected"
    rows.append(selected_copy)
    selection_rows.append(
        {
            "outer_fold_index": outer_fold_index,
            "model_name": "oof_selected_composite",
            "source_name": "oof_selected_composite",
            "candidate_family": "oof_selected",
            "threshold_objective": selected["threshold_objective"],
            "threshold": selected["threshold"],
            "oof_selection_score": selected["oof_selection_score"],
            "selected_source_name": selected["source_name"],
            "selected_model_name": selected["model_name"],
        }
    )


def build_gap_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics_df.empty:
        return pd.DataFrame()
    for _, group in metrics_df.groupby(["task", "feature_policy", "model_name"], dropna=False):
        first = group.iloc[0]
        for metric in GAP_METRICS:
            train_col = f"train_{metric}"
            if train_col not in group.columns or metric not in group.columns:
                continue
            train_values = pd.to_numeric(group[train_col], errors="coerce").to_numpy(dtype=float)
            test_values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(train_values) & np.isfinite(test_values)
            if not np.any(mask):
                continue
            gaps = test_values[mask] - train_values[mask] if metric == "log_loss" else train_values[mask] - test_values[mask]
            rows.append(
                {
                    "task": first["task"],
                    "feature_policy": first["feature_policy"],
                    "fold_count": int(group["outer_fold_index"].nunique()),
                    "model_name": first["model_name"],
                    "metric": metric,
                    "train_mean": float(train_values[mask].mean()),
                    "valid_mean": float(test_values[mask].mean()),
                    "generalization_gap_mean": float(gaps.mean()),
                    "generalization_gap_std": float(gaps.std(ddof=0)),
                    "overfit_risk_flag": bool(gaps.mean() > 0.05),
                }
            )
    return pd.DataFrame(rows)


def normalize_for_score(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    fill_value = finite.min() if higher_is_better else finite.max()
    numeric = numeric.fillna(fill_value)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    if higher_is_better:
        return (numeric - min_value) / (max_value - min_value)
    return (max_value - numeric) / (max_value - min_value)


def build_composite_ranking(summary: pd.DataFrame, gap_df: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    ranked = summary.copy()
    gap_lookup: dict[str, float] = {}
    if not gap_df.empty:
        ba_gap = gap_df[gap_df["metric"] == "balanced_accuracy"].copy()
        for _, row in ba_gap.iterrows():
            gap_lookup[str(row["model_name"])] = abs(float(row["generalization_gap_mean"]))
    ranked["generalization_gap_balanced_accuracy"] = [
        gap_lookup.get(str(name), float("nan")) for name in ranked["model_name"]
    ]
    ranked["composite_score"] = 0.0
    for metric, weight in HIGHER_IS_BETTER_WEIGHTS.items():
        if metric in ranked.columns:
            ranked[f"score_component_{metric}"] = normalize_for_score(ranked[metric], higher_is_better=True) * weight
            ranked["composite_score"] += ranked[f"score_component_{metric}"]
    for metric, weight in LOWER_IS_BETTER_WEIGHTS.items():
        if metric in ranked.columns:
            ranked[f"score_component_{metric}"] = normalize_for_score(ranked[metric], higher_is_better=False) * weight
            ranked["composite_score"] += ranked[f"score_component_{metric}"]
    ranked["composite_rank"] = ranked["composite_score"].rank(ascending=False, method="min").astype(int)
    return ranked.sort_values(
        [
            "composite_score",
            "balanced_accuracy_mean",
            "accuracy_mean",
            "class0_recall_mean",
            "generalization_gap_balanced_accuracy",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def run_outer_fold(
    prepared_X: pd.DataFrame,
    prepared_y: pd.Series,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    outer_fold_index: int,
    repeat_index: int,
    split_index: int,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
    blend_values: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    X_train_base = prepared_X.iloc[train_idx].reset_index(drop=True)
    X_test_base = prepared_X.iloc[test_idx].reset_index(drop=True)
    y_train_three = prepared_y.iloc[train_idx].reset_index(drop=True)
    y_test = task_target(prepared_y.iloc[test_idx].reset_index(drop=True), TASK)
    y_train = task_target(y_train_three, TASK)

    oof = run_inner_oof(
        X_train_base,
        y_train_three,
        inner_splits=args.inner_splits,
        outer_fold_index=outer_fold_index,
        args=args,
        checkpoint_paths=checkpoint_paths,
    )
    xgb_outer, tab_outer, _ = fit_outer_probabilities(
        X_train_base,
        y_train_three,
        X_test_base,
        outer_fold_index=outer_fold_index,
        args=args,
        checkpoint_paths=checkpoint_paths,
    )
    sources, stack_rows = make_source_probabilities(
        y_oof=oof.y,
        oof=oof,
        xgb_outer=xgb_outer,
        tab_outer=tab_outer,
        blend_values=blend_values,
        skip_stacking=args.skip_stacking,
        outer_fold_index=outer_fold_index,
        args=args,
    )
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for source in sources:
        source_rows, source_selection_rows = evaluated_rows_for_source(
            source,
            y_oof=oof.y,
            y_train=y_train,
            y_test=y_test,
            outer_fold_index=outer_fold_index,
            repeat_index=repeat_index,
            split_index=split_index,
        )
        rows.extend(source_rows)
        selection_rows.extend(source_selection_rows)
    add_oof_selected_row(rows, selection_rows, outer_fold_index=outer_fold_index)
    audit_rows = [
        {
            "outer_fold_index": outer_fold_index,
            "repeat_index": repeat_index,
            "split_index": split_index,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "train_class0": int((y_train == 0).sum()),
            "train_class1": int((y_train == 1).sum()),
            "test_class0": int((y_test == 0).sum()),
            "test_class1": int((y_test == 1).sum()),
            "inner_oof_rows": int(len(oof.rows)),
            "source_count": int(len(sources)),
            "candidate_row_count": int(len(rows)),
        }
    ]
    return rows, selection_rows, oof.rows, stack_rows


def save_partial(
    tables_dir: Path,
    metrics_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    inner_rows: list[dict[str, Any]],
    stack_frames: list[pd.DataFrame],
) -> None:
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(
            tables_dir / "fusion_metrics_by_outer_fold.partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(
            tables_dir / "fusion_oof_candidates_by_fold.partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if inner_rows:
        pd.DataFrame(inner_rows).to_csv(
            tables_dir / "fusion_inner_oof_base_metrics.partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if stack_frames:
        pd.concat(stack_frames, ignore_index=True).to_csv(
            tables_dir / "fusion_stacking_selection.partial.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = validate_checkpoints(args.tabpfn_dir, [TABPFN_VARIANT])
    prepared = prepare_data(args.input)
    feature_audit = write_feature_policy_files(tables_dir, list(prepared.X.columns))
    blend_values = blend_weights(args.blend_step)
    y_binary = task_target(prepared.y, TASK)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )
    max_outer = args.max_outer_folds if args.max_outer_folds and args.max_outer_folds > 0 else None

    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    stack_frames: list[pd.DataFrame] = []
    prediction_audit_rows: list[dict[str, Any]] = []

    for outer_fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(prepared.X, y_binary), start=1):
        if max_outer is not None and outer_fold_index > max_outer:
            break
        repeat_index = (outer_fold_index - 1) // args.outer_splits + 1
        split_index = (outer_fold_index - 1) % args.outer_splits + 1
        print(
            f"[{format_seconds(time.perf_counter() - start_time)}] "
            f"outer_fold={outer_fold_index} repeat={repeat_index} split={split_index}",
            flush=True,
        )
        fold_rows, fold_selection_rows, fold_inner_rows, stack_rows = run_outer_fold(
            prepared.X,
            prepared.y,
            train_idx=train_idx,
            test_idx=test_idx,
            outer_fold_index=outer_fold_index,
            repeat_index=repeat_index,
            split_index=split_index,
            args=args,
            checkpoint_paths=checkpoint_paths,
            blend_values=blend_values,
        )
        metrics_rows.extend(fold_rows)
        selection_rows.extend(fold_selection_rows)
        inner_rows.extend(fold_inner_rows)
        if not stack_rows.empty:
            stack_frames.append(stack_rows)
        selected = [row for row in fold_selection_rows if row.get("model_name") == "oof_selected_composite"]
        prediction_audit_rows.append(
            {
                "outer_fold_index": outer_fold_index,
                "repeat_index": repeat_index,
                "split_index": split_index,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "train_class0": int((y_binary.iloc[train_idx].reset_index(drop=True) == 0).sum()),
                "train_class1": int((y_binary.iloc[train_idx].reset_index(drop=True) == 1).sum()),
                "test_class0": int((y_binary.iloc[test_idx].reset_index(drop=True) == 0).sum()),
                "test_class1": int((y_binary.iloc[test_idx].reset_index(drop=True) == 1).sum()),
                "candidate_count": int(len(fold_rows)),
                "selected_model_name": selected[0].get("selected_model_name") if selected else None,
            }
        )
        save_partial(tables_dir, metrics_rows, selection_rows, inner_rows, stack_frames)

    metrics_df = pd.DataFrame(metrics_rows)
    selection_df = pd.DataFrame(selection_rows)
    inner_df = pd.DataFrame(inner_rows)
    prediction_audit_df = pd.DataFrame(prediction_audit_rows)
    metrics_df.to_csv(tables_dir / "fusion_metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(tables_dir / "fusion_oof_candidates_by_fold.csv", index=False, encoding="utf-8-sig")
    if not selection_df.empty:
        selected_df = selection_df[selection_df["model_name"] == "oof_selected_composite"].copy()
        selected_df.to_csv(tables_dir / "fusion_oof_selection_by_fold.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(tables_dir / "fusion_oof_selection_by_fold.csv", index=False, encoding="utf-8-sig")
    inner_df.to_csv(tables_dir / "fusion_inner_oof_base_metrics.csv", index=False, encoding="utf-8-sig")
    if stack_frames:
        pd.concat(stack_frames, ignore_index=True).to_csv(
            tables_dir / "fusion_stacking_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(tables_dir / "fusion_stacking_selection.csv", index=False, encoding="utf-8-sig")
    prediction_audit_df.to_csv(tables_dir / "fusion_prediction_audit.csv", index=False, encoding="utf-8-sig")

    summary = aggregate_metrics(
        metrics_df,
        ["task", "feature_policy", "model_name", "candidate_family", "threshold_objective"],
        seed_base=81000,
    )
    summary = summary.sort_values(
        ["balanced_accuracy_mean", "accuracy_mean", "macro_f1_mean", "class0_recall_mean"],
        ascending=[False, False, False, False],
    )
    summary.to_csv(tables_dir / "fusion_summary_mean_std.csv", index=False, encoding="utf-8-sig")
    gap_df = build_gap_table(metrics_df)
    gap_df.to_csv(tables_dir / "fusion_overfitting_indicators.csv", index=False, encoding="utf-8-sig")
    ranking = build_composite_ranking(summary, gap_df)
    ranking.to_csv(tables_dir / "fusion_composite_ranking.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "input_file": str(args.input),
        "output_dir": str(args.output_dir),
        "tabpfn_dir": str(args.tabpfn_dir),
        "checkpoint_paths": checkpoint_paths,
        "task": TASK,
        "feature_policy": FEATURE_POLICY,
        "xgb_model_name": XGB_MODEL_NAME,
        "tabpfn_variant": TABPFN_VARIANT,
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "max_outer_folds": args.max_outer_folds,
        "completed_outer_folds": int(metrics_df["outer_fold_index"].nunique()) if not metrics_df.empty else 0,
        "inner_splits": args.inner_splits,
        "n_estimators": args.n_estimators,
        "blend_step": args.blend_step,
        "blend_values": blend_values,
        "device": args.device,
        "fit_mode": args.fit_mode,
        "skip_stacking": bool(args.skip_stacking),
        "smoke_test": bool(args.smoke_test),
        "random_state": args.random_state,
        "feature_audit": feature_audit,
        "elapsed_seconds": round(time.perf_counter() - start_time, 2),
        "tables": {
            "metrics_by_outer_fold": "tables/fusion_metrics_by_outer_fold.csv",
            "oof_selection_by_fold": "tables/fusion_oof_selection_by_fold.csv",
            "oof_candidates_by_fold": "tables/fusion_oof_candidates_by_fold.csv",
            "summary_mean_std": "tables/fusion_summary_mean_std.csv",
            "composite_ranking": "tables/fusion_composite_ranking.csv",
            "overfitting_indicators": "tables/fusion_overfitting_indicators.csv",
            "prediction_audit": "tables/fusion_prediction_audit.csv",
            "feature_policy_columns": "tables/feature_policy_columns.csv",
        },
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"completed: {args.output_dir}", flush=True)
    if not ranking.empty:
        print(ranking.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
