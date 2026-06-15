from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, TARGET_COLUMN, prepare_data
from screening_0428_experiment import (
    apply_feature_policy,
    assert_screening_policy,
    collect_metrics,
    task_target,
    write_feature_policy_files,
)
import screening_constrained_0428_experiment as constrained
import screening_tuning_0428_experiment as tuning


warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "rerun_0428_outputs" / "screening_diagnostics_0428"
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]
SCREENING_POLICY = "screening_no_post"
THRESHOLD_POLICIES = ["sens090_maxspec", "sens093_maxspec", "sens095_maxspec"]
SENSITIVITY_TARGETS = {"sens090_maxspec": 0.90, "sens093_maxspec": 0.93, "sens095_maxspec": 0.95}

POST_TEST_COLUMNS = {"试验后醛固酮", "试验后肾素"}
REQUIRED_RETAINED_COLUMNS = [
    "试验前醛固酮",
    "试验前肾素",
    "确诊实验类型",
    "ARR比值>192为阳性，推荐进行确诊试验",
]
KEY_NUMERIC_FEATURES = [
    "年龄",
    "白细胞",
    "血红蛋白",
    "血小板",
    "钾",
    "钠",
    "氯",
    "肌酐",
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
    "醛固酮",
    "肾素",
    "ARR比值",
    "收缩压",
    "舒展压",
    "结节最大直径",
    "试验前醛固酮",
    "试验前肾素",
]


@dataclass(frozen=True)
class DiagnosticModel:
    model_id: str
    source: str
    task: str
    config: Any


class ProgressTracker:
    def __init__(self, output_dir: Path, total_steps: int) -> None:
        self.output_dir = output_dir
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0
        self.stage = "init"
        self.message = "initializing"
        self.context: dict[str, Any] = {}
        self.start_time = time.perf_counter()
        self.progress_file = output_dir / "progress.json"
        self.log_file = output_dir / "progress.log"
        self._write()

    def _snapshot(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.start_time
        return {
            "stage": self.stage,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "progress_percent": round(self.completed_steps / self.total_steps * 100, 2),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_human": format_seconds(elapsed),
            "current_message": self.message,
            "context": self.context,
        }

    def _write(self) -> None:
        self.progress_file.write_text(json.dumps(self._snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")

    def log(
        self,
        message: str,
        *,
        stage: str | None = None,
        advance: int = 0,
        context: dict[str, Any] | None = None,
    ) -> None:
        if stage is not None:
            self.stage = stage
        self.message = message
        if context is not None:
            self.context = context
        if advance:
            self.completed_steps = min(self.total_steps, self.completed_steps + advance)
        snapshot = self._snapshot()
        line = (
            f"[{snapshot['elapsed_human']}] {snapshot['completed_steps']}/{snapshot['total_steps']} "
            f"({snapshot['progress_percent']:.2f}%) | {self.stage} | {message}"
        )
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._write()

    def finish(self) -> None:
        self.completed_steps = self.total_steps
        self.stage = "complete"
        self.message = "complete"
        self._write()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARR 0428 screening diagnostics experiment.")
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def source_row_numbers(prepared: Any) -> pd.Series:
    # pandas index is the zero-based Excel data-row index; add 2 for header + one-based Excel row number.
    return pd.Series(prepared.cleaned_df.index.to_numpy(dtype=int) + 2, name="source_row_number").reset_index(drop=True)


def tuning_main_config(smoke: bool) -> tuning.ModelConfig:
    return tuning.ModelConfig(
        "xgb_bin_d3_l20_bal",
        "xgb",
        "binary",
        {
            "max_depth": 3,
            "min_child_weight": 10,
            "gamma": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "learning_rate": 0.03,
            "reg_alpha": 0.5,
            "reg_lambda": 20.0,
            "n_estimators": 80 if smoke else 220,
        },
        class_weight_policy="balanced",
    )


def constrained_specs(smoke: bool) -> dict[str, constrained.ModelSpec]:
    estimators = 50 if smoke else 160
    cat_iterations = 40 if smoke else 220
    return {
        "xgb_base_d3_l20": constrained.ModelSpec(
            "xgb_base_d3_l20",
            "xgb",
            "base",
            {
                "n_estimators": estimators,
                "max_depth": 3,
                "min_child_weight": 10,
                "gamma": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "learning_rate": 0.03,
                "reg_alpha": 0.5,
                "reg_lambda": 20.0,
            },
        ),
        "xgb_eng_d2_l10": constrained.ModelSpec(
            "xgb_eng_d2_l10",
            "xgb",
            "engineered",
            {
                "n_estimators": 50 if smoke else 131,
                "max_depth": 2,
                "min_child_weight": 5,
                "gamma": 0.0,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "learning_rate": 0.05,
                "reg_alpha": 0.0,
                "reg_lambda": 10.0,
            },
        ),
        "soft_vote_xgb_cat": constrained.ModelSpec(
            "soft_vote_xgb_cat",
            "soft_voting",
            "engineered",
            {},
            components=("xgb_d3_l20", "cat_d3_l50"),
        ),
        "three_catboost_d3_l50_bal": tuning.ModelConfig(
            "three_catboost_d3_l50_bal",
            "catboost",
            "three",
            {
                "iterations": cat_iterations,
                "depth": 3,
                "learning_rate": 0.03,
                "l2_leaf_reg": 50.0,
                "auto_class_weights": "Balanced",
            },
        ),
    }


def binary_models(smoke: bool) -> list[DiagnosticModel]:
    specs = constrained_specs(smoke)
    models = [
        DiagnosticModel("xgb_bin_d3_l20_bal", "tuning", "binary", tuning_main_config(smoke)),
        DiagnosticModel("xgb_eng_d2_l10", "constrained", "binary", specs["xgb_eng_d2_l10"]),
    ]
    if not smoke:
        models.extend(
            [
                DiagnosticModel("soft_vote_xgb_cat", "constrained", "binary", specs["soft_vote_xgb_cat"]),
                DiagnosticModel("xgb_base_d3_l20", "constrained", "binary", specs["xgb_base_d3_l20"]),
            ]
        )
    return models


def three_models(smoke: bool) -> list[DiagnosticModel]:
    if smoke:
        return []
    specs = constrained_specs(smoke)
    return [DiagnosticModel("three_catboost_d3_l50_bal", "tuning", "three", specs["three_catboost_d3_l50_bal"])]


def fit_binary_model(model: DiagnosticModel, X_fit: pd.DataFrame, y_fit: pd.Series, seed: int, smoke: bool) -> dict[str, Any]:
    if model.source == "tuning":
        return tuning.fit_model(X_fit, y_fit, model.config, seed)
    if model.source == "constrained":
        return constrained.fit_model(X_fit, y_fit, model.config, seed, smoke)
    raise ValueError(f"Unknown model source: {model.source}")


def predict_binary_model(model: DiagnosticModel, fitted: dict[str, Any], X_eval: pd.DataFrame) -> np.ndarray:
    if model.source == "tuning":
        return tuning.predict_model(fitted, X_eval, "binary")[:, 1]
    if model.source == "constrained":
        return constrained.predict_model(fitted, X_eval)[:, 1]
    raise ValueError(f"Unknown model source: {model.source}")


def classify_error(y_true: int, pred: int) -> str:
    if y_true == 1 and pred == 1:
        return "TP"
    if y_true == 1 and pred == 0:
        return "FN"
    if y_true == 0 and pred == 1:
        return "FP"
    return "TN"


def metrics_from_predictions(y_true: pd.Series, pred: pd.Series, prob_pos: pd.Series | np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.asarray(prob_pos, dtype=float)
    y_pred = pred.astype(int).to_numpy()
    recalls = recall_score(y, y_pred, labels=[0, 1], average=None, zero_division=0)
    precision = precision_score(y, y_pred, labels=[0, 1], average=None, zero_division=0)
    tn = int(((y == 0) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tp = int(((y == 1) & (y_pred == 1)).sum())
    try:
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    try:
        brier = float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except ValueError:
        brier = float("nan")
    confidence = np.where(y_pred == 1, p, 1.0 - p)
    correctness = (y_pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for idx in range(10):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == 9 else confidence < right)
        if mask.any():
            ece += abs(float(correctness[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "sensitivity": float(recalls[1]),
        "specificity": float(recalls[0]),
        "ppv": float(precision[1]),
        "npv": float(precision[0]),
        "auc": auc,
        "ece": float(ece),
        "brier_score": brier,
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fast_auc(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(prob, kind="mergesort")
    sorted_prob = prob[order]
    ranks = np.empty(len(prob), dtype=float)
    start = 0
    while start < len(prob):
        end = start + 1
        while end < len(prob) and sorted_prob[end] == sorted_prob[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def fast_ece(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    confidence = np.where(pred == 1, prob, 1.0 - prob)
    correctness = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == n_bins - 1 else confidence < right)
        if mask.any():
            ece += abs(float(correctness[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return float(ece)


def metrics_from_counts(
    y: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    *,
    auc: float | None = None,
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    pos = tp + fn
    neg = tn + fp
    sensitivity = tp / pos if pos else 0.0
    specificity = tn / neg if neg else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    if auc is None:
        auc = fast_auc(y, prob)
    return {
        "accuracy": float((tp + tn) / len(y)) if len(y) else 0.0,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "macro_f1": float((f1_pos + f1_neg) / 2.0),
        "weighted_f1": float((f1_pos * pos + f1_neg * neg) / len(y)) if len(y) else 0.0,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "auc": float(auc),
        "ece": fast_ece(y, pred, prob),
        "brier_score": float(np.mean((prob - y) ** 2)) if len(y) else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fast_choose_threshold(y: np.ndarray, prob: np.ndarray, policy: str) -> tuple[float, dict[str, float]]:
    y = np.asarray(y, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    thresholds = np.linspace(0.02, 0.98, 193)
    order = np.argsort(prob)
    prob_sorted = prob[order]
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = int(cum_pos[-1]) if len(cum_pos) else 0
    total_neg = int(len(y) - total_pos)
    k = np.searchsorted(prob_sorted, thresholds, side="left")
    pos_before = np.where(k > 0, cum_pos[np.maximum(k - 1, 0)], 0)
    predicted_positive = len(y) - k
    tp = total_pos - pos_before
    fp = predicted_positive - tp
    fn = total_pos - tp
    tn = total_neg - fp
    sensitivity = np.divide(tp, total_pos, out=np.zeros_like(tp, dtype=float), where=total_pos != 0)
    specificity = np.divide(tn, total_neg, out=np.zeros_like(tn, dtype=float), where=total_neg != 0)
    accuracy = (tp + tn) / max(len(y), 1)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    macro_f1 = (f1_pos + f1_neg) / 2.0
    target = SENSITIVITY_TARGETS[policy]
    feasible = np.where(sensitivity >= target)[0]
    if feasible.size == 0:
        candidate_idx = np.arange(len(thresholds))
        sort_keys = (balanced_accuracy[candidate_idx], sensitivity[candidate_idx])
    else:
        candidate_idx = feasible
        sort_keys = (macro_f1[candidate_idx], balanced_accuracy[candidate_idx], accuracy[candidate_idx], specificity[candidate_idx])
    # lexsort uses the last key as primary, so specificity/maxspec is the dominant criterion.
    best_local = np.lexsort(sort_keys)[-1]
    best_idx = int(candidate_idx[best_local])
    threshold = float(thresholds[best_idx])
    pred = (prob >= threshold).astype(int)
    return threshold, metrics_from_counts(y, prob, pred, auc=fast_auc(y, prob))


def run_binary_oof(
    X: pd.DataFrame,
    y_three: pd.Series,
    source_rows: pd.Series,
    models: list[DiagnosticModel],
    *,
    seeds: list[int],
    folds: int,
    smoke: bool,
    tracker: ProgressTracker,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_binary = task_target(y_three, "binary").reset_index(drop=True)
    X = X.reset_index(drop=True)
    y_three = y_three.astype(int).reset_index(drop=True)
    source_rows = source_rows.reset_index(drop=True)
    metric_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y_binary), start=1):
            X_train_base = X.iloc[train_idx].reset_index(drop=True)
            X_valid_base = X.iloc[valid_idx].reset_index(drop=True)
            y_train_base = y_binary.iloc[train_idx].reset_index(drop=True)
            y_valid = y_binary.iloc[valid_idx].reset_index(drop=True)
            y_valid_three = y_three.iloc[valid_idx].reset_index(drop=True)
            valid_source = source_rows.iloc[valid_idx].reset_index(drop=True)

            fit_idx, cal_idx = train_test_split(
                np.arange(len(y_train_base)),
                test_size=0.25,
                random_state=seed + fold,
                stratify=y_train_base,
            )
            X_fit_base = X_train_base.iloc[fit_idx].reset_index(drop=True)
            X_cal_base = X_train_base.iloc[cal_idx].reset_index(drop=True)
            y_fit = y_train_base.iloc[fit_idx].reset_index(drop=True)
            y_cal = y_train_base.iloc[cal_idx].reset_index(drop=True)

            fit_view = apply_feature_policy(X_fit_base, X_valid_base, SCREENING_POLICY)
            cal_view = apply_feature_policy(X_fit_base, X_cal_base, SCREENING_POLICY)

            for diag_model in models:
                context = {"seed": seed, "fold": fold, "model_id": diag_model.model_id}
                tracker.log(
                    f"binary seed={seed} fold={fold}/{folds} {diag_model.model_id}",
                    stage="binary_oof",
                    context=context,
                )
                fitted = fit_binary_model(diag_model, fit_view.X_train, y_fit, seed + fold, smoke)
                cal_raw = predict_binary_model(diag_model, fitted, cal_view.X_eval)
                valid_raw = predict_binary_model(diag_model, fitted, fit_view.X_eval)
                cal_prob, valid_prob = constrained.calibrated_probabilities(y_cal, cal_raw, valid_raw, "sigmoid")

                for policy in THRESHOLD_POLICIES:
                    threshold, cal_metrics = constrained.choose_threshold(y_cal, cal_prob, policy)
                    valid_metrics = constrained.binary_metrics(y_valid, valid_prob, threshold)
                    metric_rows.append(
                        {
                            "task": "binary",
                            "feature_policy": SCREENING_POLICY,
                            "seed": seed,
                            "fold": fold,
                            "folds": folds,
                            "model_id": diag_model.model_id,
                            "model_source": diag_model.source,
                            "calibration": "sigmoid",
                            "threshold_policy": policy,
                            "threshold": float(threshold),
                            "target_sensitivity": SENSITIVITY_TARGETS[policy],
                            "cal_sensitivity": cal_metrics["sensitivity"],
                            "cal_specificity": cal_metrics["specificity"],
                            "fit_size": int(len(y_fit)),
                            "cal_size": int(len(y_cal)),
                            "valid_size": int(len(y_valid)),
                            **valid_metrics,
                        }
                    )
                    pred = (valid_prob >= threshold).astype(int)
                    for pos in range(len(valid_idx)):
                        y_bin = int(y_valid.iloc[pos])
                        pred_bin = int(pred[pos])
                        oof_rows.append(
                            {
                                "task": "binary",
                                "feature_policy": SCREENING_POLICY,
                                "source_row_number": int(valid_source.iloc[pos]),
                                "seed": seed,
                                "fold": fold,
                                "folds": folds,
                                "model_id": diag_model.model_id,
                                "model_source": diag_model.source,
                                "calibration": "sigmoid",
                                "threshold_policy": policy,
                                "threshold": float(threshold),
                                "target_sensitivity": SENSITIVITY_TARGETS[policy],
                                "y_true_three": int(y_valid_three.iloc[pos]),
                                "y_true_binary": y_bin,
                                "prob_positive": float(valid_prob[pos]),
                                "pred_binary": pred_bin,
                                "error_type": classify_error(y_bin, pred_bin),
                            }
                        )
                write_df(tables_dir / "oof_predictions_long.partial.csv", pd.DataFrame(oof_rows))
                tracker.log(
                    f"done binary seed={seed} fold={fold}/{folds} {diag_model.model_id}",
                    stage="binary_oof_done",
                    advance=1,
                    context=context,
                )
    return pd.DataFrame(oof_rows), pd.DataFrame(metric_rows)


def run_three_oof(
    X: pd.DataFrame,
    y_three: pd.Series,
    source_rows: pd.Series,
    models: list[DiagnosticModel],
    *,
    seeds: list[int],
    folds: int,
    tracker: ProgressTracker,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not models:
        empty_pred = pd.DataFrame(
            columns=[
                "task",
                "source_row_number",
                "seed",
                "fold",
                "model_id",
                "y_true_three",
                "pred_three",
                "prob_0",
                "prob_1",
                "prob_2",
            ]
        )
        empty_metrics = pd.DataFrame(columns=["task", "seed", "fold", "model_id"])
        return empty_pred, empty_metrics

    X = X.reset_index(drop=True)
    y_three = y_three.astype(int).reset_index(drop=True)
    source_rows = source_rows.reset_index(drop=True)
    oof_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y_three), start=1):
            X_train_base = X.iloc[train_idx].reset_index(drop=True)
            X_valid_base = X.iloc[valid_idx].reset_index(drop=True)
            y_train = y_three.iloc[train_idx].reset_index(drop=True)
            y_valid = y_three.iloc[valid_idx].reset_index(drop=True)
            valid_source = source_rows.iloc[valid_idx].reset_index(drop=True)
            view = apply_feature_policy(X_train_base, X_valid_base, SCREENING_POLICY)
            for diag_model in models:
                context = {"seed": seed, "fold": fold, "model_id": diag_model.model_id}
                tracker.log(
                    f"three seed={seed} fold={fold}/{folds} {diag_model.model_id}",
                    stage="three_oof",
                    context=context,
                )
                fitted = tuning.fit_model(view.X_train, y_train, diag_model.config, seed + fold)
                valid_prob = tuning.predict_model(fitted, view.X_eval, "three")
                pred = np.argmax(valid_prob, axis=1)
                metric_rows.append(
                    {
                        "task": "three",
                        "feature_policy": SCREENING_POLICY,
                        "seed": seed,
                        "fold": fold,
                        "folds": folds,
                        "model_id": diag_model.model_id,
                        **collect_metrics(y_valid, valid_prob, "three"),
                    }
                )
                for pos in range(len(valid_idx)):
                    oof_rows.append(
                        {
                            "task": "three",
                            "feature_policy": SCREENING_POLICY,
                            "source_row_number": int(valid_source.iloc[pos]),
                            "seed": seed,
                            "fold": fold,
                            "folds": folds,
                            "model_id": diag_model.model_id,
                            "y_true_three": int(y_valid.iloc[pos]),
                            "pred_three": int(pred[pos]),
                            "prob_0": float(valid_prob[pos, 0]),
                            "prob_1": float(valid_prob[pos, 1]),
                            "prob_2": float(valid_prob[pos, 2]),
                        }
                    )
                write_df(tables_dir / "three_auxiliary_oof_predictions.partial.csv", pd.DataFrame(oof_rows))
                tracker.log(
                    f"done three seed={seed} fold={fold}/{folds} {diag_model.model_id}",
                    stage="three_oof_done",
                    advance=1,
                    context=context,
                )
    return pd.DataFrame(oof_rows), pd.DataFrame(metric_rows)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def qcut_labels(series: pd.Series, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    try:
        cut = pd.qcut(values, q=4, labels=[f"{prefix}_Q1", f"{prefix}_Q2", f"{prefix}_Q3", f"{prefix}_Q4"], duplicates="drop")
        return cut.astype("object").fillna("missing")
    except ValueError:
        return pd.Series(np.where(values.notna(), f"{prefix}_available", "missing"), index=series.index)


def sample_context(prepared: Any, source_rows: pd.Series) -> pd.DataFrame:
    X = prepared.X.reset_index(drop=True)
    y_three = prepared.y.astype(int).reset_index(drop=True)
    included = apply_feature_policy(X, X, SCREENING_POLICY).included_columns
    ctx = pd.DataFrame(
        {
            "source_row_number": source_rows,
            "y_true_three": y_three,
            "y_true_binary": task_target(y_three, "binary"),
            "row_missing_rate": X[included].isna().mean(axis=1),
        }
    )
    for col in [
        "性别",
        "年龄",
        "醛固酮",
        "肾素",
        "ARR比值",
        "钾",
        "收缩压",
        "舒展压",
        "是否有肾上腺结节",
        "是否有增生",
        "联合用药_总数",
        "确诊实验类型",
        "ARR比值>192为阳性，推荐进行确诊试验",
        "试验前醛固酮",
        "试验前肾素",
    ]:
        if col in X.columns:
            ctx[col] = X[col].reset_index(drop=True)

    arr = numeric_series(X, "ARR比值")
    ald = numeric_series(X, "醛固酮")
    renin = numeric_series(X, "肾素")
    potassium = numeric_series(X, "钾")
    sbp = numeric_series(X, "收缩压")
    dbp = numeric_series(X, "舒展压")
    age = numeric_series(X, "年龄")
    med = numeric_series(X, "联合用药_总数").fillna(0)

    ctx["strata_arr_quantile"] = qcut_labels(arr, "ARR")
    ctx["strata_aldosterone_quantile"] = qcut_labels(ald, "ALD")
    renin_q25 = float(renin.dropna().quantile(0.25)) if renin.notna().any() else np.nan
    ctx["strata_renin_low"] = np.where(renin.isna(), "missing", np.where(renin <= renin_q25, "renin_low_q1", "renin_not_low"))
    ctx["strata_low_k"] = np.where(potassium.isna(), "missing", np.where(potassium < 3.5, "K_lt_3p5", "K_ge_3p5"))
    ctx["strata_arr_gt_192"] = np.where(arr.isna(), "missing", np.where(arr > 192, "ARR_gt_192", "ARR_le_192"))
    if "ARR比值>192为阳性，推荐进行确诊试验" in X.columns:
        flag = pd.to_numeric(X["ARR比值>192为阳性，推荐进行确诊试验"], errors="coerce")
        ctx["strata_arr_rule_flag"] = np.where(flag.isna(), "missing", np.where(flag > 0, "rule_positive", "rule_negative"))
    else:
        ctx["strata_arr_rule_flag"] = ctx["strata_arr_gt_192"]
    ctx["strata_bp_group"] = np.select(
        [sbp.ge(160) | dbp.ge(100), sbp.ge(140) | dbp.ge(90)],
        ["bp_high", "bp_elevated"],
        default="bp_lower_or_missing",
    )
    ctx.loc[sbp.isna() & dbp.isna(), "strata_bp_group"] = "missing"
    ctx["strata_experiment_type"] = X["确诊实验类型"].astype("object").fillna("missing") if "确诊实验类型" in X.columns else "missing"
    ctx["strata_adrenal_nodule"] = np.where(
        numeric_series(X, "是否有肾上腺结节").isna(),
        "missing",
        np.where(numeric_series(X, "是否有肾上腺结节") > 0, "nodule_yes", "nodule_no"),
    )
    ctx["strata_hyperplasia"] = np.where(
        numeric_series(X, "是否有增生").isna(),
        "missing",
        np.where(numeric_series(X, "是否有增生") > 0, "hyperplasia_yes", "hyperplasia_no"),
    )
    ctx["strata_med_burden"] = np.select([med <= 0, med <= 2], ["med_none", "med_1_2"], default="med_ge_3")
    ctx["strata_age_group"] = np.select([age < 40, age < 60, age >= 60], ["age_lt_40", "age_40_59", "age_ge_60"], default="missing")
    ctx["strata_sex"] = X["性别"].astype("object").fillna("missing").map(lambda value: f"sex_{value}")
    ctx["strata_missing_bucket"] = pd.cut(
        ctx["row_missing_rate"],
        bins=[-0.001, 0.05, 0.20, 1.0],
        labels=["missing_low", "missing_mid", "missing_high"],
    ).astype("object")
    return ctx


def range_flag(series: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    flag = pd.Series(False, index=series.index)
    if lower is not None:
        flag |= values < lower
    if upper is not None:
        flag |= values > upper
    return flag.fillna(False)


def data_quality_tables(prepared: Any, source_rows: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = prepared.X.reset_index(drop=True)
    y_three = prepared.y.astype(int).reset_index(drop=True)
    included = apply_feature_policy(X, X, SCREENING_POLICY).included_columns
    X_policy = X[included].copy()
    feature_rows: list[dict[str, Any]] = []
    range_rules: dict[str, tuple[float | None, float | None]] = {
        "年龄": (18, 100),
        "白细胞": (0.1, 100),
        "血红蛋白": (30, 250),
        "血小板": (10, 1000),
        "钾": (2.0, 7.0),
        "钠": (120, 160),
        "氯": (80, 130),
        "肌酐": (10, 2000),
        "醛固酮": (0, None),
        "肾素": (0, None),
        "ARR比值": (0, None),
        "收缩压": (70, 260),
        "舒展压": (30, 160),
        "结节最大直径": (0, 200),
        "试验前醛固酮": (0, None),
        "试验前肾素": (0, None),
    }

    sample_flags = pd.DataFrame(
        {
            "source_row_number": source_rows,
            "y_true_three": y_three,
            "y_true_binary": task_target(y_three, "binary"),
            "row_missing_rate": X_policy.isna().mean(axis=1),
            "high_missing_rate_flag": X_policy.isna().mean(axis=1) >= 0.30,
            "duplicate_feature_row_flag": X_policy.duplicated(keep=False),
        }
    )
    out_of_range_count = pd.Series(0, index=X.index, dtype=int)
    for col in included:
        series = X[col]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric_like = numeric.notna().sum() >= max(int(series.notna().sum() * 0.8), 1)
        anomaly = pd.Series(False, index=X.index)
        if col in range_rules:
            anomaly = range_flag(series, *range_rules[col])
            out_of_range_count += anomaly.astype(int)
        row = {
            "feature": col,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "range_rule": str(range_rules.get(col, "")),
            "out_of_range_count": int(anomaly.sum()),
            "out_of_range_rate": float(anomaly.mean()),
        }
        if is_numeric_like:
            finite = numeric.dropna()
            row.update(
                {
                    "numeric_min": float(finite.min()) if not finite.empty else np.nan,
                    "numeric_p01": float(finite.quantile(0.01)) if not finite.empty else np.nan,
                    "numeric_median": float(finite.median()) if not finite.empty else np.nan,
                    "numeric_p99": float(finite.quantile(0.99)) if not finite.empty else np.nan,
                    "numeric_max": float(finite.max()) if not finite.empty else np.nan,
                }
            )
        feature_rows.append(row)

    arr = numeric_series(X, "ARR比值")
    ald = numeric_series(X, "醛固酮")
    renin = numeric_series(X, "肾素")
    valid_arr = arr.notna() & ald.notna() & renin.notna() & renin.abs().gt(1e-6)
    arr_calc = ald / renin
    arr_relative_error = (arr_calc - arr).abs() / arr.abs().clip(lower=1.0)
    sample_flags["arr_relative_error"] = np.where(valid_arr, arr_relative_error, np.nan)
    sample_flags["arr_consistency_error_flag"] = valid_arr & (arr_relative_error > 0.20)

    pre_ald = numeric_series(X, "试验前醛固酮")
    pre_renin = numeric_series(X, "试验前肾素")
    sample_flags["pre_arr_ratio"] = pre_ald / pre_renin.abs().replace(0, np.nan)
    sample_flags["pre_arr_ratio_invalid_flag"] = (
        (pre_ald.notna() & (pre_ald < 0))
        | (pre_renin.notna() & (pre_renin < 0))
        | (pre_ald.notna() & pre_renin.notna() & pre_renin.abs().le(1e-6))
    )

    sex = numeric_series(X, "性别")
    experiment = X["确诊实验类型"].astype("object") if "确诊实验类型" in X.columns else pd.Series(np.nan, index=X.index)
    sample_flags["invalid_sex_flag"] = sex.notna() & ~sex.isin([0, 1, 2])
    sample_flags["invalid_experiment_type_flag"] = experiment.notna() & ~experiment.isin(["冷盐水实验", "卡托普利实验"])
    sample_flags["out_of_range_count"] = out_of_range_count
    sample_flags["high_risk_quality_flag"] = (
        sample_flags["high_missing_rate_flag"]
        | sample_flags["duplicate_feature_row_flag"]
        | sample_flags["arr_consistency_error_flag"]
        | sample_flags["pre_arr_ratio_invalid_flag"]
        | sample_flags["invalid_sex_flag"]
        | sample_flags["invalid_experiment_type_flag"]
        | (sample_flags["out_of_range_count"] >= 2)
    )

    flag_cols = [col for col in sample_flags.columns if col.endswith("_flag")]
    label_rows: list[dict[str, Any]] = []
    for label_col in ["y_true_three", "y_true_binary"]:
        for label_value, group in sample_flags.groupby(label_col, dropna=False):
            row = {"label_type": label_col, "label_value": label_value, "n_samples": int(len(group))}
            for col in flag_cols:
                row[f"{col}_count"] = int(group[col].sum())
                row[f"{col}_rate"] = float(group[col].mean())
            label_rows.append(row)
    return pd.DataFrame(feature_rows), sample_flags, pd.DataFrame(label_rows)


def sample_error_stability(oof: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in oof.groupby(["model_id", "threshold_policy", "source_row_number"], dropna=False):
        model_id, policy, source_row = key
        counts = group["error_type"].value_counts()
        n = int(len(group))
        row = {
            "model_id": model_id,
            "threshold_policy": policy,
            "source_row_number": int(source_row),
            "n_predictions": n,
            "y_true_three": int(group["y_true_three"].iloc[0]),
            "y_true_binary": int(group["y_true_binary"].iloc[0]),
            "tp_count": int(counts.get("TP", 0)),
            "tn_count": int(counts.get("TN", 0)),
            "fp_count": int(counts.get("FP", 0)),
            "fn_count": int(counts.get("FN", 0)),
            "fp_rate": float(counts.get("FP", 0) / n),
            "fn_rate": float(counts.get("FN", 0) / n),
            "error_rate": float((counts.get("FP", 0) + counts.get("FN", 0)) / n),
            "mean_prob_positive": float(group["prob_positive"].mean()),
            "prob_std": float(group["prob_positive"].std(ddof=0)),
            "mean_threshold": float(group["threshold"].mean()),
        }
        row["hard_error_flag"] = bool(row["fp_rate"] >= 0.60 or row["fn_rate"] >= 0.60)
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.merge(ctx, on=["source_row_number", "y_true_three", "y_true_binary"], how="left")


def error_strata_summary(oof: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    merged = oof.merge(ctx, on=["source_row_number", "y_true_three", "y_true_binary"], how="left")
    strata_cols = [col for col in ctx.columns if col.startswith("strata_")]
    rows: list[dict[str, Any]] = []
    for strata_col in strata_cols:
        for key, group in merged.groupby(["model_id", "threshold_policy", strata_col], dropna=False):
            model_id, policy, stratum_value = key
            metrics = metrics_from_predictions(group["y_true_binary"], group["pred_binary"], group["prob_positive"])
            rows.append(
                {
                    "model_id": model_id,
                    "threshold_policy": policy,
                    "stratum_name": strata_col,
                    "stratum_value": str(stratum_value),
                    "n_predictions": int(len(group)),
                    "n_samples": int(group["source_row_number"].nunique()),
                    "positive_rate": float(group["y_true_binary"].mean()),
                    "fp_rate": float((group["error_type"] == "FP").mean()),
                    "fn_rate": float((group["error_type"] == "FN").mean()),
                    "mean_prob_positive": float(group["prob_positive"].mean()),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                }
            )
    return pd.DataFrame(rows)


def error_feature_contrast(stability: pd.DataFrame, prepared: Any) -> pd.DataFrame:
    X = prepared.X.reset_index(drop=True).copy()
    source_rows = pd.Series(prepared.cleaned_df.index.to_numpy(dtype=int) + 2, name="source_row_number").reset_index(drop=True)
    X.insert(0, "source_row_number", source_rows)
    rows: list[dict[str, Any]] = []
    numeric_features = [col for col in KEY_NUMERIC_FEATURES if col in X.columns]
    merged = stability.merge(X[["source_row_number", *numeric_features]], on="source_row_number", how="left")
    for key, group in merged.groupby(["model_id", "threshold_policy"], dropna=False):
        model_id, policy = key
        group = group.copy()
        group["error_group"] = np.select(
            [group["fn_rate"] >= 0.60, group["fp_rate"] >= 0.60, group["error_rate"] == 0],
            ["hard_fn", "hard_fp", "stable_correct"],
            default="mixed",
        )
        for feature in numeric_features:
            feature_col = feature
            if feature_col not in group.columns:
                if f"{feature}_x" in group.columns:
                    feature_col = f"{feature}_x"
                elif f"{feature}_y" in group.columns:
                    feature_col = f"{feature}_y"
                else:
                    continue
            values = pd.to_numeric(group[feature_col], errors="coerce")
            correct_mean = float(values[group["error_group"] == "stable_correct"].mean())
            for error_group, sub in group.groupby("error_group", dropna=False):
                sub_values = pd.to_numeric(sub[feature_col], errors="coerce")
                rows.append(
                    {
                        "model_id": model_id,
                        "threshold_policy": policy,
                        "feature": feature,
                        "error_group": error_group,
                        "n_samples": int(len(sub)),
                        "mean": float(sub_values.mean()) if sub_values.notna().any() else np.nan,
                        "median": float(sub_values.median()) if sub_values.notna().any() else np.nan,
                        "std": float(sub_values.std(ddof=0)) if sub_values.notna().any() else np.nan,
                        "missing_rate": float(sub_values.isna().mean()),
                        "stable_correct_mean": correct_mean,
                        "mean_minus_stable_correct": float(sub_values.mean() - correct_mean)
                        if sub_values.notna().any() and np.isfinite(correct_mean)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_thresholds(oof: pd.DataFrame, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        oof.sort_values(["model_id", "seed", "fold", "source_row_number"])
        .drop_duplicates(["model_id", "seed", "fold", "source_row_number"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for model_id, group in base.groupby("model_id", dropna=False):
        y = group["y_true_binary"].astype(int).reset_index(drop=True)
        p = group["prob_positive"].astype(float).reset_index(drop=True)
        n = len(group)
        y_np = y.to_numpy(dtype=int)
        p_np = p.to_numpy(dtype=float)
        for iteration in range(iterations):
            idx = rng.integers(0, n, size=n)
            y_boot = y_np[idx]
            p_boot = p_np[idx]
            for policy, target in SENSITIVITY_TARGETS.items():
                threshold, metrics = fast_choose_threshold(y_boot, p_boot, policy)
                rows.append(
                    {
                        "model_id": model_id,
                        "threshold_policy": policy,
                        "target_sensitivity": target,
                        "bootstrap_iteration": iteration + 1,
                        "threshold": float(threshold),
                        **metrics,
                    }
                )
    boot = pd.DataFrame(rows)
    recommendations: list[dict[str, Any]] = []
    for key, group in boot.groupby(["model_id", "threshold_policy", "target_sensitivity"], dropna=False):
        model_id, policy, target = key
        pred_group = base[base["model_id"] == model_id]
        y_full = pred_group["y_true_binary"].astype(int).reset_index(drop=True)
        p_full = pred_group["prob_positive"].astype(float).reset_index(drop=True)
        thresholds = group["threshold"].to_numpy(dtype=float)
        median_threshold = float(np.nanmedian(thresholds))
        sensitivity_priority_threshold = float(np.nanquantile(thresholds, 0.25))
        median_metrics = constrained.binary_metrics(y_full, p_full, median_threshold)
        priority_metrics = constrained.binary_metrics(y_full, p_full, sensitivity_priority_threshold)
        row = {
            "model_id": model_id,
            "threshold_policy": policy,
            "target_sensitivity": float(target),
            "bootstrap_iterations": int(len(group)),
            "threshold_median": median_threshold,
            "threshold_q25_sensitivity_priority": sensitivity_priority_threshold,
            "threshold_q75": float(np.nanquantile(thresholds, 0.75)),
            "threshold_ci_low": float(np.nanquantile(thresholds, 0.025)),
            "threshold_ci_high": float(np.nanquantile(thresholds, 0.975)),
            "target_met_probability": float((group["sensitivity"] >= target).mean()),
        }
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "auc", "ece", "brier_score"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_ci_low"] = float(values.quantile(0.025)) if not values.empty else np.nan
            row[f"{metric}_ci_high"] = float(values.quantile(0.975)) if not values.empty else np.nan
            row[f"median_threshold_{metric}"] = median_metrics.get(metric, np.nan)
            row[f"sensitivity_priority_{metric}"] = priority_metrics.get(metric, np.nan)
        recommendations.append(row)
    return boot, pd.DataFrame(recommendations)


def data_quality_sensitivity(oof: pd.DataFrame, sample_flags: pd.DataFrame) -> pd.DataFrame:
    merged = oof.merge(sample_flags[["source_row_number", "high_risk_quality_flag"]], on="source_row_number", how="left")
    rows: list[dict[str, Any]] = []
    for key, group in merged.groupby(["model_id", "threshold_policy"], dropna=False):
        model_id, policy = key
        for cohort, sub in [
            ("all_rows", group),
            ("exclude_high_risk_quality_flags", group[group["high_risk_quality_flag"] != True]),
            ("only_high_risk_quality_flags", group[group["high_risk_quality_flag"] == True]),
        ]:
            if sub.empty:
                rows.append({"model_id": model_id, "threshold_policy": policy, "cohort": cohort, "n_predictions": 0, "n_samples": 0})
                continue
            metrics = metrics_from_predictions(sub["y_true_binary"], sub["pred_binary"], sub["prob_positive"])
            rows.append(
                {
                    "model_id": model_id,
                    "threshold_policy": policy,
                    "cohort": cohort,
                    "n_predictions": int(len(sub)),
                    "n_samples": int(sub["source_row_number"].nunique()),
                    "high_risk_sample_rate": float((sub["high_risk_quality_flag"] == True).mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def plot_probability_by_error_type(oof: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    subset = oof[(oof["model_id"] == "xgb_bin_d3_l20_bal") & (oof["threshold_policy"] == "sens093_maxspec")].copy()
    if subset.empty:
        subset = oof.copy()
    order = ["TN", "FP", "FN", "TP"]
    data = [subset.loc[subset["error_type"] == item, "prob_positive"].dropna().to_numpy() for item in order]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel("Positive probability")
    plt.title("OOF probability by error type")
    plt.tight_layout()
    plt.savefig(figures_dir / "probability_by_error_type.png", dpi=200)
    plt.close()


def plot_threshold_bootstrap(boot: pd.DataFrame, figures_dir: Path) -> None:
    if boot.empty:
        return
    subset = boot[boot["threshold_policy"] == "sens093_maxspec"].copy()
    if subset.empty:
        subset = boot.copy()
    models = list(subset["model_id"].dropna().unique())[:4]
    plt.figure(figsize=(9, 5))
    for model_id in models:
        values = subset.loc[subset["model_id"] == model_id, "threshold"].dropna().to_numpy()
        if values.size:
            plt.hist(values, bins=25, alpha=0.45, label=model_id)
    plt.xlabel("Threshold")
    plt.ylabel("Bootstrap count")
    plt.title("Bootstrap threshold distribution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_bootstrap_distribution.png", dpi=200)
    plt.close()


def plot_error_strata_heatmap(strata: pd.DataFrame, figures_dir: Path) -> None:
    subset = strata[
        (strata["model_id"] == "xgb_bin_d3_l20_bal")
        & (strata["threshold_policy"] == "sens093_maxspec")
        & (strata["stratum_name"].isin(["strata_arr_quantile", "strata_low_k", "strata_bp_group", "strata_experiment_type"]))
    ].copy()
    if subset.empty:
        subset = strata.head(40).copy()
    subset["label"] = subset["stratum_name"].astype(str) + "=" + subset["stratum_value"].astype(str)
    subset = subset.sort_values("fp_rate", ascending=False).head(30)
    values = subset[["fp_rate", "fn_rate"]].to_numpy(dtype=float)
    plt.figure(figsize=(8, max(4, 0.25 * len(subset))))
    plt.imshow(values, aspect="auto", cmap="Reds", vmin=0, vmax=max(0.01, np.nanmax(values)))
    plt.yticks(range(len(subset)), subset["label"], fontsize=7)
    plt.xticks([0, 1], ["FP rate", "FN rate"])
    plt.colorbar(label="Rate")
    plt.title("Error strata heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "error_strata_heatmap.png", dpi=200)
    plt.close()


def model_config_table(binary: list[DiagnosticModel], three: list[DiagnosticModel]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in binary + three:
        config = model.config
        rows.append(
            {
                "model_id": model.model_id,
                "source": model.source,
                "task": model.task,
                "family": getattr(config, "model_family", getattr(config, "family", "")),
                "feature_set": getattr(config, "feature_set", "screening_no_post"),
                "class_weight_policy": getattr(config, "class_weight_policy", ""),
                "components": "|".join(getattr(config, "components", [])),
                "params": json.dumps(getattr(config, "params", {}), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / ("smoke" if args.smoke_test else "full")
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seeds[0] if args.seeds else 42] if args.smoke_test else list(dict.fromkeys(args.seeds))
    folds = 2 if args.smoke_test else args.folds
    bootstrap_iterations = 50 if args.smoke_test else args.bootstrap_iterations
    binary = binary_models(args.smoke_test)
    three = three_models(args.smoke_test)
    total_steps = len(seeds) * folds * (len(binary) + len(three)) + 9
    tracker = ProgressTracker(output_dir, total_steps=total_steps)
    tracker.log(
        "diagnostics experiment started",
        stage="startup",
        context={"seeds": seeds, "folds": folds, "bootstrap_iterations": bootstrap_iterations},
    )

    prepared = prepare_data(args.input)
    validation = assert_screening_policy(list(prepared.X.columns))
    write_feature_policy_files(output_dir, list(prepared.X.columns), [SCREENING_POLICY])
    source_rows = source_row_numbers(prepared)
    write_df(tables_dir / "model_config.csv", model_config_table(binary, three))
    tracker.log(
        "data loaded and feature policy validated",
        stage="data",
        advance=1,
        context={
            "shape": list(prepared.X.shape),
            "target_distribution": {str(k): int(v) for k, v in prepared.y.value_counts().sort_index().items()},
            "source_row_min": int(source_rows.min()),
            "source_row_max": int(source_rows.max()),
            **validation,
        },
    )

    feature_quality, sample_flags, quality_by_label = data_quality_tables(prepared, source_rows)
    write_df(tables_dir / "feature_quality_audit.csv", feature_quality)
    write_df(tables_dir / "sample_quality_flags.csv", sample_flags)
    write_df(tables_dir / "quality_flag_by_label.csv", quality_by_label)
    tracker.log("quality audit tables written", stage="quality", advance=1)

    ctx = sample_context(prepared, source_rows)
    oof, binary_metrics_by_fold = run_binary_oof(
        prepared.X,
        prepared.y,
        source_rows,
        binary,
        seeds=seeds,
        folds=folds,
        smoke=args.smoke_test,
        tracker=tracker,
        tables_dir=tables_dir,
    )
    write_df(tables_dir / "oof_predictions_long.csv", oof)
    write_df(tables_dir / "binary_metrics_by_fold.csv", binary_metrics_by_fold)
    tracker.log("binary OOF tables written", stage="binary_summary", advance=1)

    three_oof, three_metrics = run_three_oof(
        prepared.X,
        prepared.y,
        source_rows,
        three,
        seeds=seeds,
        folds=folds,
        tracker=tracker,
        tables_dir=tables_dir,
    )
    write_df(tables_dir / "three_auxiliary_oof_predictions.csv", three_oof)
    write_df(tables_dir / "three_auxiliary_metrics_by_fold.csv", three_metrics)
    tracker.log("three auxiliary tables written", stage="three_summary", advance=1)

    stability = sample_error_stability(oof, ctx)
    strata = error_strata_summary(oof, ctx)
    contrast = error_feature_contrast(stability, prepared)
    write_df(tables_dir / "sample_error_stability.csv", stability)
    write_df(tables_dir / "error_strata_summary.csv", strata)
    write_df(tables_dir / "error_feature_contrast.csv", contrast)
    tracker.log("error analysis tables written", stage="error_analysis", advance=1)

    boot, recommendations = bootstrap_thresholds(oof, bootstrap_iterations, seed=20260606)
    write_df(tables_dir / "threshold_bootstrap.csv", boot)
    write_df(tables_dir / "threshold_recommendations.csv", recommendations)
    tracker.log("threshold bootstrap tables written", stage="threshold_bootstrap", advance=1)

    quality_sensitivity = data_quality_sensitivity(oof, sample_flags)
    write_df(tables_dir / "data_quality_sensitivity.csv", quality_sensitivity)
    tracker.log("data quality sensitivity table written", stage="quality_sensitivity", advance=1)

    plot_probability_by_error_type(oof, figures_dir)
    plot_threshold_bootstrap(boot, figures_dir)
    plot_error_strata_heatmap(strata, figures_dir)
    tracker.log("figures written", stage="figures", advance=1)

    bad_feature_rows = [
        col
        for col in apply_feature_policy(prepared.X, prepared.X, SCREENING_POLICY).included_columns
        if col in POST_TEST_COLUMNS or "missingindicator" in str(col).lower() or str(col).startswith("Unnamed") or col == "住院号"
    ]
    retained = [col for col in REQUIRED_RETAINED_COLUMNS if col in apply_feature_policy(prepared.X, prepared.X, SCREENING_POLICY).included_columns]
    summary = {
        "input_file": str(args.input),
        "output_dir": str(output_dir),
        "smoke_test": bool(args.smoke_test),
        "feature_policy": SCREENING_POLICY,
        "seeds": seeds,
        "folds": folds,
        "bootstrap_iterations": bootstrap_iterations,
        "binary_models": [model.model_id for model in binary],
        "three_models": [model.model_id for model in three],
        "threshold_policies": THRESHOLD_POLICIES,
        "oof_prediction_rows": int(len(oof)),
        "sample_count": int(len(prepared.y)),
        "source_row_number_basis": "cleaned Excel source index + 2",
        "screening_policy_validation": validation,
        "forbidden_features_included": bad_feature_rows,
        "required_retained_present": retained,
        "outputs": {
            "oof_predictions_long": "tables/oof_predictions_long.csv",
            "sample_error_stability": "tables/sample_error_stability.csv",
            "error_strata_summary": "tables/error_strata_summary.csv",
            "error_feature_contrast": "tables/error_feature_contrast.csv",
            "threshold_bootstrap": "tables/threshold_bootstrap.csv",
            "threshold_recommendations": "tables/threshold_recommendations.csv",
            "feature_quality_audit": "tables/feature_quality_audit.csv",
            "sample_quality_flags": "tables/sample_quality_flags.csv",
            "quality_flag_by_label": "tables/quality_flag_by_label.csv",
            "data_quality_sensitivity": "tables/data_quality_sensitivity.csv",
            "probability_by_error_type": "figures/probability_by_error_type.png",
            "threshold_bootstrap_distribution": "figures/threshold_bootstrap_distribution.png",
            "error_strata_heatmap": "figures/error_strata_heatmap.png",
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.log("experiment summary written", stage="summary", advance=1)
    tracker.finish()


if __name__ == "__main__":
    main()
