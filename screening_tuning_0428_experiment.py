from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, prepare_data
from screening_0428_experiment import (
    apply_feature_policy,
    assert_screening_policy,
    augment_scm_v2,
    collect_metrics,
    fit_preprocessor_train_only,
    normalize_proba,
    task_target,
    transform_with_fitted,
    write_feature_policy_files,
)


warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "rerun_0428_outputs" / "screening_tuning_0428"
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]
SCREENING_POLICY = "screening_no_post"

BINARY_THRESHOLD_POLICIES = [
    "threshold_0_50",
    "maximize_balanced_accuracy",
    "maximize_macro_f1",
    "sensitivity_0_95_spec_max",
    "sensitivity_0_93_spec_max",
    "sensitivity_0_90_spec_max",
]
CALIBRATION_METHODS = ["uncalibrated", "sigmoid", "isotonic"]


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_family: str
    task: str
    params: dict[str, Any]
    class_weight_policy: str = "balanced"
    use_scm: bool = False
    use_ordinal: bool = False


class ProgressTracker:
    def __init__(self, output_dir: Path, total_steps: int) -> None:
        self.output_dir = output_dir
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0
        self.start_time = time.perf_counter()
        self.progress_file = output_dir / "progress.json"
        self.log_file = output_dir / "progress.log"
        self.stage = "init"
        self.message = "initializing"
        self.context: dict[str, Any] = {}
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
            f"[{snapshot['elapsed_human']}] "
            f"{snapshot['completed_steps']}/{snapshot['total_steps']} "
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
    parser = argparse.ArgumentParser(description="ARR 0428 screening tuning experiment.")
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--tuning-folds", type=int, default=5)
    parser.add_argument("--confirmation-folds", type=int, default=10)
    parser.add_argument("--top-binary", type=int, default=5)
    parser.add_argument("--top-three", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def xgb_params(base: dict[str, Any], *, task: str, seed: int) -> dict[str, Any]:
    params = {
        "n_estimators": base.get("n_estimators", 220),
        "max_depth": base.get("max_depth", 2),
        "learning_rate": base.get("learning_rate", 0.03),
        "subsample": base.get("subsample", 0.8),
        "colsample_bytree": base.get("colsample_bytree", 0.8),
        "min_child_weight": base.get("min_child_weight", 10),
        "gamma": base.get("gamma", 1.0),
        "reg_alpha": base.get("reg_alpha", 0.5),
        "reg_lambda": base.get("reg_lambda", 20.0),
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
    }
    if task == "binary":
        params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        params.update({"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"})
    return params


def binary_configs(smoke: bool) -> list[ModelConfig]:
    if smoke:
        return [
            ModelConfig(
                "xgb_bin_reg_smoke",
                "xgb",
                "binary",
                {
                    "max_depth": 2,
                    "min_child_weight": 10,
                    "gamma": 1.0,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "learning_rate": 0.03,
                    "reg_alpha": 0.5,
                    "reg_lambda": 20.0,
                    "n_estimators": 80,
                },
                class_weight_policy="balanced",
            ),
            ModelConfig(
                "xgb_bin_scm_smoke",
                "xgb",
                "binary",
                {
                    "max_depth": 2,
                    "min_child_weight": 10,
                    "gamma": 1.0,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "learning_rate": 0.03,
                    "reg_alpha": 0.5,
                    "reg_lambda": 20.0,
                    "n_estimators": 80,
                },
                class_weight_policy="balanced_x1_5_class0",
                use_scm=True,
            ),
        ]

    rows = [
        ("xgb_bin_d1_l20_c0x1_5", 1, 10, 1.0, 0.8, 0.8, 0.03, 0.5, 20.0, 220, "balanced_x1_5_class0", False),
        ("xgb_bin_d2_l20_c0x1_5", 2, 10, 1.0, 0.8, 0.8, 0.03, 0.5, 20.0, 220, "balanced_x1_5_class0", False),
        ("xgb_bin_d2_l50_c0x1_5", 2, 20, 1.0, 0.8, 0.8, 0.03, 1.0, 50.0, 220, "balanced_x1_5_class0", False),
        ("xgb_bin_d2_l20_c0x2", 2, 10, 1.0, 0.8, 0.8, 0.03, 0.5, 20.0, 220, "balanced_x2_class0", False),
        ("xgb_bin_d3_l20_bal", 3, 10, 1.0, 0.8, 0.8, 0.03, 0.5, 20.0, 220, "balanced", False),
        ("xgb_bin_d3_l50_c0x1_5", 3, 20, 2.0, 0.7, 0.8, 0.02, 1.0, 50.0, 260, "balanced_x1_5_class0", False),
        ("xgb_bin_d2_l10_bal", 2, 5, 0.0, 0.9, 0.9, 0.05, 0.0, 10.0, 180, "balanced", False),
        ("xgb_bin_d1_l50_c0x2", 1, 20, 2.0, 0.7, 0.7, 0.03, 2.0, 50.0, 240, "balanced_x2_class0", False),
        ("xgb_bin_none_d2_l20", 2, 10, 1.0, 0.8, 0.8, 0.03, 0.5, 20.0, 220, "none", False),
        ("xgb_bin_scm_d2_l20_c0x1_5", 2, 10, 1.0, 0.8, 0.8, 0.03, 0.5, 20.0, 200, "balanced_x1_5_class0", True),
        ("xgb_bin_scm_d2_l50_c0x1_5", 2, 20, 1.0, 0.8, 0.8, 0.03, 1.0, 50.0, 200, "balanced_x1_5_class0", True),
    ]
    configs: list[ModelConfig] = []
    for model_id, depth, child, gamma, subsample, colsample, lr, alpha, reg_lambda, estimators, weight, use_scm in rows:
        configs.append(
            ModelConfig(
                model_id=model_id,
                model_family="xgb_scm" if use_scm else "xgb",
                task="binary",
                params={
                    "max_depth": depth,
                    "min_child_weight": child,
                    "gamma": gamma,
                    "subsample": subsample,
                    "colsample_bytree": colsample,
                    "learning_rate": lr,
                    "reg_alpha": alpha,
                    "reg_lambda": reg_lambda,
                    "n_estimators": estimators,
                },
                class_weight_policy=weight,
                use_scm=use_scm,
            )
        )
    return configs


def three_configs(smoke: bool) -> list[ModelConfig]:
    if smoke:
        return [
            ModelConfig(
                "three_catboost_smoke",
                "catboost",
                "three",
                {"iterations": 80, "depth": 2, "learning_rate": 0.04, "l2_leaf_reg": 20.0, "auto_class_weights": "Balanced"},
            ),
            ModelConfig(
                "three_xgb_scm_smoke",
                "xgb_scm",
                "three",
                {
                    "max_depth": 2,
                    "min_child_weight": 10,
                    "gamma": 1.0,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "learning_rate": 0.03,
                    "reg_alpha": 0.5,
                    "reg_lambda": 20.0,
                    "n_estimators": 80,
                },
                use_scm=True,
            ),
        ]
    return [
        ModelConfig(
            "three_catboost_d2_l20_bal",
            "catboost",
            "three",
            {"iterations": 260, "depth": 2, "learning_rate": 0.035, "l2_leaf_reg": 20.0, "auto_class_weights": "Balanced"},
        ),
        ModelConfig(
            "three_catboost_d3_l50_bal",
            "catboost",
            "three",
            {"iterations": 260, "depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 50.0, "auto_class_weights": "Balanced"},
        ),
        ModelConfig(
            "three_catboost_d3_l50_sqrt",
            "catboost",
            "three",
            {"iterations": 260, "depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 50.0, "auto_class_weights": "SqrtBalanced"},
        ),
        ModelConfig(
            "three_xgb_scm_d2_l20",
            "xgb_scm",
            "three",
            {
                "max_depth": 2,
                "min_child_weight": 10,
                "gamma": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "learning_rate": 0.03,
                "reg_alpha": 0.5,
                "reg_lambda": 20.0,
                "n_estimators": 220,
            },
            class_weight_policy="balanced",
            use_scm=True,
        ),
        ModelConfig(
            "three_ordinal_scm_d2_l20",
            "ordinal_xgb_scm",
            "three",
            {
                "max_depth": 2,
                "min_child_weight": 10,
                "gamma": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "learning_rate": 0.03,
                "reg_alpha": 0.5,
                "reg_lambda": 20.0,
                "n_estimators": 220,
            },
            class_weight_policy="balanced",
            use_scm=True,
            use_ordinal=True,
        ),
    ]


def sample_weights(y: pd.Series, policy: str) -> np.ndarray | None:
    if policy == "none":
        return None
    weights = compute_sample_weight(class_weight="balanced", y=y)
    if policy == "balanced_x1_5_class0":
        weights = weights.astype(float)
        weights[y.to_numpy(dtype=int) == 0] *= 1.5
    elif policy == "balanced_x2_class0":
        weights = weights.astype(float)
        weights[y.to_numpy(dtype=int) == 0] *= 2.0
    return weights


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "", 1).replace("cat__", "", 1)


def fit_xgb_raw(
    X_fit_raw: pd.DataFrame,
    y_fit: pd.Series,
    config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    X_train = X_fit_raw.reset_index(drop=True)
    y_train = y_fit.astype(int).reset_index(drop=True)
    if config.use_scm:
        X_train, y_train, _ = augment_scm_v2(X_train, y_train, config.task, seed)
    preprocessor, X_fit_df = fit_preprocessor_train_only(X_train)
    model = XGBClassifier(**xgb_params(config.params, task=config.task, seed=seed))
    weights = sample_weights(y_train, config.class_weight_policy)
    model.fit(X_fit_df, y_train, sample_weight=weights, verbose=False)
    return {
        "kind": "xgb",
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": list(X_fit_df.columns),
        "fit_size": int(len(X_fit_raw)),
        "train_size": int(len(X_train)),
        "augmented_size": int(len(X_train) - len(X_fit_raw)),
    }


def fit_ordinal_xgb_raw(
    X_fit_raw: pd.DataFrame,
    y_fit_three: pd.Series,
    config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    X_train = X_fit_raw.reset_index(drop=True)
    y_order = y_fit_three.astype(int).replace({0: 0, 2: 1, 1: 2}).reset_index(drop=True)
    y_original = y_fit_three.astype(int).reset_index(drop=True)
    if config.use_scm:
        X_train, y_original, _ = augment_scm_v2(X_train, y_original, "three", seed)
        y_order = y_original.astype(int).replace({0: 0, 2: 1, 1: 2}).reset_index(drop=True)
    preprocessor, X_fit_df = fit_preprocessor_train_only(X_train)
    gt0 = (y_order > 0).astype(int)
    gt1 = (y_order > 1).astype(int)
    model_gt0 = XGBClassifier(**xgb_params(config.params, task="binary", seed=seed))
    model_gt1 = XGBClassifier(**xgb_params(config.params, task="binary", seed=seed + 17))
    model_gt0.fit(X_fit_df, gt0, sample_weight=sample_weights(gt0, config.class_weight_policy), verbose=False)
    model_gt1.fit(X_fit_df, gt1, sample_weight=sample_weights(gt1, config.class_weight_policy), verbose=False)
    return {
        "kind": "ordinal_xgb",
        "gt0": model_gt0,
        "gt1": model_gt1,
        "preprocessor": preprocessor,
        "feature_names": list(X_fit_df.columns),
        "fit_size": int(len(X_fit_raw)),
        "train_size": int(len(X_train)),
        "augmented_size": int(len(X_train) - len(X_fit_raw)),
    }


def fit_catboost_raw(
    X_fit_raw: pd.DataFrame,
    y_fit: pd.Series,
    config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    preprocessor, X_fit_df = fit_preprocessor_train_only(X_fit_raw.reset_index(drop=True))
    params = {
        "loss_function": "MultiClass" if config.task == "three" else "Logloss",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        **config.params,
    }
    model = CatBoostClassifier(**params)
    use_manual_weights = "auto_class_weights" not in config.params
    weights = sample_weights(y_fit.astype(int).reset_index(drop=True), config.class_weight_policy) if use_manual_weights else None
    model.fit(X_fit_df, y_fit.astype(int).reset_index(drop=True), sample_weight=weights)
    return {
        "kind": "catboost",
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": list(X_fit_df.columns),
        "fit_size": int(len(X_fit_raw)),
        "train_size": int(len(X_fit_raw)),
        "augmented_size": 0,
    }


def fit_model(
    X_fit_raw: pd.DataFrame,
    y_fit: pd.Series,
    config: ModelConfig,
    seed: int,
) -> dict[str, Any]:
    if config.model_family in {"xgb", "xgb_scm"}:
        return fit_xgb_raw(X_fit_raw, y_fit, config, seed)
    if config.model_family == "ordinal_xgb_scm":
        return fit_ordinal_xgb_raw(X_fit_raw, y_fit, config, seed)
    if config.model_family == "catboost":
        return fit_catboost_raw(X_fit_raw, y_fit, config, seed)
    raise ValueError(f"Unknown model family: {config.model_family}")


def predict_model(fitted: dict[str, Any], X_raw: pd.DataFrame, task: str) -> np.ndarray:
    X_df = transform_with_fitted(fitted["preprocessor"], X_raw.reset_index(drop=True))
    if fitted["kind"] == "ordinal_xgb":
        p_gt0 = fitted["gt0"].predict_proba(X_df)[:, 1]
        p_gt1 = fitted["gt1"].predict_proba(X_df)[:, 1]
        p_gt1 = np.minimum(p_gt1, p_gt0)
        p0 = 1.0 - p_gt0
        p2 = p_gt0 - p_gt1
        p1 = p_gt1
        return normalize_proba(np.column_stack([p0, p1, p2]), 3)
    return normalize_proba(fitted["model"].predict_proba(X_df), 2 if task == "binary" else 3)


def feature_importance_rows(fitted: dict[str, Any], config: ModelConfig, seed: int, stage: str) -> list[dict[str, Any]]:
    names = fitted.get("feature_names", [])
    values: np.ndarray | None = None
    if fitted["kind"] in {"xgb", "catboost"}:
        model = fitted["model"]
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "get_feature_importance"):
            values = np.asarray(model.get_feature_importance(), dtype=float)
    elif fitted["kind"] == "ordinal_xgb":
        values = (
            np.asarray(fitted["gt0"].feature_importances_, dtype=float)
            + np.asarray(fitted["gt1"].feature_importances_, dtype=float)
        ) / 2.0
    if values is None or len(values) != len(names):
        return []
    total = float(values.sum())
    if total > 0:
        values = values / total
    return [
        {
            "stage": stage,
            "task": config.task,
            "model_id": config.model_id,
            "seed": seed,
            "feature": clean_feature_name(name),
            "importance": float(value),
        }
        for name, value in zip(names, values)
    ]


def binary_ece(y_true: pd.Series, prob_pos: np.ndarray, n_bins: int = 10) -> float:
    y_arr = y_true.to_numpy(dtype=int)
    prob = np.clip(np.asarray(prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    confidence = np.where(pred == 1, prob, 1.0 - prob)
    correctness = (pred == y_arr).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == n_bins - 1 else confidence < right)
        if np.any(mask):
            ece += abs(float(correctness[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return float(ece)


def binary_metrics(y_true: pd.Series, prob_pos: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.astype(int).reset_index(drop=True)
    p = np.clip(np.asarray(prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (p >= threshold).astype(int)
    recalls = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    try:
        auc = float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "sensitivity": float(recalls[1]),
        "specificity": float(recalls[0]),
        "class0_recall": float(recalls[0]),
        "class1_recall": float(recalls[1]),
        "ovr_roc_auc_macro": auc,
        "auc": auc,
        "ece": binary_ece(y, p),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "mcc": float(matthews_corrcoef(y, pred)),
        "quadratic_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
    }


def choose_threshold(y_cal: pd.Series, prob_pos: np.ndarray, policy: str) -> tuple[float, dict[str, float]]:
    y = y_cal.astype(int).reset_index(drop=True)
    p = np.clip(np.asarray(prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    if policy == "threshold_0_50":
        threshold = 0.5
        return threshold, binary_metrics(y, p, threshold)
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), p]))
    scored: list[tuple[float, dict[str, float]]] = [(float(t), binary_metrics(y, p, float(t))) for t in candidates]
    if policy == "maximize_balanced_accuracy":
        return max(scored, key=lambda item: (item[1]["balanced_accuracy"], item[1]["macro_f1"], item[1]["specificity"]))
    if policy == "maximize_macro_f1":
        return max(scored, key=lambda item: (item[1]["macro_f1"], item[1]["balanced_accuracy"], item[1]["specificity"]))
    if policy.startswith("sensitivity_"):
        target = float(policy.split("_")[1] + "." + policy.split("_")[2])
        feasible = [item for item in scored if item[1]["sensitivity"] >= target]
        if not feasible:
            return max(scored, key=lambda item: (item[1]["sensitivity"], item[1]["balanced_accuracy"]))
        return max(feasible, key=lambda item: (item[1]["specificity"], item[1]["balanced_accuracy"], item[1]["macro_f1"]))
    raise ValueError(f"Unknown threshold policy: {policy}")


def calibrated_probabilities(
    y_cal: pd.Series,
    cal_prob_pos: np.ndarray,
    eval_prob_pos: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    cal_p = np.clip(np.asarray(cal_prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    eval_p = np.clip(np.asarray(eval_prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    y = y_cal.astype(int).reset_index(drop=True)
    if method == "uncalibrated" or y.nunique() < 2:
        return cal_p, eval_p
    if method == "sigmoid":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(cal_p.reshape(-1, 1), y)
        return lr.predict_proba(cal_p.reshape(-1, 1))[:, 1], lr.predict_proba(eval_p.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_p, y)
        return iso.transform(cal_p), iso.transform(eval_p)
    raise ValueError(f"Unknown calibration method: {method}")


def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for key_values, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_cols, key_values))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{metric}_std"] = float(finite.std(ddof=0)) if finite.size else float("nan")
            row[f"{metric}_var"] = float(finite.var(ddof=0)) if finite.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def make_split_indices(y: pd.Series, fold_count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def run_binary_cv(
    X: pd.DataFrame,
    y_three: pd.Series,
    configs: list[ModelConfig],
    *,
    seeds: list[int],
    fold_count: int,
    stage: str,
    tracker: ProgressTracker,
    tables_dir: Path,
    save_importance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_binary = task_target(y_three, "binary")
    rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    importance_rows_out: list[dict[str, Any]] = []
    for seed in seeds:
        for fold_idx, (train_idx, valid_idx) in enumerate(make_split_indices(y_binary, fold_count, seed), start=1):
            X_train_base = X.iloc[train_idx].reset_index(drop=True)
            X_valid_base = X.iloc[valid_idx].reset_index(drop=True)
            y_train_base = y_binary.iloc[train_idx].reset_index(drop=True)
            y_valid = y_binary.iloc[valid_idx].reset_index(drop=True)
            fit_idx, cal_idx = train_test_split(
                np.arange(len(y_train_base)),
                test_size=0.25,
                random_state=seed + fold_idx,
                stratify=y_train_base,
            )
            X_fit = X_train_base.iloc[fit_idx].reset_index(drop=True)
            X_cal = X_train_base.iloc[cal_idx].reset_index(drop=True)
            y_fit = y_train_base.iloc[fit_idx].reset_index(drop=True)
            y_cal = y_train_base.iloc[cal_idx].reset_index(drop=True)
            view = apply_feature_policy(X_fit, X_valid_base, SCREENING_POLICY)
            cal_view = apply_feature_policy(X_fit, X_cal, SCREENING_POLICY)
            for config in configs:
                context = {"stage": stage, "seed": seed, "fold": fold_idx, "model_id": config.model_id}
                tracker.log(f"{stage} binary seed={seed} fold={fold_idx}/{fold_count} {config.model_id}", stage=stage, context=context)
                fitted = fit_model(view.X_train, y_fit, config, seed + fold_idx)
                cal_raw = predict_model(fitted, cal_view.X_eval, "binary")[:, 1]
                valid_raw = predict_model(fitted, view.X_eval, "binary")[:, 1]
                fit_raw = predict_model(fitted, view.X_train, "binary")[:, 1]
                if save_importance:
                    importance_rows_out.extend(feature_importance_rows(fitted, config, seed, stage))
                for method in CALIBRATION_METHODS:
                    cal_prob, valid_prob = calibrated_probabilities(y_cal, cal_raw, valid_raw, method)
                    _, fit_prob = calibrated_probabilities(y_cal, cal_raw, fit_raw, method)
                    for policy in BINARY_THRESHOLD_POLICIES:
                        threshold, threshold_train_metrics = choose_threshold(y_cal, cal_prob, policy)
                        valid_metrics = binary_metrics(y_valid, valid_prob, threshold)
                        train_metrics = binary_metrics(y_fit, fit_prob, threshold)
                        row = {
                            "stage": stage,
                            "task": "binary",
                            "feature_policy": SCREENING_POLICY,
                            "seed": seed,
                            "fold_count": fold_count,
                            "fold": fold_idx,
                            "model_id": config.model_id,
                            "model_family": config.model_family,
                            "class_weight_policy": config.class_weight_policy,
                            "use_scm": config.use_scm,
                            "calibration": method,
                            "threshold_policy": policy,
                            "threshold": float(threshold),
                            "threshold_train_balanced_accuracy": threshold_train_metrics["balanced_accuracy"],
                            "train_size": fitted["fit_size"],
                            "model_train_size": fitted["train_size"],
                            "augmented_size": fitted["augmented_size"],
                            **valid_metrics,
                        }
                        rows.append(row)
                        gap_rows.append(
                            {
                                "stage": stage,
                                "task": "binary",
                                "feature_policy": SCREENING_POLICY,
                                "seed": seed,
                                "fold_count": fold_count,
                                "fold": fold_idx,
                                "model_id": config.model_id,
                                "calibration": method,
                                "threshold_policy": policy,
                                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                                "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                                "generalization_gap_balanced_accuracy": train_metrics["balanced_accuracy"] - valid_metrics["balanced_accuracy"],
                                "train_macro_f1": train_metrics["macro_f1"],
                                "valid_macro_f1": valid_metrics["macro_f1"],
                                "generalization_gap_macro_f1": train_metrics["macro_f1"] - valid_metrics["macro_f1"],
                                "train_specificity": train_metrics["specificity"],
                                "valid_specificity": valid_metrics["specificity"],
                                "generalization_gap_specificity": train_metrics["specificity"] - valid_metrics["specificity"],
                            }
                        )
                pd.DataFrame(rows).to_csv(tables_dir / f"{stage}_binary_by_fold.partial.csv", index=False, encoding="utf-8-sig")
                tracker.log(f"done {stage} binary seed={seed} fold={fold_idx}/{fold_count} {config.model_id}", stage=f"{stage}_done", advance=1, context=context)
    return pd.DataFrame(rows), pd.DataFrame(gap_rows), pd.DataFrame(importance_rows_out)


def run_three_cv(
    X: pd.DataFrame,
    y_three: pd.Series,
    configs: list[ModelConfig],
    *,
    seeds: list[int],
    fold_count: int,
    stage: str,
    tracker: ProgressTracker,
    tables_dir: Path,
    save_importance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    importance_rows_out: list[dict[str, Any]] = []
    for seed in seeds:
        for fold_idx, (train_idx, valid_idx) in enumerate(make_split_indices(y_three, fold_count, seed), start=1):
            X_train_base = X.iloc[train_idx].reset_index(drop=True)
            X_valid_base = X.iloc[valid_idx].reset_index(drop=True)
            y_train = y_three.iloc[train_idx].astype(int).reset_index(drop=True)
            y_valid = y_three.iloc[valid_idx].astype(int).reset_index(drop=True)
            view = apply_feature_policy(X_train_base, X_valid_base, SCREENING_POLICY)
            for config in configs:
                context = {"stage": stage, "seed": seed, "fold": fold_idx, "model_id": config.model_id}
                tracker.log(f"{stage} three seed={seed} fold={fold_idx}/{fold_count} {config.model_id}", stage=stage, context=context)
                fitted = fit_model(view.X_train, y_train, config, seed + fold_idx)
                valid_proba = predict_model(fitted, view.X_eval, "three")
                train_proba = predict_model(fitted, view.X_train, "three")
                valid_metrics = collect_metrics(y_valid, valid_proba, "three")
                train_metrics = collect_metrics(y_train, train_proba, "three")
                if save_importance:
                    importance_rows_out.extend(feature_importance_rows(fitted, config, seed, stage))
                rows.append(
                    {
                        "stage": stage,
                        "task": "three",
                        "feature_policy": SCREENING_POLICY,
                        "seed": seed,
                        "fold_count": fold_count,
                        "fold": fold_idx,
                        "model_id": config.model_id,
                        "model_family": config.model_family,
                        "class_weight_policy": config.class_weight_policy,
                        "use_scm": config.use_scm,
                        "train_size": fitted["fit_size"],
                        "model_train_size": fitted["train_size"],
                        "augmented_size": fitted["augmented_size"],
                        **valid_metrics,
                    }
                )
                gap_rows.append(
                    {
                        "stage": stage,
                        "task": "three",
                        "feature_policy": SCREENING_POLICY,
                        "seed": seed,
                        "fold_count": fold_count,
                        "fold": fold_idx,
                        "model_id": config.model_id,
                        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                        "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                        "generalization_gap_balanced_accuracy": train_metrics["balanced_accuracy"] - valid_metrics["balanced_accuracy"],
                        "train_macro_f1": train_metrics["macro_f1"],
                        "valid_macro_f1": valid_metrics["macro_f1"],
                        "generalization_gap_macro_f1": train_metrics["macro_f1"] - valid_metrics["macro_f1"],
                        "train_class0_recall": train_metrics["class0_recall"],
                        "valid_class0_recall": valid_metrics["class0_recall"],
                        "generalization_gap_class0_recall": train_metrics["class0_recall"] - valid_metrics["class0_recall"],
                    }
                )
                pd.DataFrame(rows).to_csv(tables_dir / f"{stage}_three_by_fold.partial.csv", index=False, encoding="utf-8-sig")
                tracker.log(f"done {stage} three seed={seed} fold={fold_idx}/{fold_count} {config.model_id}", stage=f"{stage}_done", advance=1, context=context)
    return pd.DataFrame(rows), pd.DataFrame(gap_rows), pd.DataFrame(importance_rows_out)


def rank_binary_candidates(summary: pd.DataFrame, gaps: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    gap_summary = aggregate_mean_std(
        gaps,
        ["stage", "model_id", "calibration", "threshold_policy"],
        ["generalization_gap_balanced_accuracy", "generalization_gap_macro_f1", "generalization_gap_specificity"],
    )
    merged = summary.merge(gap_summary, on=["stage", "model_id", "calibration", "threshold_policy"], how="left")
    merged["meets_sens_0_93"] = merged["sensitivity_mean"] >= 0.93
    merged["meets_sens_0_95"] = merged["sensitivity_mean"] >= 0.95
    merged["selection_score"] = (
        merged["balanced_accuracy_mean"].fillna(0)
        + 0.20 * merged["specificity_mean"].fillna(0)
        + 0.10 * merged["macro_f1_mean"].fillna(0)
        - 0.25 * merged["generalization_gap_balanced_accuracy_mean"].fillna(0)
        - 0.05 * merged["ece_mean"].fillna(0)
    )
    ranked = merged.sort_values(
        [
            "meets_sens_0_93",
            "balanced_accuracy_mean",
            "specificity_mean",
            "macro_f1_mean",
            "generalization_gap_balanced_accuracy_mean",
        ],
        ascending=[False, False, False, False, True],
    )
    return ranked.head(top_n).reset_index(drop=True)


def rank_three_candidates(summary: pd.DataFrame, gaps: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    gap_summary = aggregate_mean_std(gaps, ["stage", "model_id"], ["generalization_gap_balanced_accuracy", "generalization_gap_macro_f1"])
    merged = summary.merge(gap_summary, on=["stage", "model_id"], how="left")
    ranked = merged.sort_values(
        ["balanced_accuracy_mean", "macro_f1_mean", "class0_recall_mean", "class2_recall_mean", "generalization_gap_balanced_accuracy_mean"],
        ascending=[False, False, False, False, True],
    )
    return ranked.head(top_n).reset_index(drop=True)


def config_lookup(configs: list[ModelConfig]) -> dict[str, ModelConfig]:
    return {config.model_id: config for config in configs}


def top_binary_configs(ranking: pd.DataFrame, configs: list[ModelConfig]) -> list[ModelConfig]:
    lookup = config_lookup(configs)
    seen: set[str] = set()
    selected: list[ModelConfig] = []
    for model_id in ranking["model_id"].tolist():
        if model_id in lookup and model_id not in seen:
            selected.append(lookup[model_id])
            seen.add(model_id)
    return selected


def top_three_configs(ranking: pd.DataFrame, configs: list[ModelConfig]) -> list[ModelConfig]:
    return top_binary_configs(ranking.rename(columns={}), configs)


def summarize_feature_stability(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()
    return aggregate_mean_std(feature_df, ["task", "model_id", "feature"], ["importance"]).sort_values(
        ["task", "model_id", "importance_mean"], ascending=[True, True, False]
    )


def build_final_ranking(binary_confirmation: pd.DataFrame, three_confirmation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not binary_confirmation.empty:
        b = binary_confirmation.copy()
        b["rank_scope"] = "binary_primary"
        rows.extend(b.to_dict("records"))
    if not three_confirmation.empty:
        t = three_confirmation.copy()
        t["rank_scope"] = "three_auxiliary"
        rows.extend(t.to_dict("records"))
    return pd.DataFrame(rows)


def total_steps(binary_cfgs: list[ModelConfig], three_cfgs: list[ModelConfig], seeds: list[int], tuning_folds: int, confirmation_folds: int, smoke: bool, top_binary: int, top_three: int) -> int:
    if smoke:
        return len(seeds) * tuning_folds * (len(binary_cfgs) + len(three_cfgs)) + 6
    tuning = len(seeds) * tuning_folds * (len(binary_cfgs) + len(three_cfgs))
    confirmation = len(seeds) * confirmation_folds * (min(top_binary, len(binary_cfgs)) + min(top_three, len(three_cfgs)))
    return tuning + confirmation + 8


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    if args.smoke_test:
        output_dir = output_dir / "smoke"
        seeds = [args.seeds[0] if args.seeds else 42]
        tuning_folds = 2
        confirmation_folds = 2
        top_binary = 1
        top_three = 1
    else:
        output_dir = output_dir / "full"
        seeds = args.seeds
        tuning_folds = args.tuning_folds
        confirmation_folds = args.confirmation_folds
        top_binary = args.top_binary
        top_three = args.top_three

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    binary_cfgs = binary_configs(args.smoke_test)
    three_cfgs = three_configs(args.smoke_test)
    tracker = ProgressTracker(
        output_dir,
        total_steps(binary_cfgs, three_cfgs, seeds, tuning_folds, confirmation_folds, args.smoke_test, top_binary, top_three),
    )

    tracker.log("loading data", stage="setup")
    prepared = prepare_data(args.input)
    validation = assert_screening_policy(list(prepared.X.columns))
    write_feature_policy_files(output_dir, list(prepared.X.columns), [SCREENING_POLICY])
    X = prepared.X.reset_index(drop=True)
    y = prepared.y.astype(int).reset_index(drop=True)
    tracker.log("data loaded", stage="setup", advance=1, context={"shape": list(X.shape), "target_distribution": y.value_counts().sort_index().to_dict()})

    binary_by_fold, binary_gaps, binary_importance = run_binary_cv(
        X,
        y,
        binary_cfgs,
        seeds=seeds,
        fold_count=tuning_folds,
        stage="tuning",
        tracker=tracker,
        tables_dir=tables_dir,
        save_importance=True,
    )
    binary_by_fold.to_csv(tables_dir / "binary_tuning_trials.csv", index=False, encoding="utf-8-sig")
    binary_gaps.to_csv(tables_dir / "binary_tuning_overfit_by_fold.csv", index=False, encoding="utf-8-sig")
    binary_summary = aggregate_mean_std(
        binary_by_fold,
        ["stage", "model_id", "model_family", "class_weight_policy", "use_scm", "calibration", "threshold_policy"],
        [
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "sensitivity",
            "specificity",
            "auc",
            "ece",
            "brier_score",
            "log_loss",
            "mcc",
            "quadratic_kappa",
            "augmented_size",
        ],
    )
    binary_ranking = rank_binary_candidates(binary_summary, binary_gaps, top_binary)
    binary_summary.to_csv(tables_dir / "binary_candidate_summary.csv", index=False, encoding="utf-8-sig")
    binary_ranking.to_csv(tables_dir / "binary_candidate_ranking.csv", index=False, encoding="utf-8-sig")
    tracker.log("binary tuning summarized", stage="summary", advance=1)

    three_by_fold, three_gaps, three_importance = run_three_cv(
        X,
        y,
        three_cfgs,
        seeds=seeds,
        fold_count=tuning_folds,
        stage="tuning",
        tracker=tracker,
        tables_dir=tables_dir,
        save_importance=True,
    )
    three_by_fold.to_csv(tables_dir / "three_auxiliary_by_fold.csv", index=False, encoding="utf-8-sig")
    three_gaps.to_csv(tables_dir / "three_auxiliary_overfit_by_fold.csv", index=False, encoding="utf-8-sig")
    three_summary = aggregate_mean_std(
        three_by_fold,
        ["stage", "model_id", "model_family", "class_weight_policy", "use_scm"],
        [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "ovr_roc_auc_macro",
            "ece",
            "brier_score",
            "log_loss",
            "mcc",
            "quadratic_kappa",
            "class0_recall",
            "class1_recall",
            "class2_recall",
            "augmented_size",
        ],
    )
    three_ranking = rank_three_candidates(three_summary, three_gaps, top_three)
    three_summary.to_csv(tables_dir / "three_auxiliary_summary.csv", index=False, encoding="utf-8-sig")
    three_ranking.to_csv(tables_dir / "three_auxiliary_ranking.csv", index=False, encoding="utf-8-sig")
    tracker.log("three tuning summarized", stage="summary", advance=1)

    confirmation_binary_summary = pd.DataFrame()
    confirmation_three_summary = pd.DataFrame()
    confirmation_binary_ranking = pd.DataFrame()
    confirmation_three_ranking = pd.DataFrame()
    confirmation_gaps_all: list[pd.DataFrame] = []
    importance_frames = [binary_importance, three_importance]

    if not args.smoke_test:
        selected_binary = top_binary_configs(binary_ranking, binary_cfgs)
        selected_three = top_three_configs(three_ranking, three_cfgs)
        selected_binary_df = pd.DataFrame([config.__dict__ | {"params": json.dumps(config.params, ensure_ascii=False)} for config in selected_binary])
        selected_three_df = pd.DataFrame([config.__dict__ | {"params": json.dumps(config.params, ensure_ascii=False)} for config in selected_three])
        selected_binary_df.to_csv(tables_dir / "confirmation_selected_binary_configs.csv", index=False, encoding="utf-8-sig")
        selected_three_df.to_csv(tables_dir / "confirmation_selected_three_configs.csv", index=False, encoding="utf-8-sig")

        conf_binary_by_fold, conf_binary_gaps, conf_binary_importance = run_binary_cv(
            X,
            y,
            selected_binary,
            seeds=seeds,
            fold_count=confirmation_folds,
            stage="confirmation",
            tracker=tracker,
            tables_dir=tables_dir,
            save_importance=True,
        )
        conf_binary_by_fold.to_csv(tables_dir / "binary_confirmation_by_fold.csv", index=False, encoding="utf-8-sig")
        conf_binary_gaps.to_csv(tables_dir / "binary_confirmation_overfit_by_fold.csv", index=False, encoding="utf-8-sig")
        confirmation_binary_summary = aggregate_mean_std(
            conf_binary_by_fold,
            ["stage", "model_id", "model_family", "class_weight_policy", "use_scm", "calibration", "threshold_policy"],
            [
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "sensitivity",
                "specificity",
                "auc",
                "ece",
                "brier_score",
                "log_loss",
                "mcc",
                "quadratic_kappa",
                "augmented_size",
            ],
        )
        confirmation_binary_ranking = rank_binary_candidates(confirmation_binary_summary, conf_binary_gaps, top_binary)
        confirmation_binary_summary.to_csv(tables_dir / "binary_confirmation_summary.csv", index=False, encoding="utf-8-sig")
        confirmation_binary_ranking.to_csv(tables_dir / "binary_confirmation_ranking.csv", index=False, encoding="utf-8-sig")
        confirmation_gaps_all.append(conf_binary_gaps)
        importance_frames.append(conf_binary_importance)
        tracker.log("binary confirmation summarized", stage="summary", advance=1)

        conf_three_by_fold, conf_three_gaps, conf_three_importance = run_three_cv(
            X,
            y,
            selected_three,
            seeds=seeds,
            fold_count=confirmation_folds,
            stage="confirmation",
            tracker=tracker,
            tables_dir=tables_dir,
            save_importance=True,
        )
        conf_three_by_fold.to_csv(tables_dir / "three_confirmation_by_fold.csv", index=False, encoding="utf-8-sig")
        conf_three_gaps.to_csv(tables_dir / "three_confirmation_overfit_by_fold.csv", index=False, encoding="utf-8-sig")
        confirmation_three_summary = aggregate_mean_std(
            conf_three_by_fold,
            ["stage", "model_id", "model_family", "class_weight_policy", "use_scm"],
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "ovr_roc_auc_macro",
                "ece",
                "brier_score",
                "log_loss",
                "mcc",
                "quadratic_kappa",
                "class0_recall",
                "class1_recall",
                "class2_recall",
                "augmented_size",
            ],
        )
        confirmation_three_ranking = rank_three_candidates(confirmation_three_summary, conf_three_gaps, top_three)
        confirmation_three_summary.to_csv(tables_dir / "three_confirmation_summary.csv", index=False, encoding="utf-8-sig")
        confirmation_three_ranking.to_csv(tables_dir / "three_confirmation_ranking.csv", index=False, encoding="utf-8-sig")
        confirmation_gaps_all.append(conf_three_gaps)
        importance_frames.append(conf_three_importance)
        tracker.log("three confirmation summarized", stage="summary", advance=1)

    overfitting = pd.concat([binary_gaps, three_gaps] + confirmation_gaps_all, ignore_index=True)
    overfitting.to_csv(tables_dir / "overfitting_indicators.csv", index=False, encoding="utf-8-sig")

    feature_importance = pd.concat([frame for frame in importance_frames if not frame.empty], ignore_index=True) if any(not f.empty for f in importance_frames) else pd.DataFrame()
    feature_importance.to_csv(tables_dir / "feature_importance_tuning.csv", index=False, encoding="utf-8-sig")
    feature_stability = summarize_feature_stability(feature_importance)
    feature_stability.to_csv(tables_dir / "feature_stability.csv", index=False, encoding="utf-8-sig")

    threshold_calibration = pd.concat(
        [binary_summary.assign(summary_stage="tuning"), confirmation_binary_summary.assign(summary_stage="confirmation")],
        ignore_index=True,
    )
    threshold_calibration.to_csv(tables_dir / "binary_threshold_calibration.csv", index=False, encoding="utf-8-sig")
    binary_by_fold.to_csv(tables_dir / "binary_threshold_by_fold.csv", index=False, encoding="utf-8-sig")

    final_ranking = build_final_ranking(
        confirmation_binary_ranking if not confirmation_binary_ranking.empty else binary_ranking,
        confirmation_three_ranking if not confirmation_three_ranking.empty else three_ranking,
    )
    final_ranking.to_csv(tables_dir / "final_candidate_ranking.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_file": str(args.input),
        "output_dir": str(output_dir),
        "smoke_test": bool(args.smoke_test),
        "feature_policy": SCREENING_POLICY,
        "screening_policy_validation": validation,
        "seeds": seeds,
        "tuning_folds": tuning_folds,
        "confirmation_folds": confirmation_folds if not args.smoke_test else None,
        "binary_config_count": len(binary_cfgs),
        "three_config_count": len(three_cfgs),
        "best_binary": final_ranking[final_ranking.get("rank_scope", "") == "binary_primary"].head(1).to_dict("records"),
        "best_three": final_ranking[final_ranking.get("rank_scope", "") == "three_auxiliary"].head(1).to_dict("records"),
        "tables": {
            "binary_tuning_trials": "tables/binary_tuning_trials.csv",
            "binary_candidate_summary": "tables/binary_candidate_summary.csv",
            "binary_threshold_calibration": "tables/binary_threshold_calibration.csv",
            "binary_threshold_by_fold": "tables/binary_threshold_by_fold.csv",
            "three_auxiliary_summary": "tables/three_auxiliary_summary.csv",
            "overfitting_indicators": "tables/overfitting_indicators.csv",
            "feature_stability": "tables/feature_stability.csv",
            "final_candidate_ranking": "tables/final_candidate_ranking.csv",
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.log("summary written", stage="summary", advance=1)
    tracker.finish()
    print(f"Screening tuning experiment completed. Output: {output_dir}")


if __name__ == "__main__":
    main()
