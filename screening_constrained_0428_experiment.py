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
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
except Exception:  # pragma: no cover - handled at runtime for lean environments
    BalancedRandomForestClassifier = None
    EasyEnsembleClassifier = None

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, prepare_data
from screening_0428_experiment import (
    apply_feature_policy,
    assert_screening_policy,
    fit_preprocessor_train_only,
    transform_with_fitted,
    write_feature_policy_files,
)


warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "rerun_0428_outputs" / "screening_constrained_0428"
SCREENING_POLICY = "screening_no_post"
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]

ARR_COL = "ARR比值"
ALD_COL = "醛固酮"
RENIN_COL = "肾素"
K_COL = "钾"
SBP_COL = "收缩压"
DBP_COL = "舒展压"
PRE_ALD_COL = "试验前醛固酮"
PRE_RENIN_COL = "试验前肾素"
RASS_COL = "RASS_等效分数"
DIURETIC_COL = "利尿剂_等效分数"
DHP_COL = "二氢吡啶类_等效分数"
BETA_COL = "Beta_等效分数"
ALPHA_COL = "Alpha_等效分数"
NDHP_COL = "非二氢吡啶类_等效分数"

CALIBRATION_METHODS = ["uncalibrated", "sigmoid", "isotonic"]
THRESHOLD_POLICIES = [
    "fixed_0_50",
    "sens090_maxspec",
    "sens090_maxacc",
    "sens090_maxba",
    "sens093_maxspec",
    "sens093_maxacc",
    "sens093_maxba",
    "sens095_maxspec",
    "sens095_maxacc",
    "sens095_maxba",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    feature_set: str
    params: dict[str, Any]
    class_weight_policy: str = "balanced"
    components: tuple[str, ...] = ()


class ProgressTracker:
    def __init__(self, output_dir: Path, total_steps: int) -> None:
        self.output_dir = output_dir
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0
        self.start_time = time.perf_counter()
        self.progress_file = output_dir / "progress.json"
        self.log_file = output_dir / "progress.log"
        if self.log_file.exists():
            self.log_file.unlink()
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
    parser = argparse.ArgumentParser(description="ARR 0428 constrained-sensitivity screening experiment.")
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def binary_target(y: pd.Series) -> pd.Series:
    return (y.astype(int) != 0).astype(int).reset_index(drop=True)


def numeric_column(X: pd.DataFrame, column: str) -> pd.Series:
    if column not in X.columns:
        return pd.Series(np.nan, index=X.index, dtype=float)
    return pd.to_numeric(X[column], errors="coerce")


def safe_log1p(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.log1p(values.clip(lower=0))


def add_engineered_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    X = X_raw.copy()
    arr = numeric_column(X, ARR_COL)
    ald = numeric_column(X, ALD_COL)
    renin = numeric_column(X, RENIN_COL)
    potassium = numeric_column(X, K_COL)
    sbp = numeric_column(X, SBP_COL)
    dbp = numeric_column(X, DBP_COL)
    pre_ald = numeric_column(X, PRE_ALD_COL)
    pre_renin = numeric_column(X, PRE_RENIN_COL)
    suppress = (
        numeric_column(X, BETA_COL).fillna(0)
        + numeric_column(X, RASS_COL).fillna(0)
        + numeric_column(X, ALPHA_COL).fillna(0)
    )
    stimulate = (
        numeric_column(X, DIURETIC_COL).fillna(0)
        + numeric_column(X, DHP_COL).fillna(0)
        + numeric_column(X, NDHP_COL).fillna(0)
    )

    X["eng_log_arr"] = safe_log1p(arr)
    X["eng_log_aldosterone"] = safe_log1p(ald)
    X["eng_log_renin_abs"] = np.log1p(renin.abs())
    X["eng_screen_arr_ratio"] = ald / (renin.abs() + 1e-3)
    X["eng_pre_arr_ratio"] = pre_ald / (pre_renin.abs() + 1e-3)
    X["eng_pre_screen_ald_ratio"] = pre_ald / (ald.abs() + 1e-3)
    X["eng_pre_screen_renin_ratio"] = pre_renin / (renin.abs() + 1e-3)
    X["eng_arr_low_k"] = arr * np.clip(4.0 - potassium, a_min=0.0, a_max=None)
    X["eng_arr_sbp_load"] = arr * sbp / 100.0
    X["eng_arr_bp_load"] = arr * (sbp + dbp) / 200.0
    X["eng_med_suppress_score"] = suppress
    X["eng_med_stimulate_score"] = stimulate
    X["eng_arr_med_adjusted"] = arr * (1.0 + 0.1 * suppress) / (1.0 + 0.1 * stimulate)
    return X


def feature_frame(X_raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "base":
        return X_raw.copy()
    if feature_set == "engineered":
        return add_engineered_features(X_raw)
    raise ValueError(f"Unknown feature set: {feature_set}")


def xgb_model(params: dict[str, Any], seed: int) -> XGBClassifier:
    defaults = {
        "n_estimators": 220,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 1.0,
        "reg_alpha": 0.5,
        "reg_lambda": 20.0,
    }
    defaults.update(params)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **defaults,
    )


def sample_weights(y: pd.Series, policy: str) -> np.ndarray | None:
    if policy == "none":
        return None
    weights = compute_sample_weight(class_weight="balanced", y=y)
    if policy == "class0_x1_25":
        weights = weights.astype(float)
        weights[y.to_numpy(dtype=int) == 0] *= 1.25
    return weights


def fit_single_model(X_train_raw: pd.DataFrame, y_train: pd.Series, spec: ModelSpec, seed: int) -> dict[str, Any]:
    X_features = feature_frame(X_train_raw.reset_index(drop=True), spec.feature_set)
    y_fit = y_train.astype(int).reset_index(drop=True)
    preprocessor, X_df = fit_preprocessor_train_only(X_features)
    weights = sample_weights(y_fit, spec.class_weight_policy)

    if spec.family == "xgb":
        model = xgb_model(spec.params, seed)
        model.fit(X_df, y_fit, sample_weight=weights, verbose=False)
    elif spec.family == "catboost":
        model = CatBoostClassifier(
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            **spec.params,
        )
        model.fit(X_df, y_fit, sample_weight=weights)
    elif spec.family == "balanced_rf":
        if BalancedRandomForestClassifier is None:
            raise RuntimeError("imblearn BalancedRandomForestClassifier is not available")
        model = BalancedRandomForestClassifier(random_state=seed, n_jobs=-1, **spec.params)
        model.fit(X_df, y_fit)
    elif spec.family == "easy_ensemble":
        if EasyEnsembleClassifier is None:
            raise RuntimeError("imblearn EasyEnsembleClassifier is not available")
        model = EasyEnsembleClassifier(random_state=seed, n_jobs=-1, **spec.params)
        model.fit(X_df, y_fit)
    else:
        raise ValueError(f"Unknown single model family: {spec.family}")

    return {
        "kind": "single",
        "spec": spec,
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": list(X_df.columns),
        "train_size": int(len(y_fit)),
    }


def component_specs(smoke: bool) -> dict[str, ModelSpec]:
    estimator_count = 5 if smoke else 160
    cat_iterations = 5 if smoke else 220
    rf_estimators = 10 if smoke else 160
    return {
        "xgb_d3_l20": ModelSpec(
            "xgb_d3_l20",
            "xgb",
            "engineered",
            {
                "n_estimators": estimator_count,
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
        "cat_d3_l50": ModelSpec(
            "cat_d3_l50",
            "catboost",
            "engineered",
            {
                "iterations": cat_iterations,
                "depth": 3,
                "learning_rate": 0.03,
                "l2_leaf_reg": 50.0,
            },
        ),
        "brf": ModelSpec(
            "brf",
            "balanced_rf",
            "engineered",
            {
                "n_estimators": rf_estimators,
                "max_depth": 3,
                "min_samples_leaf": 4,
                "replacement": True,
                "sampling_strategy": "all",
            },
        ),
    }


def model_specs(smoke: bool) -> list[ModelSpec]:
    estimator_count = 5 if smoke else 160
    cat_iterations = 5 if smoke else 220
    rf_estimators = 10 if smoke else 160
    easy_estimators = 8 if smoke else 30
    specs = [
        ModelSpec(
            "xgb_base_d3_l20",
            "xgb",
            "base",
            {
                "n_estimators": estimator_count,
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
        ModelSpec(
            "xgb_eng_d3_l20",
            "xgb",
            "engineered",
            {
                "n_estimators": estimator_count,
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
        ModelSpec(
            "xgb_eng_d2_l10",
            "xgb",
            "engineered",
            {
                "n_estimators": max(70, int(estimator_count * 0.82)),
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
        ModelSpec(
            "xgb_eng_d2_l50_c0x1_25",
            "xgb",
            "engineered",
            {
                "n_estimators": estimator_count,
                "max_depth": 2,
                "min_child_weight": 20,
                "gamma": 2.0,
                "subsample": 0.75,
                "colsample_bytree": 0.8,
                "learning_rate": 0.03,
                "reg_alpha": 1.0,
                "reg_lambda": 50.0,
            },
            class_weight_policy="class0_x1_25",
        ),
        ModelSpec(
            "cat_eng_d3_l50",
            "catboost",
            "engineered",
            {
                "iterations": cat_iterations,
                "depth": 3,
                "learning_rate": 0.03,
                "l2_leaf_reg": 50.0,
            },
        ),
        ModelSpec(
            "brf_eng_d3",
            "balanced_rf",
            "engineered",
            {
                "n_estimators": rf_estimators,
                "max_depth": 3,
                "min_samples_leaf": 4,
                "replacement": True,
                "sampling_strategy": "all",
            },
        ),
        ModelSpec(
            "easy_ensemble_eng",
            "easy_ensemble",
            "engineered",
            {
                "n_estimators": easy_estimators,
                "sampling_strategy": "auto",
            },
        ),
        ModelSpec(
            "soft_vote_xgb_cat",
            "soft_voting",
            "engineered",
            {},
            components=("xgb_d3_l20", "cat_d3_l50"),
        ),
        ModelSpec(
            "soft_vote_xgb_cat_brf",
            "soft_voting",
            "engineered",
            {},
            components=("xgb_d3_l20", "cat_d3_l50", "brf"),
        ),
    ]
    if smoke:
        return [specs[1], specs[7]]
    return specs


def fit_model(X_train_raw: pd.DataFrame, y_train: pd.Series, spec: ModelSpec, seed: int, smoke: bool) -> dict[str, Any]:
    if spec.family != "soft_voting":
        return fit_single_model(X_train_raw, y_train, spec, seed)
    lookup = component_specs(smoke)
    components = [
        fit_single_model(X_train_raw, y_train, lookup[name], seed + idx * 37)
        for idx, name in enumerate(spec.components)
        if name in lookup
    ]
    return {"kind": "ensemble", "spec": spec, "components": components, "train_size": int(len(y_train))}


def predict_model(fitted: dict[str, Any], X_raw: pd.DataFrame) -> np.ndarray:
    if fitted["kind"] == "ensemble":
        probs = [predict_model(component, X_raw) for component in fitted["components"]]
        return np.mean(probs, axis=0)
    spec: ModelSpec = fitted["spec"]
    X_features = feature_frame(X_raw.reset_index(drop=True), spec.feature_set)
    X_df = transform_with_fitted(fitted["preprocessor"], X_features)
    proba = fitted["model"].predict_proba(X_df)
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    if arr.shape[1] == 1:
        arr = np.column_stack([1.0 - arr[:, 0], arr[:, 0]])
    arr = np.clip(arr, 1e-6, 1.0)
    return arr / arr.sum(axis=1, keepdims=True)


def feature_importance_rows(fitted: dict[str, Any], seed: int, fold: int) -> list[dict[str, Any]]:
    if fitted["kind"] == "ensemble":
        rows: list[dict[str, Any]] = []
        for component in fitted["components"]:
            rows.extend(feature_importance_rows(component, seed, fold))
        return rows
    model = fitted["model"]
    spec: ModelSpec = fitted["spec"]
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        values = np.asarray(model.get_feature_importance(), dtype=float)
    if values is None or len(values) != len(fitted["feature_names"]):
        return []
    total = float(values.sum())
    if total > 0:
        values = values / total
    return [
        {
            "model_id": spec.model_id,
            "family": spec.family,
            "feature_set": spec.feature_set,
            "seed": seed,
            "fold": fold,
            "feature": name.replace("num__", "", 1).replace("cat__", "", 1),
            "importance": float(value),
        }
        for name, value in zip(fitted["feature_names"], values)
    ]


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


def binary_ece(y_true: pd.Series, prob_pos: np.ndarray, threshold: float, n_bins: int = 10) -> float:
    y = y_true.to_numpy(dtype=int)
    prob = np.clip(np.asarray(prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (prob >= threshold).astype(int)
    confidence = np.where(pred == 1, prob, 1.0 - prob)
    correctness = (pred == y).astype(float)
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
    prob = np.clip(np.asarray(prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (prob >= threshold).astype(int)
    recalls = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    precision = precision_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    auc = float(roc_auc_score(y, prob)) if y.nunique() > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "sensitivity": float(recalls[1]),
        "specificity": float(recalls[0]),
        "ppv": float(precision[1]),
        "npv": float(precision[0]),
        "auc": auc,
        "ece": binary_ece(y, prob, threshold),
        "brier_score": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - prob, prob]), labels=[0, 1])),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_target(policy: str) -> float | None:
    if policy.startswith("sens090"):
        return 0.90
    if policy.startswith("sens093"):
        return 0.93
    if policy.startswith("sens095"):
        return 0.95
    return None


def choose_threshold(y_cal: pd.Series, prob_pos: np.ndarray, policy: str) -> tuple[float, dict[str, float]]:
    y = y_cal.astype(int).reset_index(drop=True)
    prob = np.clip(np.asarray(prob_pos, dtype=float), 1e-6, 1.0 - 1e-6)
    if policy == "fixed_0_50":
        return 0.5, binary_metrics(y, prob, 0.5)
    candidates = np.unique(np.concatenate([np.linspace(0.02, 0.98, 193), prob]))
    scored = [(float(threshold), binary_metrics(y, prob, float(threshold))) for threshold in candidates]
    target = threshold_target(policy)
    if target is not None:
        feasible = [item for item in scored if item[1]["sensitivity"] >= target]
        if not feasible:
            feasible = scored
        scored = feasible
    if policy.endswith("maxspec"):
        return max(scored, key=lambda item: (item[1]["specificity"], item[1]["accuracy"], item[1]["balanced_accuracy"]))
    if policy.endswith("maxacc"):
        return max(scored, key=lambda item: (item[1]["accuracy"], item[1]["specificity"], item[1]["balanced_accuracy"]))
    if policy.endswith("maxba"):
        return max(scored, key=lambda item: (item[1]["balanced_accuracy"], item[1]["specificity"], item[1]["accuracy"]))
    raise ValueError(f"Unknown threshold policy: {policy}")


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


def rank_candidates(summary: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    gap_summary = aggregate_mean_std(
        gaps,
        ["model_id", "calibration", "threshold_policy"],
        ["generalization_gap_balanced_accuracy", "generalization_gap_accuracy", "generalization_gap_specificity"],
    )
    ranked = summary.merge(gap_summary, on=["model_id", "calibration", "threshold_policy"], how="left")
    ranked["target_sensitivity"] = ranked["threshold_policy"].map(lambda policy: threshold_target(str(policy)))
    ranked["meets_target"] = ranked.apply(
        lambda row: True
        if pd.isna(row["target_sensitivity"])
        else bool(row["sensitivity_mean"] >= row["target_sensitivity"]),
        axis=1,
    )
    ranked["selection_score"] = (
        ranked["accuracy_mean"].fillna(0)
        + 0.40 * ranked["specificity_mean"].fillna(0)
        + 0.25 * ranked["balanced_accuracy_mean"].fillna(0)
        - 0.20 * ranked["generalization_gap_balanced_accuracy_mean"].fillna(0)
        - 0.05 * ranked["ece_mean"].fillna(0)
    )
    return ranked.sort_values(
        [
            "meets_target",
            "target_sensitivity",
            "specificity_mean",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "generalization_gap_balanced_accuracy_mean",
        ],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def run_cv(
    X: pd.DataFrame,
    y_binary: pd.Series,
    specs: list[ModelSpec],
    *,
    seeds: list[int],
    folds: int,
    smoke: bool,
    tracker: ProgressTracker,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    importance_rows_out: list[dict[str, Any]] = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y_binary), start=1):
            X_train_base = X.iloc[train_idx].reset_index(drop=True)
            X_valid_base = X.iloc[valid_idx].reset_index(drop=True)
            y_train_base = y_binary.iloc[train_idx].reset_index(drop=True)
            y_valid = y_binary.iloc[valid_idx].reset_index(drop=True)
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

            for spec in specs:
                context = {"seed": seed, "fold": fold, "model_id": spec.model_id}
                tracker.log(
                    f"seed={seed} fold={fold}/{folds} {spec.model_id}",
                    stage="cv_model",
                    context=context,
                )
                fitted = fit_model(fit_view.X_train, y_fit, spec, seed + fold, smoke)
                cal_raw = predict_model(fitted, cal_view.X_eval)[:, 1]
                valid_raw = predict_model(fitted, fit_view.X_eval)[:, 1]
                train_raw = predict_model(fitted, fit_view.X_train)[:, 1]
                importance_rows_out.extend(feature_importance_rows(fitted, seed, fold))
                for calibration in CALIBRATION_METHODS:
                    cal_prob, valid_prob = calibrated_probabilities(y_cal, cal_raw, valid_raw, calibration)
                    _, train_prob = calibrated_probabilities(y_cal, cal_raw, train_raw, calibration)
                    for threshold_policy in THRESHOLD_POLICIES:
                        threshold, cal_metrics = choose_threshold(y_cal, cal_prob, threshold_policy)
                        valid_metrics = binary_metrics(y_valid, valid_prob, threshold)
                        train_metrics = binary_metrics(y_fit, train_prob, threshold)
                        metric_rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "folds": folds,
                                "model_id": spec.model_id,
                                "family": spec.family,
                                "feature_set": spec.feature_set,
                                "class_weight_policy": spec.class_weight_policy,
                                "calibration": calibration,
                                "threshold_policy": threshold_policy,
                                "threshold": threshold,
                                "cal_sensitivity": cal_metrics["sensitivity"],
                                "cal_specificity": cal_metrics["specificity"],
                                "train_size": int(len(y_fit)),
                                "valid_size": int(len(y_valid)),
                                **valid_metrics,
                            }
                        )
                        gap_rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "folds": folds,
                                "model_id": spec.model_id,
                                "family": spec.family,
                                "feature_set": spec.feature_set,
                                "calibration": calibration,
                                "threshold_policy": threshold_policy,
                                "train_accuracy": train_metrics["accuracy"],
                                "valid_accuracy": valid_metrics["accuracy"],
                                "generalization_gap_accuracy": train_metrics["accuracy"] - valid_metrics["accuracy"],
                                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                                "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                                "generalization_gap_balanced_accuracy": train_metrics["balanced_accuracy"] - valid_metrics["balanced_accuracy"],
                                "train_specificity": train_metrics["specificity"],
                                "valid_specificity": valid_metrics["specificity"],
                                "generalization_gap_specificity": train_metrics["specificity"] - valid_metrics["specificity"],
                            }
                        )
                pd.DataFrame(metric_rows).to_csv(tables_dir / "constrained_metrics_by_fold.partial.csv", index=False, encoding="utf-8-sig")
                tracker.log(
                    f"done seed={seed} fold={fold}/{folds} {spec.model_id}",
                    stage="cv_model_done",
                    advance=1,
                    context=context,
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(gap_rows), pd.DataFrame(importance_rows_out)


def feature_stability(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    return aggregate_mean_std(importance, ["model_id", "family", "feature_set", "feature"], ["importance"]).sort_values(
        ["model_id", "importance_mean"], ascending=[True, False]
    )


def write_model_config(tables_dir: Path, specs: list[ModelSpec]) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "model_id": spec.model_id,
                "family": spec.family,
                "feature_set": spec.feature_set,
                "class_weight_policy": spec.class_weight_policy,
                "components": "|".join(spec.components),
                "params": json.dumps(spec.params, ensure_ascii=False),
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "model_config.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / ("smoke" if args.smoke_test else "full")
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seeds[0] if args.seeds else 42] if args.smoke_test else args.seeds
    folds = 2 if args.smoke_test else args.folds
    specs = model_specs(args.smoke_test)
    tracker = ProgressTracker(output_dir, total_steps=len(seeds) * folds * len(specs) + 5)

    tracker.log("loading data", stage="setup")
    prepared = prepare_data(args.input)
    validation = assert_screening_policy(list(prepared.X.columns))
    write_feature_policy_files(output_dir, list(prepared.X.columns), [SCREENING_POLICY])
    write_model_config(tables_dir, specs)
    X = prepared.X.reset_index(drop=True)
    y_binary = binary_target(prepared.y)
    tracker.log(
        "data loaded",
        stage="setup",
        advance=1,
        context={"shape": list(X.shape), "binary_distribution": y_binary.value_counts().sort_index().to_dict()},
    )

    metrics, gaps, importance = run_cv(
        X,
        y_binary,
        specs,
        seeds=seeds,
        folds=folds,
        smoke=args.smoke_test,
        tracker=tracker,
        tables_dir=tables_dir,
    )
    metrics.to_csv(tables_dir / "constrained_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(tables_dir / "overfitting_indicators.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(tables_dir / "feature_importance_constrained.csv", index=False, encoding="utf-8-sig")
    tracker.log("fold tables written", stage="summary", advance=1)

    summary = aggregate_mean_std(
        metrics,
        ["model_id", "family", "feature_set", "class_weight_policy", "calibration", "threshold_policy"],
        [
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "auc",
            "ece",
            "brier_score",
            "log_loss",
            "mcc",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )
    ranking = rank_candidates(summary, gaps)
    threshold_summary = ranking[ranking["threshold_policy"].isin([policy for policy in THRESHOLD_POLICIES if policy.startswith("sens")])].copy()
    feature_summary = feature_stability(importance)
    summary.to_csv(tables_dir / "constrained_metrics_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(tables_dir / "final_candidate_ranking.csv", index=False, encoding="utf-8-sig")
    threshold_summary.to_csv(tables_dir / "threshold_policy_summary.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(tables_dir / "feature_stability.csv", index=False, encoding="utf-8-sig")
    tracker.log("summary tables written", stage="summary", advance=1)

    result = {
        "input_file": str(args.input),
        "output_dir": str(output_dir),
        "smoke_test": bool(args.smoke_test),
        "feature_policy": SCREENING_POLICY,
        "screening_policy_validation": validation,
        "seeds": seeds,
        "folds": folds,
        "model_count": len(specs),
        "threshold_policies": THRESHOLD_POLICIES,
        "calibration_methods": CALIBRATION_METHODS,
        "best_overall": ranking.head(5).to_dict("records"),
        "tables": {
            "model_config": "tables/model_config.csv",
            "constrained_metrics_by_fold": "tables/constrained_metrics_by_fold.csv",
            "constrained_metrics_summary": "tables/constrained_metrics_summary.csv",
            "threshold_policy_summary": "tables/threshold_policy_summary.csv",
            "overfitting_indicators": "tables/overfitting_indicators.csv",
            "feature_importance_constrained": "tables/feature_importance_constrained.csv",
            "feature_stability": "tables/feature_stability.csv",
            "final_candidate_ranking": "tables/final_candidate_ranking.csv",
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.log("experiment summary written", stage="summary", advance=1)
    tracker.finish()
    print(f"Constrained screening experiment completed. Output: {output_dir}")


if __name__ == "__main__":
    main()
