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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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

from causal_xgboost_variants_experiment import (
    XGB_BEST_PARAMS,
    fit_custom_softprob_booster,
    fit_ordinal_xgboost,
    predict_custom_softprob_bundle,
)
from frontier_scm_v2_experiment import SCMMixV2Augmentor, build_hard_case_index_map
from multiclass_ensemble_experiment import (
    DEFAULT_INPUT_FILE,
    apply_controlled_adasyn,
    build_preprocessor,
    prepare_data,
    transform_with_preprocessor,
)


warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "rerun_0428_outputs" / "screening_no_post"
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]
DEFAULT_CV_FOLDS = [5, 10]

POST_TEST_COLUMNS = ["试验后醛固酮", "试验后肾素"]
REQUIRED_RETAINED_COLUMNS = [
    "试验前醛固酮",
    "试验前肾素",
    "确诊实验类型",
    "ARR比值>192为阳性，推荐进行确诊试验",
]
FEATURE_POLICIES = ["full_reference", "screening_no_post", "post_mask_stress"]
TASK_MODES = ["three", "binary", "both"]

THREE_CORE_MODELS = [
    "xgb_reference_raw",
    "xgb_reference_adasyn",
    "catboost_reference",
    "xgb_scm_v2_best",
]
THREE_CANDIDATE_MODELS = [
    "causal_mainline_xgboost",
    "ordinal_mainline_xgboost",
    "causal_scm_v2_joint",
    "ordinal_scm_v2_joint",
]
BINARY_CORE_MODELS = [
    "xgb_binary_reference",
    "catboost_binary_reference",
    "xgb_binary_scm_v2",
]
BINARY_CANDIDATE_MODELS = ["causal_binary_xgboost"]

CATBOOST_BEST_PARAMS = {
    "iterations": 339,
    "depth": 4,
    "learning_rate": 0.05182367293641893,
    "l2_leaf_reg": 1.4808945119975185,
    "random_strength": 0.0016834192018216926,
    "bagging_temperature": 0.9488855372533332,
    "border_count": 249,
}
SCM_V2_BEST_CONFIG = {
    "seed_strategy": "hard_case_seed",
    "target_classes": "class0_and_class2",
    "treat_mix_prob": 0.2,
    "residual_scale": 0.8,
    "teacher_mode": "single",
}
SUMMARY_METRICS = [
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
    "top2_accuracy",
    "class0_recall",
    "class1_recall",
    "class2_recall",
    "sensitivity",
    "specificity",
]
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
    "class2_recall",
    "sensitivity",
    "specificity",
]


@dataclass
class FeatureView:
    X_train: pd.DataFrame
    X_eval: pd.DataFrame
    X_train_eval: pd.DataFrame
    included_columns: list[str]
    dropped_columns: list[str]
    masked_columns: list[str]


@dataclass
class ModelBundle:
    bundle_type: str
    task: str
    model_name: str
    model: Any
    preprocessor: Any | None
    feature_names: list[str]
    train_size: int
    resampled_train_size: int
    n_classes: int
    transformed_train: pd.DataFrame | None = None
    extra: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARR 0428 screening experiment without post-test features.")
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-mode", choices=TASK_MODES, default="both")
    parser.add_argument(
        "--feature-policy",
        choices=FEATURE_POLICIES,
        nargs="+",
        default=FEATURE_POLICIES,
    )
    parser.add_argument("--cv-folds", type=int, nargs="*", default=DEFAULT_CV_FOLDS)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--include-causal-binary",
        action="store_true",
        help="Force causal_binary_xgboost even when binary SCM does not clear the gain rule.",
    )
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def requested_tasks(task_mode: str) -> list[str]:
    if task_mode == "both":
        return ["three", "binary"]
    return [task_mode]


def binary_target(y: pd.Series) -> pd.Series:
    return (y.astype(int) != 0).astype(int)


def task_target(y: pd.Series, task: str) -> pd.Series:
    return y.astype(int).reset_index(drop=True) if task == "three" else binary_target(y).reset_index(drop=True)


def task_classes(task: str) -> list[int]:
    return [0, 1, 2] if task == "three" else [0, 1]


def task_label_names(task: str) -> dict[int, str]:
    if task == "three":
        return {0: "non_confirmed", 1: "confirmed", 2: "gray_zone"}
    return {0: "non_confirmed", 1: "confirmed_or_gray"}


def normalize_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    if arr.shape[1] != n_classes:
        aligned = np.zeros((arr.shape[0], n_classes), dtype=float)
        copy_cols = min(arr.shape[1], n_classes)
        aligned[:, :copy_cols] = arr[:, :copy_cols]
        arr = aligned
    arr = np.clip(arr, 1e-6, 1.0)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return arr / row_sum


def calibration_metrics(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    y_arr = y_true.to_numpy(dtype=int)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_arr)), y_arr] = 1.0
    brier = float(np.mean(np.sum((one_hot - proba) ** 2, axis=1)))
    confidence = proba.max(axis=1)
    prediction = np.argmax(proba, axis=1)
    correctness = (prediction == y_arr).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = bin_edges[idx], bin_edges[idx + 1]
        mask = (confidence >= left) & (confidence <= right if idx == n_bins - 1 else confidence < right)
        if np.any(mask):
            ece += abs(float(correctness[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return {"ece": float(ece), "brier_score": brier}


def safe_roc_auc(y_true: pd.Series, proba: np.ndarray, classes: list[int]) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def top_k_accuracy(y_true: pd.Series, proba: np.ndarray, k: int) -> float:
    if k >= proba.shape[1]:
        return 1.0
    top = np.argsort(proba, axis=1)[:, -k:]
    y_arr = y_true.to_numpy(dtype=int)
    return float(np.mean([label in top[idx] for idx, label in enumerate(y_arr)]))


def collect_metrics(y_true: pd.Series, proba: np.ndarray, task: str) -> dict[str, float]:
    classes = task_classes(task)
    normalized = normalize_proba(proba, len(classes))
    pred = np.argmax(normalized, axis=1)
    cal = calibration_metrics(y_true, normalized)
    recalls = recall_score(y_true, pred, labels=classes, average=None, zero_division=0)
    row: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "ovr_roc_auc_macro": safe_roc_auc(y_true, normalized, classes),
        "ece": cal["ece"],
        "brier_score": cal["brier_score"],
        "log_loss": float(log_loss(y_true, normalized, labels=classes)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "top2_accuracy": top_k_accuracy(y_true, normalized, k=min(2, len(classes))),
        "class0_recall": float(recalls[0]) if len(recalls) > 0 else float("nan"),
        "class1_recall": float(recalls[1]) if len(recalls) > 1 else float("nan"),
        "class2_recall": float(recalls[2]) if len(recalls) > 2 else float("nan"),
        "sensitivity": float(recalls[1]) if task == "binary" and len(recalls) > 1 else float("nan"),
        "specificity": float(recalls[0]) if task == "binary" and len(recalls) > 0 else float("nan"),
    }
    return row


def compute_bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 2500) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def get_policy_columns(all_columns: list[str], policy: str) -> tuple[list[str], list[str], list[str]]:
    if policy == "full_reference":
        return all_columns.copy(), [], []
    if policy == "screening_no_post":
        dropped = [col for col in POST_TEST_COLUMNS if col in all_columns]
        return [col for col in all_columns if col not in dropped], dropped, []
    if policy == "post_mask_stress":
        masked = [col for col in POST_TEST_COLUMNS if col in all_columns]
        return all_columns.copy(), [], masked
    raise ValueError(f"Unknown feature policy: {policy}")


def apply_feature_policy(X_train: pd.DataFrame, X_eval: pd.DataFrame, policy: str) -> FeatureView:
    columns, dropped, masked = get_policy_columns(list(X_train.columns), policy)
    train_view = X_train[columns].copy().reset_index(drop=True)
    eval_view = X_eval[columns].copy().reset_index(drop=True)
    train_eval = train_view.copy()
    if policy == "post_mask_stress":
        for col in masked:
            eval_view[col] = np.nan
    return FeatureView(
        X_train=train_view,
        X_eval=eval_view,
        X_train_eval=train_eval,
        included_columns=columns,
        dropped_columns=dropped,
        masked_columns=masked,
    )


def assert_screening_policy(prepared_columns: list[str]) -> dict[str, Any]:
    included, dropped, _ = get_policy_columns(prepared_columns, "screening_no_post")
    retained = [col for col in REQUIRED_RETAINED_COLUMNS if col in included]
    missing_retained = [col for col in REQUIRED_RETAINED_COLUMNS if col not in included]
    bad_cols = [
        col
        for col in included
        if col in POST_TEST_COLUMNS or "missingindicator" in col.lower() or col == "住院号" or str(col).startswith("Unnamed")
    ]
    if bad_cols:
        raise AssertionError(f"screening_no_post contains forbidden columns: {bad_cols}")
    return {
        "screening_no_post_dropped": dropped,
        "required_retained_present": retained,
        "required_retained_missing": missing_retained,
        "screening_no_post_feature_count": len(included),
    }


def fit_preprocessor_train_only(X_train_raw: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X_train_raw.columns if col not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, _ = transform_with_preprocessor(preprocessor, X_train_raw, None)
    return preprocessor, X_train_df


def transform_with_fitted(preprocessor: Any, X_raw: pd.DataFrame) -> pd.DataFrame:
    transformed = preprocessor.transform(X_raw)
    return pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out(), index=X_raw.index)


def discrete_features_from_raw(X_raw: pd.DataFrame) -> list[str]:
    numeric_features = X_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    discrete: list[str] = []
    for col in numeric_features:
        values = X_raw[col].dropna()
        if values.empty:
            continue
        unique_count = values.nunique()
        if unique_count <= 12 or np.allclose(values, np.round(values)):
            discrete.append(col)
    return discrete


def safe_adasyn(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    discrete_numeric_features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, bool]:
    try:
        X_resampled, y_resampled = apply_controlled_adasyn(
            X_train_df,
            y_train,
            discrete_numeric_features,
            seed,
        )
        return X_resampled, y_resampled, len(y_resampled) != len(y_train)
    except Exception:
        return X_train_df, y_train, False


def make_xgb_model(task: str, seed: int) -> XGBClassifier:
    if task == "binary":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **XGB_BEST_PARAMS,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )


def make_catboost_model(task: str, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss" if task == "binary" else "MultiClass",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        **CATBOOST_BEST_PARAMS,
    )


def fit_xgb_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: str,
    model_name: str,
    seed: int,
    use_adasyn: bool,
) -> ModelBundle:
    discrete = discrete_features_from_raw(X_train_raw)
    preprocessor, X_train_df = fit_preprocessor_train_only(X_train_raw)
    y_fit = y_train.copy().reset_index(drop=True)
    X_fit = X_train_df
    adasyn_used = False
    if use_adasyn:
        X_fit, y_fit, adasyn_used = safe_adasyn(X_train_df, y_fit, discrete, seed)
    model = make_xgb_model(task, seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    model.fit(X_fit, y_fit, sample_weight=sample_weight, verbose=False)
    return ModelBundle(
        bundle_type="xgb",
        task=task,
        model_name=model_name,
        model=model,
        preprocessor=preprocessor,
        feature_names=list(X_train_df.columns),
        train_size=int(len(X_train_raw)),
        resampled_train_size=int(len(y_fit)),
        n_classes=len(task_classes(task)),
        transformed_train=X_train_df,
        extra={"adasyn_used": adasyn_used},
    )


def fit_catboost_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: str,
    model_name: str,
    seed: int,
) -> ModelBundle:
    preprocessor, X_train_df = fit_preprocessor_train_only(X_train_raw)
    model = make_catboost_model(task, seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_df, y_train, sample_weight=sample_weight)
    return ModelBundle(
        bundle_type="catboost",
        task=task,
        model_name=model_name,
        model=model,
        preprocessor=preprocessor,
        feature_names=list(X_train_df.columns),
        train_size=int(len(X_train_raw)),
        resampled_train_size=int(len(y_train)),
        n_classes=len(task_classes(task)),
        transformed_train=X_train_df,
    )


def fit_causal_three_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_name: str,
    seed: int,
) -> ModelBundle:
    preprocessor, X_train_df = fit_preprocessor_train_only(X_train_raw)
    model_bundle, _, _ = fit_custom_softprob_booster(
        X_train_df,
        X_train_df.copy(),
        y_train,
        discrete_numeric_features=discrete_features_from_raw(X_train_raw),
        objective_mode="emd",
        use_fw_gpl=True,
        seed=seed,
    )
    return ModelBundle(
        bundle_type="causal_three",
        task="three",
        model_name=model_name,
        model=model_bundle,
        preprocessor=preprocessor,
        feature_names=list(X_train_df.columns),
        train_size=int(len(X_train_raw)),
        resampled_train_size=int(model_bundle["train_size"]),
        n_classes=3,
        transformed_train=X_train_df,
    )


def fit_ordinal_three_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_name: str,
    seed: int,
) -> ModelBundle:
    preprocessor, X_train_df = fit_preprocessor_train_only(X_train_raw)
    model_bundle, _ = fit_ordinal_xgboost(
        X_train_df,
        X_train_df.copy(),
        y_train,
        discrete_numeric_features=discrete_features_from_raw(X_train_raw),
    )
    X_fit, y_fit, _ = safe_adasyn(X_train_df.copy(), y_train.copy(), discrete_features_from_raw(X_train_raw), seed)
    return ModelBundle(
        bundle_type="ordinal_three",
        task="three",
        model_name=model_name,
        model=model_bundle,
        preprocessor=preprocessor,
        feature_names=list(X_train_df.columns),
        train_size=int(len(X_train_raw)),
        resampled_train_size=int(len(y_fit) if len(y_fit) else len(X_fit)),
        n_classes=3,
        transformed_train=X_train_df,
    )


def add_binary_causal_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    X = X_raw.copy()

    def numeric(col: str) -> pd.Series:
        if col not in X.columns:
            return pd.Series(np.nan, index=X.index, dtype=float)
        return pd.to_numeric(X[col], errors="coerce")

    arr = numeric("ARR比值")
    ald = numeric("醛固酮")
    renin = numeric("肾素")
    k = numeric("钾")
    beta = numeric("Beta_等效分数").fillna(0.0)
    rass = numeric("RASS_等效分数").fillna(0.0)
    diuretic = numeric("利尿剂_等效分数").fillna(0.0)
    dhp = numeric("二氢吡啶类_等效分数").fillna(0.0)
    ndhp = numeric("非二氢吡啶类_等效分数").fillna(0.0)
    alpha = numeric("Alpha_等效分数").fillna(0.0)
    suppress = beta + rass + alpha
    stimulate = diuretic + dhp + ndhp
    X["binary_causal_log_arr"] = np.log1p(arr.clip(lower=0))
    X["binary_causal_ald_renin_ratio"] = ald / (renin.abs() + 1e-3)
    X["binary_causal_low_k_arr"] = arr * np.clip(4.0 - k, a_min=0.0, a_max=None)
    X["binary_causal_medication_balance"] = suppress - stimulate
    X["binary_causal_arr_medication_adjusted"] = arr * (1.0 + 0.1 * suppress) / (1.0 + 0.1 * stimulate)
    return X


def fit_binary_causal_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int,
) -> ModelBundle:
    engineered = add_binary_causal_features(X_train_raw)
    return fit_xgb_bundle(
        engineered,
        y_train,
        task="binary",
        model_name="causal_binary_xgboost",
        seed=seed,
        use_adasyn=True,
    )


def predict_bundle(bundle: ModelBundle, X_raw: pd.DataFrame) -> np.ndarray:
    X_apply = add_binary_causal_features(X_raw) if bundle.bundle_type == "binary_causal" else X_raw
    if bundle.bundle_type in {"xgb", "catboost", "binary_causal"}:
        assert bundle.preprocessor is not None
        X_df = transform_with_fitted(bundle.preprocessor, X_apply)
        return normalize_proba(bundle.model.predict_proba(X_df), bundle.n_classes)
    if bundle.bundle_type == "causal_three":
        assert bundle.preprocessor is not None
        X_df = transform_with_fitted(bundle.preprocessor, X_apply)
        prediction = predict_custom_softprob_bundle(bundle.model, X_df)
        return normalize_proba(prediction["standard_proba"], 3)
    if bundle.bundle_type == "ordinal_three":
        assert bundle.preprocessor is not None
        X_df = transform_with_fitted(bundle.preprocessor, X_apply)
        p_gt0 = bundle.model["gt0"].predict_proba(X_df)[:, 1]
        p_gt1 = bundle.model["gt1"].predict_proba(X_df)[:, 1]
        p_gt1 = np.minimum(p_gt1, p_gt0)
        p0 = 1.0 - p_gt0
        p2 = p_gt0 - p_gt1
        p1 = p_gt1
        return normalize_proba(np.column_stack([p0, p1, p2]), 3)
    raise ValueError(f"Unknown bundle type: {bundle.bundle_type}")


def augment_scm_v2(X_train_raw: pd.DataFrame, y_train: pd.Series, task: str, seed: int) -> tuple[pd.DataFrame, pd.Series, int]:
    hard_case_map = build_hard_case_index_map(X_train_raw, y_train, seed) if task == "three" else None
    augmentor = SCMMixV2Augmentor(
        random_state=seed,
        seed_strategy=SCM_V2_BEST_CONFIG["seed_strategy"],
        target_classes=SCM_V2_BEST_CONFIG["target_classes"],
        treat_mix_prob=SCM_V2_BEST_CONFIG["treat_mix_prob"],
        residual_scale=SCM_V2_BEST_CONFIG["residual_scale"],
        teacher_mode=SCM_V2_BEST_CONFIG["teacher_mode"],
    )
    aug_result = augmentor.generate(X_train_raw, y_train, hard_case_map=hard_case_map)
    X_aug = pd.concat([X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
    y_aug = pd.concat([y_train, aug_result.y_aug], axis=0, ignore_index=True)
    return X_aug, y_aug.astype(int), int(len(aug_result.X_aug))


def fit_bundle_by_name(
    model_name: str,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: str,
    seed: int,
) -> tuple[ModelBundle, int]:
    if model_name in {"xgb_reference_raw", "xgb_binary_reference"}:
        return (
            fit_xgb_bundle(
                X_train_raw,
                y_train,
                task=task,
                model_name=model_name,
                seed=seed,
                use_adasyn=False,
            ),
            0,
        )
    if model_name == "xgb_reference_adasyn":
        return (
            fit_xgb_bundle(
                X_train_raw,
                y_train,
                task=task,
                model_name=model_name,
                seed=seed,
                use_adasyn=True,
            ),
            0,
        )
    if model_name in {"catboost_reference", "catboost_binary_reference"}:
        return (
            fit_catboost_bundle(
                X_train_raw,
                y_train,
                task=task,
                model_name=model_name,
                seed=seed,
            ),
            0,
        )
    if model_name in {"xgb_scm_v2_best", "xgb_binary_scm_v2"}:
        X_aug, y_aug, augmented_size = augment_scm_v2(X_train_raw, y_train, task, seed)
        return (
            fit_xgb_bundle(
                X_aug,
                y_aug,
                task=task,
                model_name=model_name,
                seed=seed,
                use_adasyn=True,
            ),
            augmented_size,
        )
    if task == "three" and model_name == "causal_mainline_xgboost":
        return fit_causal_three_bundle(X_train_raw, y_train, model_name=model_name, seed=seed), 0
    if task == "three" and model_name == "ordinal_mainline_xgboost":
        return fit_ordinal_three_bundle(X_train_raw, y_train, model_name=model_name, seed=seed), 0
    if task == "three" and model_name == "causal_scm_v2_joint":
        X_aug, y_aug, augmented_size = augment_scm_v2(X_train_raw, y_train, task, seed)
        return fit_causal_three_bundle(X_aug, y_aug, model_name=model_name, seed=seed + 200), augmented_size
    if task == "three" and model_name == "ordinal_scm_v2_joint":
        X_aug, y_aug, augmented_size = augment_scm_v2(X_train_raw, y_train, task, seed)
        return fit_ordinal_three_bundle(X_aug, y_aug, model_name=model_name, seed=seed + 200), augmented_size
    if task == "binary" and model_name == "causal_binary_xgboost":
        bundle = fit_binary_causal_bundle(X_train_raw, y_train, seed=seed)
        bundle.bundle_type = "binary_causal"
        return bundle, 0
    raise ValueError(f"Unsupported model/task combination: {task} / {model_name}")


def model_names_for(task: str, policy: str, smoke: bool, include_causal_binary: bool) -> list[str]:
    if smoke:
        return ["xgb_reference_raw", "xgb_scm_v2_best"] if task == "three" else ["xgb_binary_reference", "xgb_binary_scm_v2"]
    if task == "three":
        if policy == "screening_no_post":
            return THREE_CORE_MODELS + THREE_CANDIDATE_MODELS
        if policy == "full_reference":
            return THREE_CORE_MODELS
        return ["xgb_reference_raw", "xgb_reference_adasyn", "catboost_reference"]
    if policy == "screening_no_post":
        models = BINARY_CORE_MODELS.copy()
        if include_causal_binary:
            models += BINARY_CANDIDATE_MODELS
        return models
    if policy == "full_reference":
        return BINARY_CORE_MODELS.copy()
    return ["xgb_binary_reference", "catboost_binary_reference"]


def build_metric_row(
    *,
    task: str,
    feature_policy: str,
    seed: int,
    model_name: str,
    y_true: pd.Series,
    proba: np.ndarray,
    train_size: int,
    eval_size: int,
    resampled_train_size: int,
    augmented_size: int,
    split_type: str,
) -> dict[str, Any]:
    return {
        "task": task,
        "feature_policy": feature_policy,
        "seed": seed,
        "model_name": model_name,
        "split_type": split_type,
        "train_size": train_size,
        "eval_size": eval_size,
        "resampled_train_size": resampled_train_size,
        "augmented_size": augmented_size,
        **collect_metrics(y_true, proba, task),
    }


def feature_importance_rows(bundle: ModelBundle, *, task: str, policy: str, seed: int) -> list[dict[str, Any]]:
    values: np.ndarray | None = None
    names = bundle.feature_names
    if bundle.bundle_type in {"xgb", "binary_causal"} and hasattr(bundle.model, "feature_importances_"):
        values = np.asarray(bundle.model.feature_importances_, dtype=float)
    elif bundle.bundle_type == "catboost" and hasattr(bundle.model, "get_feature_importance"):
        values = np.asarray(bundle.model.get_feature_importance(), dtype=float)
    elif bundle.bundle_type == "ordinal_three":
        imp0 = np.asarray(bundle.model["gt0"].feature_importances_, dtype=float)
        imp1 = np.asarray(bundle.model["gt1"].feature_importances_, dtype=float)
        values = (imp0 + imp1) / 2.0
    if values is None or len(values) != len(names):
        return []
    total = float(np.sum(values))
    if total > 0:
        values = values / total
    return [
        {
            "task": task,
            "feature_policy": policy,
            "seed": seed,
            "model_name": bundle.model_name,
            "feature": name,
            "importance": float(value),
        }
        for name, value in zip(names, values)
    ]


def aggregate_metrics(df: pd.DataFrame, group_cols: list[str], seed_base: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for idx, key_values in enumerate(df[group_cols].drop_duplicates().itertuples(index=False, name=None)):
        key = dict(zip(group_cols, key_values))
        group = df.copy()
        for col, value in key.items():
            group = group[group[col] == value]
        row: dict[str, Any] = key.copy()
        for metric in SUMMARY_METRICS + ["augmented_size", "resampled_train_size"]:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_var"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
                continue
            ci_low, ci_high = compute_bootstrap_ci(finite, seed=seed_base + idx * 31 + len(metric))
            row[f"{metric}_mean"] = float(finite.mean())
            row[f"{metric}_var"] = float(finite.var(ddof=0))
            row[f"{metric}_std"] = float(finite.std(ddof=0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def gap_rows_from_fold_metrics(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    *,
    task: str,
    policy: str,
    fold_count: int,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in GAP_METRICS:
        train_values = pd.to_numeric(pd.Series([row.get(metric) for row in train_rows]), errors="coerce").to_numpy(dtype=float)
        valid_values = pd.to_numeric(pd.Series([row.get(metric) for row in valid_rows]), errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(train_values) & np.isfinite(valid_values)
        if not np.any(mask):
            continue
        if metric == "log_loss":
            gaps = valid_values[mask] - train_values[mask]
        else:
            gaps = train_values[mask] - valid_values[mask]
        rows.append(
            {
                "task": task,
                "feature_policy": policy,
                "fold_count": fold_count,
                "model_name": model_name,
                "metric": metric,
                "train_mean": float(train_values[mask].mean()),
                "valid_mean": float(valid_values[mask].mean()),
                "generalization_gap_mean": float(gaps.mean()),
                "generalization_gap_std": float(gaps.std(ddof=0)),
                "overfit_risk_flag": bool(gaps.mean() > 0.05),
            }
        )
    return rows


def save_partial(tables_dir: Path, seed_rows: list[dict[str, Any]], cv_rows: list[dict[str, Any]]) -> None:
    if seed_rows:
        pd.DataFrame(seed_rows).to_csv(tables_dir / "screening_metrics_by_seed.partial.csv", index=False, encoding="utf-8-sig")
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(tables_dir / "cross_validation_by_fold.partial.csv", index=False, encoding="utf-8-sig")


def run_seed_suite(
    prepared_X: pd.DataFrame,
    prepared_y: pd.Series,
    *,
    tasks: list[str],
    policies: list[str],
    seeds: list[int],
    smoke: bool,
    include_causal_binary: bool,
    tracker: ProgressTracker,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    importance: list[dict[str, Any]] = []
    for seed in seeds:
        train_idx, test_idx = train_test_split(
            np.arange(len(prepared_y)),
            test_size=0.2,
            random_state=seed,
            stratify=prepared_y,
        )
        X_train_base = prepared_X.iloc[train_idx].reset_index(drop=True)
        X_test_base = prepared_X.iloc[test_idx].reset_index(drop=True)
        y_train_base = prepared_y.iloc[train_idx].reset_index(drop=True)
        y_test_base = prepared_y.iloc[test_idx].reset_index(drop=True)
        for policy in policies:
            view = apply_feature_policy(X_train_base, X_test_base, policy)
            for task in tasks:
                y_train = task_target(y_train_base, task)
                y_test = task_target(y_test_base, task)
                for model_name in model_names_for(task, policy, smoke, include_causal_binary):
                    tracker.log(
                        f"seed={seed} {task}/{policy}/{model_name}",
                        stage="seed_model",
                        context={"seed": seed, "task": task, "feature_policy": policy, "model_name": model_name},
                    )
                    bundle, augmented_size = fit_bundle_by_name(
                        model_name,
                        view.X_train,
                        y_train,
                        task=task,
                        seed=seed,
                    )
                    proba = predict_bundle(bundle, view.X_eval)
                    rows.append(
                        build_metric_row(
                            task=task,
                            feature_policy=policy,
                            seed=seed,
                            model_name=model_name,
                            y_true=y_test,
                            proba=proba,
                            train_size=bundle.train_size,
                            eval_size=len(y_test),
                            resampled_train_size=bundle.resampled_train_size,
                            augmented_size=augmented_size,
                            split_type="test",
                        )
                    )
                    importance.extend(feature_importance_rows(bundle, task=task, policy=policy, seed=seed))
                    save_partial(tables_dir, rows, [])
                    tracker.log(
                        f"done seed={seed} {task}/{policy}/{model_name}",
                        stage="seed_model_done",
                        advance=1,
                        context={"seed": seed, "task": task, "feature_policy": policy, "model_name": model_name},
                    )
    return pd.DataFrame(rows), pd.DataFrame(importance)


def run_cv_suite(
    prepared_X: pd.DataFrame,
    prepared_y: pd.Series,
    *,
    tasks: list[str],
    policies: list[str],
    folds: list[int],
    smoke: bool,
    include_causal_binary: bool,
    tracker: ProgressTracker,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for fold_count in folds:
        for policy in policies:
            for task in tasks:
                y_for_split = prepared_y if task == "three" else binary_target(prepared_y)
                cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42 + fold_count)
                for model_idx, model_name in enumerate(model_names_for(task, policy, smoke, include_causal_binary)):
                    fold_train_metrics: list[dict[str, Any]] = []
                    fold_valid_metrics: list[dict[str, Any]] = []
                    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(prepared_X, y_for_split), start=1):
                        X_train_base = prepared_X.iloc[train_idx].reset_index(drop=True)
                        X_valid_base = prepared_X.iloc[valid_idx].reset_index(drop=True)
                        y_train_base = prepared_y.iloc[train_idx].reset_index(drop=True)
                        y_valid_base = prepared_y.iloc[valid_idx].reset_index(drop=True)
                        y_train = task_target(y_train_base, task)
                        y_valid = task_target(y_valid_base, task)
                        view = apply_feature_policy(X_train_base, X_valid_base, policy)
                        seed = 10000 + fold_count * 100 + fold_idx + model_idx * 13
                        tracker.log(
                            f"cv={fold_count} fold={fold_idx} {task}/{policy}/{model_name}",
                            stage="cross_validation",
                            context={
                                "fold_count": fold_count,
                                "fold_index": fold_idx,
                                "task": task,
                                "feature_policy": policy,
                                "model_name": model_name,
                            },
                        )
                        bundle, augmented_size = fit_bundle_by_name(
                            model_name,
                            view.X_train,
                            y_train,
                            task=task,
                            seed=seed,
                        )
                        train_proba = predict_bundle(bundle, view.X_train_eval)
                        valid_proba = predict_bundle(bundle, view.X_eval)
                        train_metrics = collect_metrics(y_train, train_proba, task)
                        valid_metrics = collect_metrics(y_valid, valid_proba, task)
                        fold_train_metrics.append(train_metrics)
                        fold_valid_metrics.append(valid_metrics)
                        cv_rows.append(
                            {
                                "task": task,
                                "feature_policy": policy,
                                "fold_count": fold_count,
                                "fold_index": fold_idx,
                                "model_name": model_name,
                                "split_type": "valid",
                                "train_size": int(len(y_train)),
                                "valid_size": int(len(y_valid)),
                                "augmented_size": augmented_size,
                                "resampled_train_size": bundle.resampled_train_size,
                                **valid_metrics,
                            }
                        )
                        save_partial(tables_dir, [], cv_rows)
                    gap_rows.extend(
                        gap_rows_from_fold_metrics(
                            fold_train_metrics,
                            fold_valid_metrics,
                            task=task,
                            policy=policy,
                            fold_count=fold_count,
                            model_name=model_name,
                        )
                    )
                    tracker.log(
                        f"done cv={fold_count} {task}/{policy}/{model_name}",
                        stage="cross_validation_done",
                        advance=1,
                        context={"fold_count": fold_count, "task": task, "feature_policy": policy, "model_name": model_name},
                    )
    return pd.DataFrame(cv_rows), pd.DataFrame(gap_rows)


def build_post_dependency_audit(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if seed_summary.empty:
        return pd.DataFrame()
    key_cols = ["task", "model_name"]
    full_df = seed_summary[seed_summary["feature_policy"] == "full_reference"]
    stress_df = seed_summary[seed_summary["feature_policy"] == "post_mask_stress"]
    for _, full_row in full_df.iterrows():
        match = stress_df[
            (stress_df["task"] == full_row["task"])
            & (stress_df["model_name"] == full_row["model_name"])
        ]
        if match.empty:
            continue
        stress_row = match.iloc[0]
        row = {col: full_row[col] for col in key_cols}
        for metric in ["balanced_accuracy", "macro_f1", "accuracy", "class0_recall", "class1_recall", "class2_recall", "sensitivity", "specificity"]:
            full_value = full_row.get(f"{metric}_mean", np.nan)
            stress_value = stress_row.get(f"{metric}_mean", np.nan)
            row[f"{metric}_full_reference_mean"] = full_value
            row[f"{metric}_post_mask_mean"] = stress_value
            row[f"{metric}_drop"] = (
                float(full_value) - float(stress_value)
                if pd.notna(full_value) and pd.notna(stress_value)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ranking_summary(seed_summary: pd.DataFrame, cv_gap_df: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    gap_lookup: dict[tuple[str, str, str], float] = {}
    if not cv_gap_df.empty:
        ba_gap = cv_gap_df[cv_gap_df["metric"] == "balanced_accuracy"].copy()
        for _, row in ba_gap.iterrows():
            key = (row["task"], row["feature_policy"], row["model_name"])
            current = gap_lookup.get(key)
            gap = float(row["generalization_gap_mean"])
            if current is None or abs(gap) > abs(current):
                gap_lookup[key] = gap
    ranked = seed_summary.copy()
    ranked["generalization_gap_balanced_accuracy"] = [
        gap_lookup.get((row["task"], row["feature_policy"], row["model_name"]), float("nan"))
        for _, row in ranked.iterrows()
    ]
    return ranked.sort_values(
        ["task", "feature_policy", "balanced_accuracy_mean", "macro_f1_mean", "generalization_gap_balanced_accuracy"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)


def should_include_binary_causal(seed_summary: pd.DataFrame, forced: bool) -> tuple[bool, dict[str, Any]]:
    if forced:
        return True, {"forced": True}
    if seed_summary.empty:
        return False, {"reason": "no_seed_summary"}
    subset = seed_summary[
        (seed_summary["task"] == "binary")
        & (seed_summary["feature_policy"] == "screening_no_post")
        & (seed_summary["model_name"].isin(["xgb_binary_reference", "xgb_binary_scm_v2"]))
    ]
    if subset["model_name"].nunique() < 2:
        return False, {"reason": "missing_binary_reference_or_scm"}
    ref = subset[subset["model_name"] == "xgb_binary_reference"].iloc[0]
    scm = subset[subset["model_name"] == "xgb_binary_scm_v2"].iloc[0]
    ba_delta = float(scm["balanced_accuracy_mean"] - ref["balanced_accuracy_mean"])
    f1_delta = float(scm["macro_f1_mean"] - ref["macro_f1_mean"])
    include = ba_delta >= 0.02 or f1_delta >= 0.02
    return include, {"forced": False, "ba_delta": ba_delta, "macro_f1_delta": f1_delta, "threshold": 0.02}


def write_feature_policy_files(output_dir: Path, prepared_columns: list[str], policies: list[str]) -> None:
    policy_rows: list[dict[str, Any]] = []
    policy_json: dict[str, Any] = {}
    for policy in policies:
        included, dropped, masked = get_policy_columns(prepared_columns, policy)
        policy_json[policy] = {
            "included_columns": included,
            "dropped_columns": dropped,
            "masked_columns": masked,
            "included_count": len(included),
        }
        for col in included:
            policy_rows.append({"feature_policy": policy, "column": col, "status": "included"})
        for col in dropped:
            policy_rows.append({"feature_policy": policy, "column": col, "status": "dropped"})
        for col in masked:
            policy_rows.append({"feature_policy": policy, "column": col, "status": "masked_at_eval"})
    (output_dir / "feature_policy_columns.json").write_text(json.dumps(policy_json, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(policy_rows).to_csv(output_dir / "tables" / "feature_policy_columns.csv", index=False, encoding="utf-8-sig")


def write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    validation: dict[str, Any],
    tasks: list[str],
    policies: list[str],
    seed_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    gap_df: pd.DataFrame,
    binary_causal_decision: dict[str, Any],
) -> None:
    best_rows: list[dict[str, Any]] = []
    for task in tasks:
        subset = seed_summary[(seed_summary["task"] == task) & (seed_summary["feature_policy"] == "screening_no_post")]
        if subset.empty:
            continue
        best = subset.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=False).iloc[0]
        best_rows.append(
            {
                "task": task,
                "model_name": best["model_name"],
                "balanced_accuracy_mean": float(best["balanced_accuracy_mean"]),
                "macro_f1_mean": float(best["macro_f1_mean"]),
                "accuracy_mean": float(best["accuracy_mean"]),
            }
        )
    summary = {
        "input_file": str(args.input),
        "output_dir": str(output_dir),
        "smoke_test": bool(args.smoke_test),
        "tasks": tasks,
        "feature_policies": policies,
        "seeds": args.seeds,
        "cv_folds": args.cv_folds,
        "screening_policy_validation": validation,
        "binary_causal_decision": binary_causal_decision,
        "best_screening_no_post_by_task": best_rows,
        "tables": {
            "screening_metrics_by_seed": "tables/screening_metrics_by_seed.csv",
            "screening_metrics_mean_std": "tables/screening_metrics_mean_std.csv",
            "cross_validation_by_fold": "tables/cross_validation_by_fold.csv",
            "cross_validation_mean_var": "tables/cross_validation_mean_var.csv",
            "overfitting_indicators": "tables/overfitting_indicators.csv",
            "feature_importance_screening": "tables/feature_importance_screening.csv",
            "post_dependency_audit": "tables/post_dependency_audit.csv",
        },
        "row_counts": {
            "seed_summary_rows": int(len(seed_summary)),
            "cv_summary_rows": int(len(cv_summary)),
            "overfitting_rows": int(len(gap_df)),
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def estimate_total_steps(tasks: list[str], policies: list[str], seeds: list[int], folds: list[int], smoke: bool) -> int:
    seed_steps = 0
    cv_steps = 0
    for policy in policies:
        for task in tasks:
            seed_steps += len(seeds) * len(model_names_for(task, policy, smoke, include_causal_binary=True))
            cv_steps += len(folds) * len(model_names_for(task, policy, smoke, include_causal_binary=True))
    return max(seed_steps + cv_steps + 6, 1)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if args.smoke_test:
        args.feature_policy = ["screening_no_post"]
        args.cv_folds = [2]
        args.seeds = [args.seeds[0] if args.seeds else 42]
    tasks = requested_tasks(args.task_mode)
    policies = list(dict.fromkeys(args.feature_policy))
    folds = sorted({int(fold) for fold in args.cv_folds if int(fold) >= 2})
    seeds = list(dict.fromkeys(args.seeds))

    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    tracker = ProgressTracker(output_dir, total_steps=estimate_total_steps(tasks, policies, seeds, folds, args.smoke_test))
    tracker.log(
        "screening experiment started",
        stage="startup",
        context={"tasks": tasks, "feature_policies": policies, "seeds": seeds, "cv_folds": folds},
    )

    prepared = prepare_data(args.input)
    validation = assert_screening_policy(list(prepared.X.columns))
    write_feature_policy_files(output_dir, list(prepared.X.columns), policies)
    target_distribution = prepared.y.value_counts().sort_index().to_dict()
    binary_distribution = binary_target(prepared.y).value_counts().sort_index().to_dict()
    tracker.log(
        "data loaded and feature policies validated",
        stage="data",
        advance=1,
        context={
            "shape": list(prepared.X.shape),
            "target_distribution": {str(k): int(v) for k, v in target_distribution.items()},
            "binary_distribution": {str(k): int(v) for k, v in binary_distribution.items()},
            **validation,
        },
    )

    seed_df, importance_df = run_seed_suite(
        prepared.X,
        prepared.y,
        tasks=tasks,
        policies=policies,
        seeds=seeds,
        smoke=args.smoke_test,
        include_causal_binary=False,
        tracker=tracker,
        tables_dir=tables_dir,
    )
    seed_summary = aggregate_metrics(seed_df, ["task", "feature_policy", "model_name"], seed_base=31000)
    include_binary_causal, binary_decision = should_include_binary_causal(seed_summary, args.include_causal_binary)

    if include_binary_causal and not args.smoke_test and "binary" in tasks and "screening_no_post" in policies:
        tracker.log(
            "binary SCM cleared gain rule; running causal_binary_xgboost",
            stage="binary_causal_dispatch",
            context=binary_decision,
        )
        extra_seed_df, extra_importance_df = run_seed_suite(
            prepared.X,
            prepared.y,
            tasks=["binary"],
            policies=["screening_no_post"],
            seeds=seeds,
            smoke=False,
            include_causal_binary=True,
            tracker=tracker,
            tables_dir=tables_dir,
        )
        extra_seed_df = extra_seed_df[extra_seed_df["model_name"] == "causal_binary_xgboost"].copy()
        extra_importance_df = extra_importance_df[
            extra_importance_df["model_name"] == "causal_binary_xgboost"
        ].copy()
        seed_df = pd.concat([seed_df, extra_seed_df], axis=0, ignore_index=True)
        importance_df = pd.concat([importance_df, extra_importance_df], axis=0, ignore_index=True)
        seed_summary = aggregate_metrics(seed_df, ["task", "feature_policy", "model_name"], seed_base=31000)

    seed_df.to_csv(tables_dir / "screening_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    seed_summary = seed_summary.sort_values(
        ["task", "feature_policy", "balanced_accuracy_mean", "macro_f1_mean"],
        ascending=[True, True, False, False],
    )
    seed_summary.to_csv(tables_dir / "screening_metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    if not importance_df.empty:
        importance_df.to_csv(tables_dir / "feature_importance_screening.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["task", "feature_policy", "seed", "model_name", "feature", "importance"]).to_csv(
            tables_dir / "feature_importance_screening.csv",
            index=False,
            encoding="utf-8-sig",
        )
    tracker.log("seed suite summarized", stage="seed_summary", advance=1)

    cv_df, gap_df = run_cv_suite(
        prepared.X,
        prepared.y,
        tasks=tasks,
        policies=policies,
        folds=folds,
        smoke=args.smoke_test,
        include_causal_binary=include_binary_causal,
        tracker=tracker,
        tables_dir=tables_dir,
    )
    cv_summary = aggregate_metrics(cv_df, ["task", "feature_policy", "fold_count", "model_name"], seed_base=41000)
    cv_df.to_csv(tables_dir / "cross_validation_by_fold.csv", index=False, encoding="utf-8-sig")
    cv_summary = cv_summary.sort_values(
        ["task", "feature_policy", "fold_count", "balanced_accuracy_mean", "macro_f1_mean"],
        ascending=[True, True, True, False, False],
    )
    cv_summary.to_csv(tables_dir / "cross_validation_mean_var.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(tables_dir / "overfitting_indicators.csv", index=False, encoding="utf-8-sig")
    tracker.log("cross validation summarized", stage="cv_summary", advance=1)

    ranking = build_ranking_summary(seed_summary, gap_df)
    ranking.to_csv(tables_dir / "ranking_summary.csv", index=False, encoding="utf-8-sig")
    post_audit = build_post_dependency_audit(seed_summary)
    post_audit.to_csv(tables_dir / "post_dependency_audit.csv", index=False, encoding="utf-8-sig")
    tracker.log("ranking and dependency audit written", stage="audit", advance=1)

    write_summary(
        output_dir,
        args=args,
        validation=validation,
        tasks=tasks,
        policies=policies,
        seed_summary=seed_summary,
        cv_summary=cv_summary,
        gap_df=gap_df,
        binary_causal_decision=binary_decision,
    )
    tracker.log("experiment summary written", stage="summary", advance=1)
    tracker.finish()
    print(f"Screening experiment completed. Output: {output_dir}")


if __name__ == "__main__":
    main()
