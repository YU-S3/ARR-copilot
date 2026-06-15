from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, log_loss, matthews_corrcoef, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from causal_xgboost_variants_experiment import (
    MAINLINE_VARIANT_NAME,
    XGB_BEST_PARAMS,
    calibrate_with_temperature,
    collect_metric_row,
    evaluate_calibration_metrics,
    fit_custom_softprob_booster,
    fit_ordinal_xgboost,
    fit_preprocessor_and_transform,
    get_discrete_features,
    predict_custom_softprob_bundle,
    transform_with_fitted_preprocessor,
)
from frontier_scm_v2_experiment import SCMMixV2Augmentor, build_hard_case_index_map
from multiclass_ensemble_experiment import (
    RANDOM_STATE,
    TARGET_LABELS,
    apply_controlled_adasyn,
    build_preprocessor,
    evaluate_predictions,
    prepare_data,
    transform_with_preprocessor,
)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["axes.unicode_minus"] = False

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "joint_causal_scm_v2_outputs"
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]
CV_FOLD_OPTIONS = [5, 7, 10]
CLASSES = [0, 1, 2]
MODEL_ORDER = [
    "xgb_reference_raw",
    "xgb_reference_adasyn",
    "xgb_scm_v2_best",
    "causal_mainline_xgboost",
    "ordinal_mainline_xgboost",
    "causal_scm_v2_joint",
    "ordinal_scm_v2_joint",
]
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
]
CV_GAP_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "mcc",
    "quadratic_kappa",
    "top2_accuracy",
    "class0_recall",
]
SCM_V2_BEST_CONFIG = {
    "seed_strategy": "hard_case_seed",
    "target_classes": "class0_only",
    "treat_mix_prob": 0.4,
    "residual_scale": 0.5,
    "teacher_mode": "single",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="联合因果增强与 SCM-v2 实验脚本。")
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="是否在主实验后继续运行 5/7/10 折交叉验证。",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        nargs="*",
        default=CV_FOLD_OPTIONS,
        help="交叉验证折数列表，默认 5 7 10。",
    )
    parser.add_argument(
        "--cv-models",
        type=str,
        nargs="*",
        default=MODEL_ORDER,
        help="参与交叉验证的模型列表，默认全部模型。",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=DEFAULT_SEEDS,
        help="主实验随机种子列表，默认 42 2024 2025 2026 2027。",
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
        self.current_stage = "init"
        self.current_message = "初始化"
        self.current_context: dict[str, Any] = {}
        self._write_progress()

    def _snapshot(self) -> dict[str, Any]:
        elapsed_seconds = time.perf_counter() - self.start_time
        return {
            "stage": self.current_stage,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "progress_percent": round(self.completed_steps / self.total_steps * 100, 2),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "elapsed_human": format_seconds(elapsed_seconds),
            "current_message": self.current_message,
            "context": self.current_context,
        }

    def _write_progress(self) -> None:
        snapshot = self._snapshot()
        self.progress_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(
        self,
        message: str,
        *,
        stage: str | None = None,
        advance: int = 0,
        context: dict[str, Any] | None = None,
    ) -> None:
        if stage is not None:
            self.current_stage = stage
        self.current_message = message
        if context is not None:
            self.current_context = context
        if advance:
            self.completed_steps = min(self.total_steps, self.completed_steps + advance)

        snapshot = self._snapshot()
        line = (
            f"[{snapshot['elapsed_human']}] "
            f"{snapshot['completed_steps']}/{snapshot['total_steps']} "
            f"({snapshot['progress_percent']:.2f}%) | "
            f"{snapshot['stage']} | {snapshot['current_message']}"
        )
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._write_progress()


def make_xgb_classifier(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )


def extract_class_recall(metrics: dict[str, Any], class_name: str) -> float:
    return float(metrics.get("classification_report", {}).get(class_name, {}).get("recall", 0.0))


def compute_bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 4000) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, 1e-6, 1.0)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return proba / row_sum


def collect_prediction_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    normalized = normalize_proba(proba)
    pred_metrics = evaluate_predictions(y_true, normalized, TARGET_LABELS, CLASSES)
    calibration = evaluate_calibration_metrics(y_true, normalized)
    pred = np.argmax(normalized, axis=1)
    return {
        "accuracy": float(pred_metrics["accuracy"]),
        "balanced_accuracy": float(pred_metrics["balanced_accuracy"]),
        "macro_f1": float(pred_metrics["macro_f1"]),
        "weighted_f1": float(pred_metrics["weighted_f1"]),
        "ovr_roc_auc_macro": float(pred_metrics["ovr_roc_auc_macro"]),
        "ece": float(calibration["ece"]),
        "brier_score": float(calibration["brier_score"]),
        "log_loss": float(log_loss(y_true, normalized, labels=CLASSES)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "top2_accuracy": float(top_k_accuracy_score(y_true, normalized, k=min(2, normalized.shape[1]), labels=CLASSES)),
        "class0_recall": extract_class_recall(pred_metrics, "非确诊"),
        "class1_recall": extract_class_recall(pred_metrics, "确诊"),
        "class2_recall": extract_class_recall(pred_metrics, "灰色区域"),
    }


def prepare_split(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    prepared = prepare_data(PROJECT_DIR / "数据表格测试.xlsx")
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=seed,
        stratify=prepared.y,
    )
    return (
        x_train_raw.reset_index(drop=True),
        x_test_raw.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def fit_preprocessor_train_only(
    X_train_raw: pd.DataFrame,
) -> tuple[Any, pd.DataFrame]:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    transformed = preprocessor.fit_transform(X_train_raw)
    return (
        preprocessor,
        pd.DataFrame(
            transformed,
            columns=preprocessor.get_feature_names_out(),
            index=X_train_raw.index,
        ),
    )


def fit_xgb_from_raw(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    seed: int,
    use_adasyn: bool,
) -> dict[str, Any]:
    discrete_numeric_features = get_discrete_features(X_train_raw)
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, X_test_df = transform_with_preprocessor(preprocessor, X_train_raw, X_test_raw)

    resampled_train_size = int(len(X_train_df))
    y_fit = y_train.copy()
    if use_adasyn:
        X_train_df, y_fit = apply_controlled_adasyn(
            X_train_df,
            y_fit,
            discrete_numeric_features,
            seed,
        )
        resampled_train_size = int(len(X_train_df))

    model = make_xgb_classifier(seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    model.fit(X_train_df, y_fit, sample_weight=sample_weight, verbose=False)
    proba = np.asarray(model.predict_proba(X_test_df))
    return {
        "proba": proba,
        "train_size": int(len(X_train_raw)),
        "resampled_train_size": resampled_train_size,
    }


def fit_xgb_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    use_adasyn: bool,
) -> dict[str, Any]:
    discrete_numeric_features = get_discrete_features(X_train_raw)
    preprocessor, X_train_df = fit_preprocessor_train_only(X_train_raw)

    resampled_train_size = int(len(X_train_df))
    y_fit = y_train.copy()
    if use_adasyn:
        X_train_df, y_fit = apply_controlled_adasyn(
            X_train_df,
            y_fit,
            discrete_numeric_features,
            seed,
        )
        resampled_train_size = int(len(X_train_df))

    model = make_xgb_classifier(seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    model.fit(X_train_df, y_fit, sample_weight=sample_weight, verbose=False)
    return {
        "bundle_type": "xgb",
        "preprocessor": preprocessor,
        "model": model,
        "train_size": int(len(X_train_raw)),
        "resampled_train_size": resampled_train_size,
    }


def predict_with_bundle(bundle: dict[str, Any], X_raw: pd.DataFrame) -> np.ndarray:
    bundle_type = bundle["bundle_type"]
    if bundle_type == "xgb":
        transformed = bundle["preprocessor"].transform(X_raw)
        X_df = pd.DataFrame(
            transformed,
            columns=bundle["preprocessor"].get_feature_names_out(),
            index=X_raw.index,
        )
        return np.asarray(bundle["model"].predict_proba(X_df))
    if bundle_type == "causal":
        X_df = transform_with_fitted_preprocessor(bundle["preprocessor"], X_raw)
        prediction = predict_custom_softprob_bundle(bundle["model_bundle"], X_df)
        return np.asarray(prediction["standard_proba"])
    if bundle_type == "ordinal":
        X_df = transform_with_fitted_preprocessor(bundle["preprocessor"], X_raw)
        p_gt0 = bundle["model_gt0"].predict_proba(X_df)[:, 1]
        p_gt1 = bundle["model_gt1"].predict_proba(X_df)[:, 1]
        p_gt1 = np.minimum(p_gt1, p_gt0)
        p0 = 1.0 - p_gt0
        p2 = p_gt0 - p_gt1
        p1 = p_gt1
        return normalize_proba(np.column_stack([p0, p1, p2]))
    raise ValueError(f"未知 bundle_type: {bundle_type}")


def fit_causal_mainline_from_raw(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    _, X_train_df, X_test_df = fit_preprocessor_and_transform(X_train_raw, X_test_raw)
    discrete_numeric_features = get_discrete_features(X_train_raw)
    model_bundle, proba, severity = fit_custom_softprob_booster(
        X_train_df,
        X_test_df,
        y_train,
        discrete_numeric_features=discrete_numeric_features,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=seed,
    )
    return {
        "model_bundle": model_bundle,
        "proba": proba,
        "severity_score": severity,
        "train_size": int(len(X_train_raw)),
        "resampled_train_size": int(model_bundle["train_size"]),
    }


def fit_causal_mainline_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> dict[str, Any]:
    preprocessor, X_train_df = fit_preprocessor_train_only(X_train_raw)
    discrete_numeric_features = get_discrete_features(X_train_raw)
    model_bundle, _, _ = fit_custom_softprob_booster(
        X_train_df,
        X_train_df.copy(),
        y_train,
        discrete_numeric_features=discrete_numeric_features,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=seed,
    )
    return {
        "bundle_type": "causal",
        "preprocessor": preprocessor,
        "model_bundle": model_bundle,
        "train_size": int(len(X_train_raw)),
        "resampled_train_size": int(model_bundle["train_size"]),
    }


def fit_ordinal_mainline_from_raw(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
) -> dict[str, Any]:
    _, X_train_df, X_test_df = fit_preprocessor_and_transform(X_train_raw, X_test_raw)
    discrete_numeric_features = get_discrete_features(X_train_raw)
    model_bundle, proba = fit_ordinal_xgboost(
        X_train_df,
        X_test_df,
        y_train,
        discrete_numeric_features=discrete_numeric_features,
    )
    train_size = len(X_train_df)
    if discrete_numeric_features:
        X_fit, _ = apply_controlled_adasyn(
            X_train_df.copy(),
            y_train.copy(),
            discrete_numeric_features,
            42,
        )
        resampled_train_size = int(len(X_fit))
    else:
        resampled_train_size = int(train_size)
    return {
        "model_bundle": model_bundle,
        "proba": proba,
        "train_size": int(len(X_train_raw)),
        "resampled_train_size": resampled_train_size,
    }


def fit_ordinal_mainline_bundle(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    preprocessor, X_train_df = fit_preprocessor_train_only(X_train_raw)
    discrete_numeric_features = get_discrete_features(X_train_raw)
    model_bundle, _ = fit_ordinal_xgboost(
        X_train_df,
        X_train_df.copy(),
        y_train,
        discrete_numeric_features=discrete_numeric_features,
    )
    train_size = int(len(X_train_raw))
    if discrete_numeric_features:
        X_fit, _ = apply_controlled_adasyn(
            X_train_df.copy(),
            y_train.copy(),
            discrete_numeric_features,
            42,
        )
        resampled_train_size = int(len(X_fit))
    else:
        resampled_train_size = train_size
    return {
        "bundle_type": "ordinal",
        "preprocessor": preprocessor,
        "model_gt0": model_bundle["gt0"],
        "model_gt1": model_bundle["gt1"],
        "train_size": train_size,
        "resampled_train_size": resampled_train_size,
    }


def augment_with_best_scm_v2(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> dict[str, Any]:
    hard_case_map = build_hard_case_index_map(X_train_raw, y_train, seed)
    augmentor = SCMMixV2Augmentor(
        random_state=seed,
        seed_strategy=SCM_V2_BEST_CONFIG["seed_strategy"],
        target_classes=SCM_V2_BEST_CONFIG["target_classes"],
        treat_mix_prob=SCM_V2_BEST_CONFIG["treat_mix_prob"],
        residual_scale=SCM_V2_BEST_CONFIG["residual_scale"],
        teacher_mode=SCM_V2_BEST_CONFIG["teacher_mode"],
    )
    aug_result = augmentor.generate(X_train_raw, y_train, hard_case_map=hard_case_map)
    X_aug_train = pd.concat([X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
    y_aug_train = pd.concat([y_train, aug_result.y_aug], axis=0, ignore_index=True)
    return {
        "X_train_aug": X_aug_train,
        "y_train_aug": y_aug_train,
        "aug_result": aug_result,
    }


def build_metric_row(
    seed: int,
    model_name: str,
    y_test: pd.Series,
    proba: np.ndarray,
    *,
    train_size: int,
    resampled_train_size: int,
    augmented_size: int,
) -> dict[str, Any]:
    extended_metrics = collect_prediction_metrics(y_test, proba)
    return {
        "seed": seed,
        "model_name": model_name,
        "train_size": train_size,
        "resampled_train_size": resampled_train_size,
        "augmented_size": augmented_size,
        **extended_metrics,
    }


def aggregate_metrics(metrics_df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for idx, model_name in enumerate(model_names):
        model_df = metrics_df[metrics_df["model_name"] == model_name].copy()
        row: dict[str, Any] = {"model_name": model_name}
        for metric in SUMMARY_METRICS + ["augmented_size"]:
            values = model_df[metric].to_numpy(dtype=float)
            ci_low, ci_high = compute_bootstrap_ci(values, seed=6000 + idx * 17 + len(metric))
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        row["train_size_mean"] = float(model_df["train_size"].mean())
        row["resampled_train_size_mean"] = float(model_df["resampled_train_size"].mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def fit_model_bundle_by_name(
    model_name: str,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> tuple[dict[str, Any], int]:
    if model_name == "xgb_reference_raw":
        bundle = fit_xgb_bundle(X_train_raw, y_train, seed=seed, use_adasyn=False)
        return bundle, 0
    if model_name == "xgb_reference_adasyn":
        bundle = fit_xgb_bundle(X_train_raw, y_train, seed=seed, use_adasyn=True)
        return bundle, 0
    if model_name == "xgb_scm_v2_best":
        scm_bundle = augment_with_best_scm_v2(X_train_raw, y_train, seed=seed)
        augmented_size = int(len(scm_bundle["aug_result"].X_aug))
        bundle = fit_xgb_bundle(scm_bundle["X_train_aug"], scm_bundle["y_train_aug"], seed=seed, use_adasyn=True)
        return bundle, augmented_size
    if model_name == "causal_mainline_xgboost":
        bundle = fit_causal_mainline_bundle(X_train_raw, y_train, seed=seed)
        return bundle, 0
    if model_name == "ordinal_mainline_xgboost":
        bundle = fit_ordinal_mainline_bundle(X_train_raw, y_train)
        return bundle, 0
    if model_name == "causal_scm_v2_joint":
        scm_bundle = augment_with_best_scm_v2(X_train_raw, y_train, seed=seed)
        augmented_size = int(len(scm_bundle["aug_result"].X_aug))
        bundle = fit_causal_mainline_bundle(scm_bundle["X_train_aug"], scm_bundle["y_train_aug"], seed=seed + 200)
        return bundle, augmented_size
    if model_name == "ordinal_scm_v2_joint":
        scm_bundle = augment_with_best_scm_v2(X_train_raw, y_train, seed=seed)
        augmented_size = int(len(scm_bundle["aug_result"].X_aug))
        bundle = fit_ordinal_mainline_bundle(scm_bundle["X_train_aug"], scm_bundle["y_train_aug"])
        return bundle, augmented_size
    raise ValueError(f"未知模型名称: {model_name}")


def summarize_cv_table(cv_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, (fold_count, model_name) in enumerate(
        cv_df[["fold_count", "model_name"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        model_df = cv_df[(cv_df["fold_count"] == fold_count) & (cv_df["model_name"] == model_name)].copy()
        row: dict[str, Any] = {"fold_count": int(fold_count), "model_name": model_name}
        for metric in SUMMARY_METRICS + ["augmented_size", "resampled_train_size"]:
            values = model_df[metric].to_numpy(dtype=float)
            ci_low, ci_high = compute_bootstrap_ci(values, seed=12000 + idx * 29 + len(metric))
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_var"] = float(values.var(ddof=0))
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fold_count", "macro_f1_mean", "balanced_accuracy_mean"], ascending=[True, False, False])


def run_cross_validation_suite(
    tracker: ProgressTracker,
    output_dir: Path,
    fold_options: list[int],
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_data(PROJECT_DIR / "数据表格测试.xlsx")
    cv_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    cv_total = len(fold_options) * len(model_names)
    tracker.log(
        "开始配置 5/7/10 折交叉验证模块",
        stage="cross_validation_setup",
        context={"fold_options": fold_options, "model_names": model_names},
    )
    completed = 0
    for fold_count in fold_options:
        cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE + fold_count)
        for model_idx, model_name in enumerate(model_names):
            tracker.log(
                f"交叉验证：{fold_count} 折运行 {model_name}",
                stage="cross_validation",
                context={"fold_count": fold_count, "model_name": model_name, "progress_index": completed, "progress_total": cv_total},
            )
            fold_train_metric_rows: list[dict[str, float]] = []
            fold_valid_metric_rows: list[dict[str, float]] = []
            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(prepared.X, prepared.y), start=1):
                X_fold_train = prepared.X.iloc[train_idx].reset_index(drop=True)
                y_fold_train = prepared.y.iloc[train_idx].reset_index(drop=True)
                X_fold_valid = prepared.X.iloc[valid_idx].reset_index(drop=True)
                y_fold_valid = prepared.y.iloc[valid_idx].reset_index(drop=True)
                bundle, augmented_size = fit_model_bundle_by_name(
                    model_name,
                    X_fold_train,
                    y_fold_train,
                    seed=RANDOM_STATE + fold_count * 100 + fold_idx + model_idx * 7,
                )
                train_proba = predict_with_bundle(bundle, X_fold_train)
                valid_proba = predict_with_bundle(bundle, X_fold_valid)
                train_metrics = collect_prediction_metrics(y_fold_train, train_proba)
                valid_metrics = collect_prediction_metrics(y_fold_valid, valid_proba)
                fold_train_metric_rows.append(train_metrics)
                fold_valid_metric_rows.append(valid_metrics)
                cv_rows.append(
                    {
                        "fold_count": fold_count,
                        "fold_index": fold_idx,
                        "model_name": model_name,
                        "split_type": "valid",
                        "train_size": int(len(X_fold_train)),
                        "valid_size": int(len(X_fold_valid)),
                        "augmented_size": augmented_size,
                        "resampled_train_size": int(bundle["resampled_train_size"]),
                        **valid_metrics,
                    }
                )
            for metric in CV_GAP_METRICS:
                train_values = np.asarray([row[metric] for row in fold_train_metric_rows], dtype=float)
                valid_values = np.asarray([row[metric] for row in fold_valid_metric_rows], dtype=float)
                gap = train_values - valid_values
                gap_rows.append(
                    {
                        "fold_count": fold_count,
                        "model_name": model_name,
                        "metric": metric,
                        "train_mean": float(train_values.mean()),
                        "valid_mean": float(valid_values.mean()),
                        "generalization_gap_mean": float(gap.mean()),
                        "generalization_gap_std": float(gap.std(ddof=0)),
                        "overfit_risk_flag": bool(
                            (metric != "log_loss" and gap.mean() > 0.03)
                            or (metric == "log_loss" and gap.mean() < -0.03)
                        ),
                    }
                )
            completed += 1
    cv_df = pd.DataFrame(cv_rows)
    cv_summary_df = summarize_cv_table(cv_df)
    gap_df = pd.DataFrame(gap_rows)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(tables_dir / "cross_validation_by_fold.csv", index=False, encoding="utf-8-sig")
    cv_summary_df.to_csv(tables_dir / "cross_validation_mean_var.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(tables_dir / "overfitting_indicators.csv", index=False, encoding="utf-8-sig")
    return cv_df, gap_df


def plot_metric_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df[
        [
            "model_name",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "class0_recall_mean",
        ]
    ].melt(id_vars="model_name", var_name="metric", value_name="score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model_name")
    plt.ylim(0, 1.05)
    plt.title("联合实验主指标对比")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df[["model_name", "ece_mean", "brier_score_mean"]].melt(
        id_vars="model_name",
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model_name")
    plt.title("联合实验概率质量对比")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_joint_temperature_calibration(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> list[dict[str, Any]]:
    X_fit_raw, X_calib_raw, y_fit, y_calib = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.25,
        random_state=seed + 500,
        stratify=y_train,
    )

    aug_bundle = augment_with_best_scm_v2(
        X_fit_raw.reset_index(drop=True),
        y_fit.reset_index(drop=True),
        seed=seed + 700,
    )
    X_joint_fit_raw = aug_bundle["X_train_aug"]
    y_joint_fit = aug_bundle["y_train_aug"]
    augmented_size = int(len(aug_bundle["aug_result"].X_aug))

    preprocessor, X_joint_fit_df, X_calib_df = fit_preprocessor_and_transform(
        X_joint_fit_raw,
        X_calib_raw.reset_index(drop=True),
    )
    X_test_df = transform_with_fitted_preprocessor(preprocessor, X_test_raw)
    discrete_numeric_features = get_discrete_features(X_joint_fit_raw)
    model_bundle, _, _ = fit_custom_softprob_booster(
        X_joint_fit_df,
        X_calib_df,
        y_joint_fit,
        discrete_numeric_features=discrete_numeric_features,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=seed + 900,
    )
    calib_prediction = predict_custom_softprob_bundle(model_bundle, X_calib_df)
    test_prediction = predict_custom_softprob_bundle(model_bundle, X_test_df)
    temp_artifact, temp_proba = calibrate_with_temperature(
        calib_prediction["raw_margin"],
        y_calib.reset_index(drop=True),
        test_prediction["raw_margin"],
    )

    rows = [
        {
            "seed": seed,
            **collect_metric_row("causal_scm_v2_joint_uncalibrated", y_test, test_prediction["standard_proba"]),
            "train_fit_size": int(len(y_fit)),
            "calibration_size": int(len(y_calib)),
            "augmented_size": augmented_size,
            "temperature": 1.0,
        },
        {
            "seed": seed,
            **collect_metric_row("causal_scm_v2_joint_temperature_scaling", y_test, temp_proba),
            "train_fit_size": int(len(y_fit)),
            "calibration_size": int(len(y_calib)),
            "augmented_size": augmented_size,
            "temperature": float(temp_artifact["temperature"]),
        },
    ]
    return rows


def aggregate_calibration_rows(calibration_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(calibration_df["variant"].drop_duplicates().tolist()):
        variant_df = calibration_df[calibration_df["variant"] == variant].copy()
        row: dict[str, Any] = {"variant": variant}
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "ovr_roc_auc_macro",
            "ece",
            "brier_score",
            "temperature",
            "augmented_size",
        ]:
            values = variant_df[metric].to_numpy(dtype=float)
            ci_low, ci_high = compute_bootstrap_ci(values, seed=8000 + idx * 23 + len(metric))
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        row["train_fit_size_mean"] = float(variant_df["train_fit_size"].mean())
        row["calibration_size_mean"] = float(variant_df["calibration_size"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ece_mean", "brier_score_mean", "macro_f1_mean"], ascending=[True, True, False])


def write_markdown_report(
    summary_df: pd.DataFrame,
    calibration_summary_df: pd.DataFrame,
    cv_available: bool,
    output_path: Path,
) -> None:
    rank_df = summary_df.sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)
    best_name = str(rank_df.iloc[0]["model_name"])
    report = f"""# 联合因果增强 + SCM-v2 实验报告

## 1. 实验定位

本轮实验将两个既有主线的最优选择合并验证：

- 因果增强 XGBoost 主线：`{MAINLINE_VARIANT_NAME}`
- 增广主线：`scm_v2_hard_c0_tm40_rs50_teacher_single`

并以原始 `XGBoost` 作为 baseline，统一在 `5` 个随机种子下做综合比较。

## 2. 对比组

- `xgb_reference_raw`
- `xgb_reference_adasyn`
- `xgb_scm_v2_best`
- `causal_mainline_xgboost`
- `causal_scm_v2_joint`

## 3. 主结果汇总

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | 0类 Recall | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in rank_df.iterrows():
        report += (
            f"| `{row['model_name']}` | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} | "
            f"{row['macro_f1_mean']:.4f} | "
            f"{row['class0_recall_mean']:.4f} | "
            f"{row['ece_mean']:.4f} | "
            f"{row['brier_score_mean']:.4f} |\n"
        )

    report += f"""

## 4. 联合模型校准子实验

| 方案 | Accuracy | Balanced Accuracy | Macro F1 | ECE | Brier | 温度均值 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in calibration_summary_df.iterrows():
        report += (
            f"| `{row['variant']}` | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} | "
            f"{row['macro_f1_mean']:.4f} | "
            f"{row['ece_mean']:.4f} | "
            f"{row['brier_score_mean']:.4f} | "
            f"{row['temperature_mean']:.4f} |\n"
        )

    report += f"""

## 5. 扩展验证

- 代码已支持 `5/7/10` 折交叉验证结果输出：
  - `joint_causal_scm_v2_outputs/tables/cross_validation_by_fold.csv`
  - `joint_causal_scm_v2_outputs/tables/cross_validation_mean_var.csv`
  - `joint_causal_scm_v2_outputs/tables/overfitting_indicators.csv`
- 交叉验证结果中将报告均值、方差、标准差和 bootstrap 置信区间
- 过拟合指标将基于 train/valid 的泛化差距给出 `overfit_risk_flag`
- 当前报告写入时，交叉验证结果是否已完成：`{"是" if cv_available else "否"}`

## 6. 当前结论

- 综合主结果第一名：`{best_name}`
- 若联合模型的温度校准版本在 `ECE/Brier` 上更优，则可作为部署候选
- 本轮结果文件已写入 `joint_causal_scm_v2_outputs/`
"""
    output_path.write_text(report, encoding="utf-8")


def save_partial_outputs(
    tables_dir: Path,
    metric_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> None:
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(
            tables_dir / "metrics_by_seed.partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            tables_dir / "joint_calibration_by_seed.partial.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    args = parse_args()
    seed_list = args.seeds or DEFAULT_SEEDS
    cv_folds = sorted({int(fold) for fold in (args.cv_folds or CV_FOLD_OPTIONS) if int(fold) >= 2})
    cv_models = [name for name in (args.cv_models or MODEL_ORDER) if name in MODEL_ORDER]
    if not cv_models:
        cv_models = MODEL_ORDER.copy()
    tables_dir = OUTPUT_DIR / "tables"
    figures_dir = OUTPUT_DIR / "figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    total_steps = len(seed_list) * 8 + 4
    if args.run_cv:
        total_steps += len(cv_folds) * len(cv_models)
    tracker = ProgressTracker(OUTPUT_DIR, total_steps=total_steps)
    tracker.log(
        "联合实验启动，准备运行 baseline / 增广 / 因果 / 联合 / 校准子实验",
        stage="init",
        context={"seeds": seed_list, "run_cv": bool(args.run_cv), "cv_folds": cv_folds, "cv_models": cv_models},
    )

    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for seed_idx, seed in enumerate(seed_list, start=1):
        tracker.log(
            f"开始处理随机种子 {seed}（{seed_idx}/{len(seed_list)}）",
            stage="seed_start",
            context={"seed": seed, "seed_index": seed_idx},
        )
        X_train_raw, X_test_raw, y_train, y_test = prepare_split(seed)

        tracker.log(
            f"seed={seed}：运行 xgb_reference_raw",
            stage="baseline_raw",
            context={"seed": seed, "model_name": "xgb_reference_raw"},
        )
        raw_result = fit_xgb_from_raw(
            X_train_raw,
            y_train,
            X_test_raw,
            seed=seed,
            use_adasyn=False,
        )
        metric_rows.append(
            build_metric_row(
                seed,
                "xgb_reference_raw",
                y_test,
                raw_result["proba"],
                train_size=raw_result["train_size"],
                resampled_train_size=raw_result["resampled_train_size"],
                augmented_size=0,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成 xgb_reference_raw",
            stage="baseline_raw",
            advance=1,
            context={"seed": seed, "model_name": "xgb_reference_raw"},
        )

        tracker.log(
            f"seed={seed}：运行 xgb_reference_adasyn",
            stage="baseline_adasyn",
            context={"seed": seed, "model_name": "xgb_reference_adasyn"},
        )
        adasyn_result = fit_xgb_from_raw(
            X_train_raw,
            y_train,
            X_test_raw,
            seed=seed,
            use_adasyn=True,
        )
        metric_rows.append(
            build_metric_row(
                seed,
                "xgb_reference_adasyn",
                y_test,
                adasyn_result["proba"],
                train_size=adasyn_result["train_size"],
                resampled_train_size=adasyn_result["resampled_train_size"],
                augmented_size=0,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成 xgb_reference_adasyn",
            stage="baseline_adasyn",
            advance=1,
            context={"seed": seed, "model_name": "xgb_reference_adasyn"},
        )

        tracker.log(
            f"seed={seed}：生成 SCM-v2 最优增广样本",
            stage="augment_scm_v2",
            context={"seed": seed, "config": SCM_V2_BEST_CONFIG},
        )
        scm_bundle = augment_with_best_scm_v2(X_train_raw, y_train, seed=seed)
        X_train_aug = scm_bundle["X_train_aug"]
        y_train_aug = scm_bundle["y_train_aug"]
        augmented_size = int(len(scm_bundle["aug_result"].X_aug))
        tracker.log(
            f"seed={seed}：SCM-v2 增广完成，新增样本 {augmented_size}",
            stage="augment_scm_v2",
            context={"seed": seed, "augmented_size": augmented_size},
        )

        tracker.log(
            f"seed={seed}：运行 xgb_scm_v2_best",
            stage="scm_only",
            context={"seed": seed, "model_name": "xgb_scm_v2_best", "augmented_size": augmented_size},
        )
        scm_result = fit_xgb_from_raw(
            X_train_aug,
            y_train_aug,
            X_test_raw,
            seed=seed,
            use_adasyn=True,
        )
        metric_rows.append(
            build_metric_row(
                seed,
                "xgb_scm_v2_best",
                y_test,
                scm_result["proba"],
                train_size=scm_result["train_size"],
                resampled_train_size=scm_result["resampled_train_size"],
                augmented_size=augmented_size,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成 xgb_scm_v2_best",
            stage="scm_only",
            advance=1,
            context={"seed": seed, "model_name": "xgb_scm_v2_best", "augmented_size": augmented_size},
        )

        tracker.log(
            f"seed={seed}：运行 causal_mainline_xgboost",
            stage="causal_only",
            context={"seed": seed, "model_name": "causal_mainline_xgboost"},
        )
        causal_result = fit_causal_mainline_from_raw(
            X_train_raw,
            y_train,
            X_test_raw,
            seed=seed + 100,
        )
        metric_rows.append(
            build_metric_row(
                seed,
                "causal_mainline_xgboost",
                y_test,
                causal_result["proba"],
                train_size=causal_result["train_size"],
                resampled_train_size=causal_result["resampled_train_size"],
                augmented_size=0,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成 causal_mainline_xgboost",
            stage="causal_only",
            advance=1,
            context={"seed": seed, "model_name": "causal_mainline_xgboost"},
        )

        tracker.log(
            f"seed={seed}：运行 ordinal_mainline_xgboost",
            stage="ordinal_only",
            context={"seed": seed, "model_name": "ordinal_mainline_xgboost"},
        )
        ordinal_result = fit_ordinal_mainline_from_raw(
            X_train_raw,
            y_train,
            X_test_raw,
        )
        metric_rows.append(
            build_metric_row(
                seed,
                "ordinal_mainline_xgboost",
                y_test,
                ordinal_result["proba"],
                train_size=ordinal_result["train_size"],
                resampled_train_size=ordinal_result["resampled_train_size"],
                augmented_size=0,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成 ordinal_mainline_xgboost",
            stage="ordinal_only",
            advance=1,
            context={"seed": seed, "model_name": "ordinal_mainline_xgboost"},
        )

        tracker.log(
            f"seed={seed}：运行 causal_scm_v2_joint",
            stage="joint_model",
            context={"seed": seed, "model_name": "causal_scm_v2_joint", "augmented_size": augmented_size},
        )
        joint_result = fit_causal_mainline_from_raw(
            X_train_aug,
            y_train_aug,
            X_test_raw,
            seed=seed + 200,
        )
        metric_rows.append(
            build_metric_row(
                seed,
                "causal_scm_v2_joint",
                y_test,
                joint_result["proba"],
                train_size=joint_result["train_size"],
                resampled_train_size=joint_result["resampled_train_size"],
                augmented_size=augmented_size,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成 causal_scm_v2_joint",
            stage="joint_model",
            advance=1,
            context={"seed": seed, "model_name": "causal_scm_v2_joint", "augmented_size": augmented_size},
        )

        tracker.log(
            f"seed={seed}：运行 ordinal_scm_v2_joint",
            stage="ordinal_joint_model",
            context={"seed": seed, "model_name": "ordinal_scm_v2_joint", "augmented_size": augmented_size},
        )
        ordinal_joint_result = fit_ordinal_mainline_from_raw(
            X_train_aug,
            y_train_aug,
            X_test_raw,
        )
        metric_rows.append(
            build_metric_row(
                seed,
                "ordinal_scm_v2_joint",
                y_test,
                ordinal_joint_result["proba"],
                train_size=ordinal_joint_result["train_size"],
                resampled_train_size=ordinal_joint_result["resampled_train_size"],
                augmented_size=augmented_size,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成 ordinal_scm_v2_joint",
            stage="ordinal_joint_model",
            advance=1,
            context={"seed": seed, "model_name": "ordinal_scm_v2_joint", "augmented_size": augmented_size},
        )

        tracker.log(
            f"seed={seed}：运行联合模型温度校准子实验",
            stage="joint_calibration",
            context={"seed": seed},
        )
        calibration_rows.extend(
            run_joint_temperature_calibration(
                X_train_raw,
                y_train,
                X_test_raw,
                y_test,
                seed=seed,
            )
        )
        save_partial_outputs(tables_dir, metric_rows, calibration_rows)
        tracker.log(
            f"seed={seed}：完成联合模型温度校准子实验",
            stage="joint_calibration",
            advance=2,
            context={"seed": seed},
        )

    tracker.log("全部随机种子运行完成，开始汇总结果", stage="aggregate")
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(tables_dir / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")

    summary_df = aggregate_metrics(metrics_df, MODEL_ORDER).sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=False,
    )
    summary_df.to_csv(tables_dir / "metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(tables_dir / "joint_calibration_by_seed.csv", index=False, encoding="utf-8-sig")
    calibration_summary_df = aggregate_calibration_rows(calibration_df)
    calibration_summary_df.to_csv(
        tables_dir / "joint_calibration_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tracker.log("结果汇总表已写入 tables", stage="aggregate", advance=1)

    plot_metric_comparison(summary_df, figures_dir / "joint_main_metrics.png")
    plot_calibration_comparison(summary_df, figures_dir / "joint_probability_quality.png")
    tracker.log("图表已生成", stage="figures", advance=1)

    cv_available = False
    if args.run_cv:
        tracker.log("开始运行交叉验证模块", stage="cross_validation_dispatch")
        _, gap_df = run_cross_validation_suite(
            tracker=tracker,
            output_dir=OUTPUT_DIR,
            fold_options=cv_folds,
            model_names=cv_models,
        )
        cv_available = True
        tracker.log(
            "交叉验证结果已写入 tables",
            stage="cross_validation_done",
            context={
                "cv_folds": cv_folds,
                "cv_models": cv_models,
                "overfit_flags": int(gap_df["overfit_risk_flag"].sum()) if not gap_df.empty else 0,
            },
        )

    best_row = summary_df.iloc[0]
    summary = {
        "seeds": seed_list,
        "cross_validation_fold_options": cv_folds,
        "scm_v2_best_config": SCM_V2_BEST_CONFIG,
        "causal_mainline_variant": MAINLINE_VARIANT_NAME,
        "best_overall_model": str(best_row["model_name"]),
        "best_overall_metrics": {
            "accuracy_mean": float(best_row["accuracy_mean"]),
            "balanced_accuracy_mean": float(best_row["balanced_accuracy_mean"]),
            "macro_f1_mean": float(best_row["macro_f1_mean"]),
            "class0_recall_mean": float(best_row["class0_recall_mean"]),
            "ece_mean": float(best_row["ece_mean"]),
            "brier_score_mean": float(best_row["brier_score_mean"]),
        },
        "joint_calibration_best": calibration_summary_df.iloc[0].to_dict(),
        "extended_metrics_enabled": SUMMARY_METRICS,
        "cross_validation_outputs_ready": cv_available,
        "cross_validation_model_subset": cv_models,
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        summary_df,
        calibration_summary_df,
        cv_available=cv_available,
        output_path=PROJECT_DIR / "联合因果增强与SCM_v2实验报告.md",
    )
    tracker.log(
        "联合实验全部完成",
        stage="done",
        advance=2,
        context={
            "best_overall_model": str(best_row["model_name"]),
            "output_dir": str(OUTPUT_DIR),
        },
    )

    print("联合实验完成")
    print(summary_df.to_string(index=False))
    print()
    print(calibration_summary_df.to_string(index=False))
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
