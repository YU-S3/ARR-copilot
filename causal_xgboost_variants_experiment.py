from __future__ import annotations

import joblib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from multiclass_ensemble_experiment import (
    EXPERIMENT_TYPE_COLUMN,
    TARGET_COLUMN,
    TARGET_LABELS,
    ZERO_FILL_COLUMNS,
    apply_controlled_adasyn,
    build_preprocessor,
    clean_string_cell,
    detect_discrete_numeric_features,
    evaluate_predictions,
    maybe_convert_object_to_numeric,
    normalize_experiment_type,
    plot_confusion_heatmap,
    plot_multiclass_roc,
    prepare_data,
    transform_with_preprocessor,
)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42
MAINLINE_VARIANT_NAME = "ordinal_emd_fw_gpl_xgboost"
ORDERED_CLASSES = [0, 2, 1]
CLASS_TO_RANK = {cls: idx for idx, cls in enumerate(ORDERED_CLASSES)}
RANK_TO_STANDARD_INDEX = [0, 2, 1]
FW_GPL_SOFT_LABELS = {
    0: np.asarray([0.85, 0.15, 0.00], dtype=float),
    2: np.asarray([0.15, 0.70, 0.15], dtype=float),
    1: np.asarray([0.00, 0.15, 0.85], dtype=float),
}
CHAIN_COST_MATRIX = np.asarray(
    [
        [0.0, 1.0, 2.4],
        [1.1, 0.0, 1.2],
        [2.8, 1.0, 0.0],
    ],
    dtype=float,
)
BOUNDARY_WEIGHT_LOOKUP = {
    0: np.asarray([1.00, 1.10], dtype=float),
    2: np.asarray([1.20, 1.20], dtype=float),
    1: np.asarray([1.65, 2.10], dtype=float),
}
XGB_BEST_PARAMS = {
    "n_estimators": 276,
    "max_depth": 5,
    "learning_rate": 0.010573268083515799,
    "subsample": 0.9909729556485982,
    "colsample_bytree": 0.9497327922401265,
    "min_child_weight": 2,
    "reg_alpha": 0.0035113563139704067,
    "reg_lambda": 0.03549878832196503,
}
TREATMENT_COLUMNS = [
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
]
POST_TREATMENT_COLUMNS = [
    "试验前醛固酮",
    "试验前肾素",
    "试验后醛固酮",
    "试验后肾素",
]
PHYSIOLOGY_CORE_COLUMNS = {
    "ARR比值",
    "醛固酮",
    "肾素",
    "钾",
    "钠",
    "氯",
    "收缩压",
    "舒展压",
    "年龄",
    "是否有肾上腺结节",
    "是否有增生",
    "结节最大直径",
}
EXTERNAL_TEST_COLUMN_MAP = {
    "舒张压": "舒展压",
    "二氢吡啶类": "二氢吡啶类_等效分数",
    "非二氢吡啶类": "非二氢吡啶类_等效分数",
    "醛固酮（初筛）": "醛固酮",
    "肾素（初筛）": "肾素",
    "ARR数值": "ARR比值",
    "确诊实验（卡托普利/盐水负荷）": "确诊实验类型",
    "试验前醛固酮": "试验前醛固酮",
    "试验后醛固酮": "试验后醛固酮",
}


def make_xgb_classifier(seed: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )


def make_xgb_regressor(seed: int = RANDOM_STATE, n_estimators: int = 180) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=0.2,
        n_jobs=-1,
    )


def parse_args() -> tuple[Path, Path]:
    project_dir = Path(__file__).resolve().parent
    return (
        project_dir / "data_0428.xlsx",
        project_dir / "causal_xgboost_outputs_v2",
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def preprocess_pair(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, X_train_df, X_test_df = fit_preprocessor_and_transform(X_train_raw, X_test_raw)
    return X_train_df, X_test_df


def drop_columns_if_present(X_raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in X_raw.columns]
    return X_raw.drop(columns=existing).copy()


def build_strict_prospective_view(X_raw: pd.DataFrame) -> pd.DataFrame:
    return drop_columns_if_present(X_raw, POST_TREATMENT_COLUMNS)


def fit_preprocessor_and_transform(
    X_train_raw: pd.DataFrame,
    X_apply_raw: pd.DataFrame,
) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, X_apply_df = transform_with_preprocessor(preprocessor, X_train_raw, X_apply_raw)
    return preprocessor, X_train_df, X_apply_df


def transform_with_fitted_preprocessor(
    preprocessor: ColumnTransformer,
    X_raw: pd.DataFrame,
) -> pd.DataFrame:
    transformed = preprocessor.transform(X_raw)
    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=X_raw.index,
    )


def get_discrete_features(raw_df: pd.DataFrame) -> list[str]:
    numeric_features = raw_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return detect_discrete_numeric_features(raw_df, numeric_features)


def fit_baseline_xgb(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    X_test_df: pd.DataFrame,
    discrete_numeric_features: list[str] | None = None,
) -> tuple[Any, np.ndarray]:
    if discrete_numeric_features:
        X_train_df, y_train = apply_controlled_adasyn(
            X_train_df,
            y_train,
            discrete_numeric_features,
            RANDOM_STATE,
        )
    model = make_xgb_classifier(RANDOM_STATE)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_df, y_train, sample_weight=sample_weight, verbose=False)
    return model, model.predict_proba(X_test_df)


def raw_feature_name(transformed_name: str) -> str:
    if transformed_name.startswith("num__missingindicator_"):
        return transformed_name.replace("num__missingindicator_", "", 1)
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)
    if transformed_name.startswith("cat__"):
        return transformed_name.replace("cat__", "", 1)
    return transformed_name


def build_adaptive_feature_weights(feature_names: list[str]) -> np.ndarray:
    weights: list[float] = []
    for name in feature_names:
        raw_name = raw_feature_name(name)
        if raw_name in PHYSIOLOGY_CORE_COLUMNS:
            weight = 2.5
        elif raw_name in TREATMENT_COLUMNS:
            weight = 0.45
        elif raw_name in POST_TREATMENT_COLUMNS:
            weight = 0.55
        elif "missingindicator" in name:
            weight = 0.9
        else:
            weight = 1.1
        weights.append(weight)
    return np.asarray(weights, dtype=float)


def fit_weighted_xgb_booster(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    X_test_df: pd.DataFrame,
    feature_weights: np.ndarray | None = None,
    adaptive: bool = False,
    discrete_numeric_features: list[str] | None = None,
) -> tuple[xgb.Booster, np.ndarray]:
    if discrete_numeric_features:
        X_train_df, y_train = apply_controlled_adasyn(
            X_train_df,
            y_train,
            discrete_numeric_features,
            RANDOM_STATE,
        )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": RANDOM_STATE,
        "max_depth": XGB_BEST_PARAMS["max_depth"],
        "learning_rate": XGB_BEST_PARAMS["learning_rate"],
        "subsample": XGB_BEST_PARAMS["subsample"],
        "colsample_bytree": 0.72 if adaptive else XGB_BEST_PARAMS["colsample_bytree"],
        "colsample_bylevel": 0.72 if adaptive else 1.0,
        "min_child_weight": XGB_BEST_PARAMS["min_child_weight"],
        "reg_alpha": XGB_BEST_PARAMS["reg_alpha"],
        "reg_lambda": XGB_BEST_PARAMS["reg_lambda"],
    }
    dtrain = xgb.DMatrix(
        X_train_df,
        label=y_train.to_numpy(),
        weight=sample_weight,
        feature_weights=feature_weights,
    )
    dtest = xgb.DMatrix(X_test_df, feature_weights=feature_weights)
    booster = xgb.train(params, dtrain, num_boost_round=XGB_BEST_PARAMS["n_estimators"])
    proba = booster.predict(dtest)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 3)
    return booster, np.asarray(proba)


def reshape_logits(predt: np.ndarray, num_class: int = 3) -> np.ndarray:
    if predt.ndim == 2:
        return predt
    return predt.reshape(-1, num_class)


def softmax(logits: np.ndarray) -> np.ndarray:
    stabilized = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(stabilized)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def build_rank_targets(y: pd.Series) -> np.ndarray:
    return y.map(CLASS_TO_RANK).to_numpy(dtype=int)


def build_ordinal_binary_targets(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ranks = build_rank_targets(y)
    return (ranks > 0).astype(int), (ranks > 1).astype(int)


def build_soft_labels_fw_gpl(y: pd.Series) -> np.ndarray:
    return np.vstack([FW_GPL_SOFT_LABELS[int(label)] for label in y.to_numpy()])


def compute_continuous_severity_score(rank_proba: np.ndarray) -> np.ndarray:
    ranks = np.arange(rank_proba.shape[1], dtype=float)
    return (rank_proba * ranks).sum(axis=1) / float(rank_proba.shape[1] - 1)


def reorder_rank_proba_to_standard(rank_proba: np.ndarray) -> np.ndarray:
    proba = rank_proba[:, RANK_TO_STANDARD_INDEX]
    proba = np.clip(proba, 1e-6, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def evaluate_calibration_metrics(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    y_arr = y_true.to_numpy()
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
        if idx == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        ece += abs(float(correctness[mask].mean()) - float(confidence[mask].mean())) * float(mask.mean())
    return {"ece": float(ece), "brier_score": brier}


def evaluate_severity_score(y_true: pd.Series, severity_score: np.ndarray) -> dict[str, float]:
    df = pd.DataFrame({"label": y_true.to_numpy(), "severity_score": severity_score})
    summary: dict[str, float] = {}
    class_means: dict[int, float] = {}
    for cls in [0, 2, 1]:
        cls_mean = float(df.loc[df["label"] == cls, "severity_score"].mean())
        summary[f"class_{cls}_mean_severity"] = cls_mean
        class_means[cls] = cls_mean
    summary["severity_monotonic"] = float(class_means[0] < class_means[2] < class_means[1])
    summary["severity_gap_0_to_2"] = class_means[2] - class_means[0]
    summary["severity_gap_2_to_1"] = class_means[1] - class_means[2]
    return summary


def clinical_boundary_weights_from_soft_targets(soft_targets_rank: np.ndarray) -> np.ndarray:
    lookup = np.vstack(
        [
            BOUNDARY_WEIGHT_LOOKUP[0],
            BOUNDARY_WEIGHT_LOOKUP[2],
            BOUNDARY_WEIGHT_LOOKUP[1],
        ]
    )
    return soft_targets_rank @ lookup


def make_soft_ce_objective(
    soft_targets_rank: np.ndarray,
    lambda_ce: float = 1.0,
):
    def _obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        logits = reshape_logits(predt, num_class=soft_targets_rank.shape[1])
        probs = softmax(logits)
        grad = lambda_ce * (probs - soft_targets_rank)
        hess = np.maximum(lambda_ce * probs * (1.0 - probs), 1e-6)
        return grad.reshape(-1), hess.reshape(-1)

    return _obj


def make_relaxed_emd_objective(
    soft_targets_rank: np.ndarray,
    lambda_emd: float = 0.7,
    lambda_ce: float = 0.3,
):
    boundary_weights = clinical_boundary_weights_from_soft_targets(soft_targets_rank)

    def _obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        logits = reshape_logits(predt, num_class=soft_targets_rank.shape[1])
        probs = softmax(logits)
        ce_grad = probs - soft_targets_rank

        cdf_pred = np.cumsum(probs[:, :-1], axis=1)
        cdf_true = np.cumsum(soft_targets_rank[:, :-1], axis=1)
        cdf_diff = cdf_pred - cdf_true
        weighted_cdf = boundary_weights * cdf_diff

        grad_prob = np.zeros_like(probs)
        grad_prob[:, 0] = weighted_cdf[:, 0] + weighted_cdf[:, 1]
        grad_prob[:, 1] = weighted_cdf[:, 1]
        grad_prob[:, 2] = 0.0
        grad_emd = probs * (grad_prob - np.sum(probs * grad_prob, axis=1, keepdims=True))

        grad = lambda_emd * grad_emd + lambda_ce * ce_grad
        hess_scale = lambda_ce + lambda_emd * np.mean(boundary_weights, axis=1, keepdims=True)
        hess = np.maximum(hess_scale * probs * (1.0 - probs), 1e-6)
        return grad.reshape(-1), hess.reshape(-1)

    return _obj


def fit_custom_softprob_booster(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: pd.Series,
    discrete_numeric_features: list[str] | None = None,
    objective_mode: str = "emd",
    use_fw_gpl: bool = False,
    seed: int = RANDOM_STATE,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    X_fit = X_train_df.copy()
    y_fit = y_train.copy()
    if discrete_numeric_features:
        X_fit, y_fit = apply_controlled_adasyn(
            X_fit,
            y_fit,
            discrete_numeric_features,
            seed,
        )
    rank_labels = build_rank_targets(y_fit)
    if use_fw_gpl:
        soft_targets_rank = build_soft_labels_fw_gpl(y_fit)
    else:
        soft_targets_rank = np.eye(3, dtype=float)[rank_labels]

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": seed,
        "max_depth": XGB_BEST_PARAMS["max_depth"],
        "learning_rate": XGB_BEST_PARAMS["learning_rate"],
        "subsample": XGB_BEST_PARAMS["subsample"],
        "colsample_bytree": XGB_BEST_PARAMS["colsample_bytree"],
        "min_child_weight": XGB_BEST_PARAMS["min_child_weight"],
        "reg_alpha": XGB_BEST_PARAMS["reg_alpha"],
        "reg_lambda": XGB_BEST_PARAMS["reg_lambda"],
    }
    dtrain = xgb.DMatrix(X_fit, label=rank_labels, weight=sample_weight)
    dtest = xgb.DMatrix(X_test_df)
    if objective_mode == "soft_ce":
        obj = make_soft_ce_objective(soft_targets_rank, lambda_ce=1.0)
    elif objective_mode == "emd":
        obj = make_relaxed_emd_objective(soft_targets_rank, lambda_emd=0.7, lambda_ce=0.3)
    else:
        raise ValueError(f"未知 objective_mode: {objective_mode}")

    booster = xgb.train(params, dtrain, num_boost_round=XGB_BEST_PARAMS["n_estimators"], obj=obj)
    prediction_details = predict_custom_softprob_bundle({"booster": booster}, X_test_df)
    return {
        "booster": booster,
        "objective_mode": objective_mode,
        "use_fw_gpl": use_fw_gpl,
        "train_size": int(len(y_fit)),
    }, prediction_details["standard_proba"], prediction_details["severity_score"]


def predict_custom_softprob_bundle(
    model_bundle: dict[str, Any],
    X_apply_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    dapply = xgb.DMatrix(X_apply_df)
    raw_margin = model_bundle["booster"].predict(dapply, output_margin=True)
    raw_margin = reshape_logits(np.asarray(raw_margin), num_class=3)
    raw_rank_proba = softmax(raw_margin)
    standard_proba = reorder_rank_proba_to_standard(raw_rank_proba)
    severity_score = compute_continuous_severity_score(raw_rank_proba)
    return {
        "raw_margin": raw_margin,
        "raw_rank_proba": raw_rank_proba,
        "standard_proba": standard_proba,
        "severity_score": severity_score,
    }


def normalize_probability_rows(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=float), 1e-6, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def multiclass_log_loss(y_true: pd.Series, proba: np.ndarray) -> float:
    y_arr = y_true.to_numpy(dtype=int)
    clipped = np.clip(proba, 1e-6, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(y_arr)), y_arr])))


def calibrate_with_temperature(
    calib_raw_margin: np.ndarray,
    y_calib: pd.Series,
    test_raw_margin: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    candidate_temperatures = np.concatenate(
        [
            np.linspace(0.5, 2.0, 16),
            np.linspace(2.25, 4.0, 8),
        ]
    )
    best_temperature = 1.0
    best_loss = float("inf")
    y_rank = build_rank_targets(y_calib)
    for temperature in np.unique(candidate_temperatures):
        rank_proba = softmax(calib_raw_margin / float(temperature))
        loss = float(-np.mean(np.log(np.clip(rank_proba[np.arange(len(y_rank)), y_rank], 1e-6, 1.0))))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    calibrated_rank_proba = softmax(test_raw_margin / best_temperature)
    return {
        "method": "temperature_scaling",
        "temperature": best_temperature,
        "calibration_loss": best_loss,
    }, reorder_rank_proba_to_standard(calibrated_rank_proba)


def fit_one_vs_rest_sigmoid_calibrator(calib_proba: np.ndarray, y_calib: pd.Series) -> dict[str, Any]:
    y_arr = y_calib.to_numpy(dtype=int)
    calibrators: list[LogisticRegression] = []
    for cls in [0, 1, 2]:
        labels = (y_arr == cls).astype(int)
        if labels.min() == labels.max():
            raise ValueError(f"sigmoid 校准缺少类别 {cls} 的正负样本")
        calibrator = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs")
        calibrator.fit(calib_proba[:, [cls]], labels)
        calibrators.append(calibrator)
    return {"method": "ovr_sigmoid", "calibrators": calibrators}


def apply_one_vs_rest_sigmoid_calibrator(calibrator_bundle: dict[str, Any], proba: np.ndarray) -> np.ndarray:
    calibrated_columns = []
    for cls, calibrator in enumerate(calibrator_bundle["calibrators"]):
        calibrated_columns.append(calibrator.predict_proba(proba[:, [cls]])[:, 1])
    return normalize_probability_rows(np.column_stack(calibrated_columns))


def fit_one_vs_rest_isotonic_calibrator(calib_proba: np.ndarray, y_calib: pd.Series) -> dict[str, Any]:
    y_arr = y_calib.to_numpy(dtype=int)
    calibrators: list[IsotonicRegression] = []
    for cls in [0, 1, 2]:
        labels = (y_arr == cls).astype(int)
        if labels.min() == labels.max():
            raise ValueError(f"isotonic 校准缺少类别 {cls} 的正负样本")
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(calib_proba[:, cls], labels)
        calibrators.append(calibrator)
    return {"method": "ovr_isotonic", "calibrators": calibrators}


def apply_one_vs_rest_isotonic_calibrator(calibrator_bundle: dict[str, Any], proba: np.ndarray) -> np.ndarray:
    calibrated_columns = []
    for cls, calibrator in enumerate(calibrator_bundle["calibrators"]):
        calibrated_columns.append(calibrator.predict(proba[:, cls]))
    return normalize_probability_rows(np.column_stack(calibrated_columns))


def collect_metric_row(name: str, y_true: pd.Series, proba: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_predictions(y_true, proba, TARGET_LABELS, [0, 1, 2])
    calibration = evaluate_calibration_metrics(y_true, proba)
    return {
        "variant": name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "ovr_roc_auc_macro": metrics["ovr_roc_auc_macro"],
        "ece": calibration["ece"],
        "brier_score": calibration["brier_score"],
    }


def run_mainline_calibration_experiment(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    X_fit_raw, X_calib_raw, y_fit, y_calib = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE + 200,
        stratify=y_train,
    )
    preprocessor, X_fit_df, X_calib_df = fit_preprocessor_and_transform(X_fit_raw, X_calib_raw)
    X_test_df = transform_with_fitted_preprocessor(preprocessor, X_test_raw)
    discrete_features = get_discrete_features(X_fit_raw)
    model_bundle, _, _ = fit_custom_softprob_booster(
        X_fit_df,
        X_calib_df,
        y_fit,
        discrete_numeric_features=discrete_features,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 120,
    )

    calib_prediction = predict_custom_softprob_bundle(model_bundle, X_calib_df)
    test_prediction = predict_custom_softprob_bundle(model_bundle, X_test_df)

    comparison_rows = [collect_metric_row("uncalibrated", y_test, test_prediction["standard_proba"])]
    calibration_artifacts: dict[str, Any] = {
        "split": {
            "fit_size": int(len(y_fit)),
            "calibration_size": int(len(y_calib)),
            "test_size": int(len(y_test)),
        }
    }

    temp_artifact, temp_proba = calibrate_with_temperature(
        calib_prediction["raw_margin"],
        y_calib,
        test_prediction["raw_margin"],
    )
    comparison_rows.append(collect_metric_row("temperature_scaling", y_test, temp_proba))
    calibration_artifacts["temperature_scaling"] = temp_artifact

    try:
        sigmoid_bundle = fit_one_vs_rest_sigmoid_calibrator(calib_prediction["standard_proba"], y_calib)
        sigmoid_proba = apply_one_vs_rest_sigmoid_calibrator(sigmoid_bundle, test_prediction["standard_proba"])
        comparison_rows.append(collect_metric_row("ovr_sigmoid", y_test, sigmoid_proba))
        calibration_artifacts["ovr_sigmoid"] = {"status": "success"}
    except Exception as exc:
        calibration_artifacts["ovr_sigmoid"] = {"status": "failed", "error": str(exc)}

    try:
        isotonic_bundle = fit_one_vs_rest_isotonic_calibrator(calib_prediction["standard_proba"], y_calib)
        isotonic_proba = apply_one_vs_rest_isotonic_calibrator(isotonic_bundle, test_prediction["standard_proba"])
        comparison_rows.append(collect_metric_row("ovr_isotonic", y_test, isotonic_proba))
        calibration_artifacts["ovr_isotonic"] = {"status": "success"}
    except Exception as exc:
        calibration_artifacts["ovr_isotonic"] = {"status": "failed", "error": str(exc)}

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["ece", "brier_score", "macro_f1"], ascending=[True, True, False]
    )
    comparison_df.to_csv(
        tables_dir / "mainline_calibration_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_variant_comparison(comparison_df, figures_dir / "mainline_calibration_metrics.png")
    plot_calibration_comparison(
        comparison_df[["variant", "ece", "brier_score"]],
        figures_dir / "mainline_calibration_comparison.png",
    )

    best_row = comparison_df.iloc[0]
    best_method = str(best_row["variant"])
    calibrated_map = {
        "uncalibrated": test_prediction["standard_proba"],
        "temperature_scaling": temp_proba,
    }
    if "ovr_sigmoid" in comparison_df["variant"].values:
        calibrated_map["ovr_sigmoid"] = sigmoid_proba
    if "ovr_isotonic" in comparison_df["variant"].values:
        calibrated_map["ovr_isotonic"] = isotonic_proba

    best_proba = calibrated_map[best_method]
    plot_confusion_heatmap(
        y_test,
        np.argmax(best_proba, axis=1),
        figures_dir / "mainline_best_calibrated_confusion_heatmap.png",
    )
    plot_multiclass_roc(y_test, best_proba, figures_dir / "mainline_best_calibrated_multiclass_roc.png")

    calibration_predictions = pd.DataFrame(index=y_test.index)
    calibration_predictions["true_label"] = y_test
    calibration_predictions["true_label_name"] = y_test.map(TARGET_LABELS)
    for method_name, proba in calibrated_map.items():
        calibration_predictions[f"{method_name}_pred"] = np.argmax(proba, axis=1)
        calibration_predictions[f"{method_name}_prob_0"] = proba[:, 0]
        calibration_predictions[f"{method_name}_prob_1"] = proba[:, 1]
        calibration_predictions[f"{method_name}_prob_2"] = proba[:, 2]
    calibration_predictions.to_excel(output_dir / "mainline_calibration_predictions.xlsx", index=True)

    summary = {
        "mainline_variant": MAINLINE_VARIANT_NAME,
        "recommended_method": best_method,
        "comparison_rank": comparison_df.to_dict(orient="records"),
        "artifacts": calibration_artifacts,
    }
    (artifacts_dir / "mainline_calibration_artifacts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_strict_prospective_experiment(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    X_train_strict = build_strict_prospective_view(X_train_raw)
    X_test_strict = build_strict_prospective_view(X_test_raw)

    _, X_train_base_df, X_test_base_df = fit_preprocessor_and_transform(X_train_strict, X_test_strict)
    baseline_discrete = get_discrete_features(X_train_strict)
    baseline_model, baseline_proba = fit_baseline_xgb(
        X_train_base_df,
        y_train,
        X_test_base_df,
        discrete_numeric_features=baseline_discrete,
    )

    X_train_main_df = X_train_base_df.copy()
    X_test_main_df = X_test_base_df.copy()
    mainline_model, _, _ = fit_custom_softprob_booster(
        X_train_main_df,
        X_test_main_df,
        y_train,
        discrete_numeric_features=baseline_discrete,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 220,
    )
    mainline_prediction = predict_custom_softprob_bundle(mainline_model, X_test_main_df)
    mainline_proba = mainline_prediction["standard_proba"]

    metric_rows = [
        collect_metric_row("baseline_xgboost_strict_prospective", y_test, baseline_proba),
        collect_metric_row(f"{MAINLINE_VARIANT_NAME}_strict_prospective", y_test, mainline_proba),
    ]
    metric_df = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"], ascending=False
    )
    metric_df.to_csv(
        tables_dir / "strict_prospective_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_variant_comparison(metric_df, figures_dir / "strict_prospective_metrics.png")
    plot_calibration_comparison(
        metric_df[["variant", "ece", "brier_score"]],
        figures_dir / "strict_prospective_calibration.png",
    )
    plot_roc_overview(
        y_test,
        {
            "baseline_strict": baseline_proba,
            "mainline_strict": mainline_proba,
        },
        figures_dir / "strict_prospective_roc_overview.png",
    )
    plot_confusion_heatmap(
        y_test,
        np.argmax(mainline_proba, axis=1),
        figures_dir / "strict_prospective_mainline_confusion_heatmap.png",
    )
    plot_multiclass_roc(
        y_test,
        mainline_proba,
        figures_dir / "strict_prospective_mainline_multiclass_roc.png",
    )

    predictions = X_test_strict.copy()
    predictions["true_label"] = y_test
    predictions["true_label_name"] = y_test.map(TARGET_LABELS)
    predictions["baseline_xgboost_strict_pred"] = np.argmax(baseline_proba, axis=1)
    predictions["mainline_strict_pred"] = np.argmax(mainline_proba, axis=1)
    for cls in [0, 1, 2]:
        predictions[f"baseline_xgboost_strict_prob_{cls}"] = baseline_proba[:, cls]
        predictions[f"mainline_strict_prob_{cls}"] = mainline_proba[:, cls]
    predictions.to_excel(output_dir / "strict_prospective_predictions.xlsx", index=True)

    summary = {
        "removed_columns": POST_TREATMENT_COLUMNS,
        "n_features_after_drop": int(X_train_strict.shape[1]),
        "variant_rank": metric_df.to_dict(orient="records"),
        "best_variant": str(metric_df.iloc[0]["variant"]),
    }
    (artifacts_dir / "strict_prospective_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_dml_covariates(X_raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col
        for col in X_raw.columns
        if col not in TREATMENT_COLUMNS and col not in POST_TREATMENT_COLUMNS and col != "ARR比值"
    ]
    return X_raw[cols].copy()


def impute_simple_raw(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    for col in train.columns:
        if train[col].dtype == "object":
            if col == "确诊实验类型":
                train[col] = train[col].map(normalize_experiment_type)
                test[col] = test[col].map(normalize_experiment_type)
            fill_value = train[col].mode(dropna=True)
            fill_value = fill_value.iloc[0] if not fill_value.empty else "Missing"
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)
        else:
            median = pd.to_numeric(train[col], errors="coerce").median()
            train[col] = pd.to_numeric(train[col], errors="coerce").fillna(median)
            test[col] = pd.to_numeric(test[col], errors="coerce").fillna(median)
    return train, test


def crossfit_residuals(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    target_train: pd.Series,
    target_test: pd.Series,
    reg_factory,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(target_train), dtype=float)
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw), start=1):
        X_tr = X_train_raw.iloc[tr_idx]
        X_va = X_train_raw.iloc[va_idx]
        y_tr = target_train.iloc[tr_idx]
        X_tr_imp, X_va_imp = impute_simple_raw(X_tr, X_va)
        X_tr_df, X_va_df = preprocess_pair(X_tr_imp, X_va_imp)
        model = reg_factory(RANDOM_STATE + fold)
        model.fit(X_tr_df, y_tr)
        oof[va_idx] = model.predict(X_va_df)

    X_train_imp, X_test_imp = impute_simple_raw(X_train_raw, X_test_raw)
    X_train_df, X_test_df = preprocess_pair(X_train_imp, X_test_imp)
    final_model = reg_factory(RANDOM_STATE)
    final_model.fit(X_train_df, target_train)
    test_pred = final_model.predict(X_test_df)
    return oof, test_pred


def build_dml_features(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    W_train_raw = build_dml_covariates(X_train_raw)
    W_test_raw = build_dml_covariates(X_test_raw)

    treatment_residuals_train: dict[str, np.ndarray] = {}
    treatment_residuals_test: dict[str, np.ndarray] = {}
    for col in TREATMENT_COLUMNS:
        pred_train, pred_test = crossfit_residuals(
            W_train_raw,
            W_test_raw,
            pd.to_numeric(X_train_raw[col], errors="coerce").fillna(0.0),
            pd.to_numeric(X_test_raw[col], errors="coerce").fillna(0.0),
            lambda seed: make_xgb_regressor(seed, n_estimators=150),
        )
        treatment_residuals_train[col] = pd.to_numeric(X_train_raw[col], errors="coerce").fillna(0.0).to_numpy() - pred_train
        treatment_residuals_test[col] = pd.to_numeric(X_test_raw[col], errors="coerce").fillna(0.0).to_numpy() - pred_test

    arr_train = pd.to_numeric(X_train_raw["ARR比值"], errors="coerce")
    arr_test = pd.to_numeric(X_test_raw["ARR比值"], errors="coerce")
    arr_fill = float(arr_train.median())
    arr_train = arr_train.fillna(arr_fill)
    arr_test = arr_test.fillna(arr_fill)
    arr_base_train, arr_base_test = crossfit_residuals(
        W_train_raw,
        W_test_raw,
        arr_train,
        arr_test,
        lambda seed: make_xgb_regressor(seed, n_estimators=170),
    )
    arr_res_train = arr_train.to_numpy() - arr_base_train

    tret_train_df = pd.DataFrame(treatment_residuals_train, index=X_train_raw.index)
    tret_test_df = pd.DataFrame(treatment_residuals_test, index=X_test_raw.index)
    effect_train, effect_test = crossfit_residuals(
        tret_train_df,
        tret_test_df,
        pd.Series(arr_res_train, index=X_train_raw.index),
        pd.Series(np.zeros(len(X_test_raw)), index=X_test_raw.index),
        lambda seed: make_xgb_regressor(seed, n_estimators=140),
    )

    corrected_arr_train = arr_train.to_numpy() - effect_train
    corrected_arr_test = arr_test.to_numpy() - effect_test

    X_train_variant = X_train_raw.copy()
    X_test_variant = X_test_raw.copy()
    X_train_variant = X_train_variant.drop(columns=["ARR比值"])
    X_test_variant = X_test_variant.drop(columns=["ARR比值"])
    X_train_variant["DML_校正ARR"] = corrected_arr_train
    X_test_variant["DML_校正ARR"] = corrected_arr_test
    for col in TREATMENT_COLUMNS:
        X_train_variant[f"DML_残差_{col}"] = treatment_residuals_train[col]
        X_test_variant[f"DML_残差_{col}"] = treatment_residuals_test[col]

    artifacts = {
        "mean_abs_treatment_residual": float(np.mean(np.abs(tret_train_df.to_numpy()))),
        "mean_corrected_arr_shift": float(np.mean(corrected_arr_test - arr_test.to_numpy())),
        "corrected_arr_train_mean": float(np.mean(corrected_arr_train)),
        "corrected_arr_test_mean": float(np.mean(corrected_arr_test)),
        "corrected_arr_test_std": float(np.std(corrected_arr_test)),
    }
    return X_train_variant, X_test_variant, artifacts


def fit_ordinal_xgboost(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: pd.Series,
    discrete_numeric_features: list[str] | None = None,
) -> tuple[dict[str, XGBClassifier], np.ndarray]:
    if discrete_numeric_features:
        X_train_df, y_train = apply_controlled_adasyn(
            X_train_df,
            y_train,
            discrete_numeric_features,
            RANDOM_STATE,
        )
    order_rank = y_train.map({0: 0, 2: 1, 1: 2})
    y_gt0 = (order_rank > 0).astype(int)
    y_gt1 = (order_rank > 1).astype(int)

    model_gt0 = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )
    model_gt1 = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE + 1,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )
    sw0 = compute_sample_weight(class_weight="balanced", y=y_gt0)
    sw1 = compute_sample_weight(class_weight="balanced", y=y_gt1)
    model_gt0.fit(X_train_df, y_gt0, sample_weight=sw0, verbose=False)
    model_gt1.fit(X_train_df, y_gt1, sample_weight=sw1, verbose=False)

    p_gt0 = model_gt0.predict_proba(X_test_df)[:, 1]
    p_gt1 = model_gt1.predict_proba(X_test_df)[:, 1]
    p_gt1 = np.minimum(p_gt1, p_gt0)
    p0 = 1.0 - p_gt0
    p2 = p_gt0 - p_gt1  # original label 2 is gray
    p1 = p_gt1  # original label 1 is confirmed
    proba = np.column_stack([p0, p1, p2])
    proba = np.clip(proba, 1e-6, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {"gt0": model_gt0, "gt1": model_gt1}, proba


def compute_cost_sensitive_weights(
    original_y: pd.Series,
    threshold_name: str,
) -> np.ndarray:
    base_weight = compute_sample_weight(class_weight="balanced", y=original_y)
    if threshold_name == "gt0":
        # Encourage separating class 0 from the two higher clinical states.
        multiplier = original_y.map({0: 1.6, 2: 1.15, 1: 1.45}).to_numpy(dtype=float)
    elif threshold_name == "gt1":
        # Make class 1 vs (0,2) harder and treat gray zone as adjacent, not identical to 0.
        multiplier = original_y.map({0: 0.85, 2: 1.35, 1: 1.85}).to_numpy(dtype=float)
    else:
        raise ValueError(f"未知阈值名称: {threshold_name}")
    return base_weight * multiplier


def apply_ordinal_cost_matrix(
    proba: np.ndarray,
    cost_matrix: np.ndarray | None = None,
    blend: float = 0.35,
    temperature: float = 0.8,
) -> np.ndarray:
    if cost_matrix is None:
        cost_matrix = np.asarray(
            [
                [0.0, 2.3, 1.0],  # true 0
                [2.6, 0.0, 1.0],  # true 1
                [1.0, 1.0, 0.0],  # true 2
            ],
            dtype=float,
        )
    expected_cost = proba @ cost_matrix
    stabilized = -expected_cost / temperature
    stabilized -= stabilized.max(axis=1, keepdims=True)
    risk_softmax = np.exp(stabilized)
    risk_softmax /= risk_softmax.sum(axis=1, keepdims=True)
    adjusted = (1.0 - blend) * proba + blend * risk_softmax
    adjusted = np.clip(adjusted, 1e-6, 1.0)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def fit_ordinal_xgboost_v2(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: pd.Series,
    discrete_numeric_features: list[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    X_fit = X_train_df.copy()
    y_fit = y_train.copy()
    if discrete_numeric_features:
        X_fit, y_fit = apply_controlled_adasyn(
            X_fit,
            y_fit,
            discrete_numeric_features,
            RANDOM_STATE,
        )
    order_rank = y_fit.map({0: 0, 2: 1, 1: 2})
    y_gt0 = (order_rank > 0).astype(int)
    y_gt1 = (order_rank > 1).astype(int)

    model_gt0 = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE + 10,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )
    model_gt1 = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE + 11,
        n_jobs=-1,
        **XGB_BEST_PARAMS,
    )
    sw0 = compute_cost_sensitive_weights(y_fit, "gt0")
    sw1 = compute_cost_sensitive_weights(y_fit, "gt1")
    model_gt0.fit(X_fit, y_gt0, sample_weight=sw0, verbose=False)
    model_gt1.fit(X_fit, y_gt1, sample_weight=sw1, verbose=False)

    p_gt0 = model_gt0.predict_proba(X_test_df)[:, 1]
    p_gt1 = model_gt1.predict_proba(X_test_df)[:, 1]
    p_gt1 = np.minimum(p_gt1, p_gt0)
    p0 = 1.0 - p_gt0
    p2 = p_gt0 - p_gt1
    p1 = p_gt1
    raw_proba = np.column_stack([p0, p1, p2])
    raw_proba = np.clip(raw_proba, 1e-6, 1.0)
    raw_proba = raw_proba / raw_proba.sum(axis=1, keepdims=True)

    cost_matrix = np.asarray(
        [
            [0.0, 2.3, 1.0],
            [2.6, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    final_proba = apply_ordinal_cost_matrix(raw_proba, cost_matrix=cost_matrix, blend=0.35, temperature=0.8)
    return {
        "gt0": model_gt0,
        "gt1": model_gt1,
        "cost_matrix": cost_matrix.tolist(),
        "blend": 0.35,
        "temperature": 0.8,
    }, final_proba


def add_physiology_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    X = X_raw.copy()
    arr = pd.to_numeric(X.get("ARR比值"), errors="coerce")
    renin = pd.to_numeric(X.get("肾素"), errors="coerce")
    ald = pd.to_numeric(X.get("醛固酮"), errors="coerce")
    k = pd.to_numeric(X.get("钾"), errors="coerce")
    beta = pd.to_numeric(X.get("Beta_等效分数"), errors="coerce").fillna(0.0)
    diur = pd.to_numeric(X.get("利尿剂_等效分数"), errors="coerce").fillna(0.0)
    rass = pd.to_numeric(X.get("RASS_等效分数"), errors="coerce").fillna(0.0)
    alpha = pd.to_numeric(X.get("Alpha_等效分数"), errors="coerce").fillna(0.0)
    dhp = pd.to_numeric(X.get("二氢吡啶类_等效分数"), errors="coerce").fillna(0.0)
    ndhp = pd.to_numeric(X.get("非二氢吡啶类_等效分数"), errors="coerce").fillna(0.0)

    suppress = beta + rass + alpha
    stimulate = diur + dhp + ndhp
    total_burden = suppress + stimulate

    X["药理_总负荷"] = total_burden
    X["药理_净抑制分数"] = suppress - stimulate
    X["药理_ARR_Beta交互"] = arr * beta
    X["药理_ARR_RASS交互"] = arr * rass
    X["药理_ARR_利尿剂校正"] = arr / (1.0 + diur)
    X["药理_ARR_CCB校正"] = arr / (1.0 + dhp + ndhp)
    X["药理_肾素_Beta校正"] = renin / (1.0 + beta)
    X["药理_肾素_利尿剂刺激"] = renin * (1.0 + diur)
    X["药理_醛固酮_RASS交互"] = ald * (1.0 + rass)
    X["药理_低钾ARR"] = arr * np.clip(4.0 - k, a_min=0.0, a_max=None)
    X["药理_综合校正ARR"] = arr * (1.0 + 0.15 * beta + 0.08 * rass + 0.05 * alpha) / (
        1.0 + 0.12 * diur + 0.10 * dhp + 0.06 * ndhp
    )
    return X


def build_best_variant_feature_importance(
    best_name: str,
    best_model: Any,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    if best_name in {"ordinal_xgboost", "ordinal_xgboost_v2", "dml_ordinal_xgboost"}:
        imp0 = pd.Series(best_model["gt0"].feature_importances_, index=feature_names)
        imp1 = pd.Series(best_model["gt1"].feature_importances_, index=feature_names)
        importance = ((imp0 + imp1) / 2).sort_values(ascending=False).reset_index()
        importance.columns = ["feature", "importance"]
    elif isinstance(best_model, xgb.Booster):
        gain_map = best_model.get_score(importance_type="gain")
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": [gain_map.get(f, 0.0) for f in feature_names],
            }
        ).sort_values("importance", ascending=False)
    else:
        importance = pd.DataFrame(
            {"feature": feature_names, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)

    importance.to_csv(
        output_dir / "tables" / "best_variant_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    show_df = importance.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(show_df["feature"], show_df["importance"], color="#4C72B0")
    plt.title("最优变体 Top20 特征重要性")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "best_variant_feature_importance.png", dpi=300)
    plt.close()


def plot_variant_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    melted = metrics_df[
        ["variant", "accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"]
    ].melt(id_vars="variant", var_name="metric", value_name="score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="variant")
    plt.ylim(0, 1.05)
    plt.title("XGBoost 因果/机制变体指标对比")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_recall_heatmap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for _, row in metrics_df.iterrows():
        report = row["classification_report"]
        rows.extend(
            [
                {"variant": row["variant"], "class": "非确诊", "recall": report["非确诊"]["recall"]},
                {"variant": row["variant"], "class": "确诊", "recall": report["确诊"]["recall"]},
                {"variant": row["variant"], "class": "灰色区域", "recall": report["灰色区域"]["recall"]},
            ]
        )
    pivot = pd.DataFrame(rows).pivot(index="variant", columns="class", values="recall")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("各变体分类别召回率热力图")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_overview(y_true: pd.Series, probas: dict[str, np.ndarray], output_path: Path) -> None:
    y_bin = label_binarize_like(y_true)
    plt.figure(figsize=(8, 6))
    for name, proba in probas.items():
        auc_score = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin[:, 1], proba[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("各变体对确诊类的 ROC 对比")
    plt.xlabel("假阳性率")
    plt.ylabel("真正率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def label_binarize_like(y_true: pd.Series) -> np.ndarray:
    arr = np.zeros((len(y_true), 3), dtype=int)
    for idx, cls in enumerate([0, 1, 2]):
        arr[:, idx] = (y_true.to_numpy() == cls).astype(int)
    return arr


def plot_calibration_comparison(calibration_df: pd.DataFrame, output_path: Path) -> None:
    melted = calibration_df.melt(
        id_vars="variant",
        value_vars=["ece", "brier_score"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="variant")
    plt.title("各变体校准指标对比")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_severity_distributions(severity_df: pd.DataFrame, violin_path: Path, boxplot_path: Path) -> None:
    plot_df = severity_df.copy()
    plot_df["true_label_name"] = plot_df["true_label"].map(TARGET_LABELS)
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=plot_df, x="variant", y="severity_score", hue="true_label_name", cut=0)
    plt.ylim(-0.02, 1.02)
    plt.title("各变体连续严重程度评分分布")
    plt.tight_layout()
    plt.savefig(violin_path, dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df, x="variant", y="severity_score", hue="true_label_name")
    plt.ylim(-0.02, 1.02)
    plt.title("各变体连续严重程度评分箱线图")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=300)
    plt.close()


def normalize_external_experiment_type(value: Any) -> Any:
    value = clean_string_cell(value)
    if pd.isna(value):
        return np.nan
    mapping = {
        "盐水试验": "冷盐水实验",
        "盐水实验": "冷盐水实验",
        "盐水负荷": "冷盐水实验",
        "冷盐水": "冷盐水实验",
        "卡托普利": "卡托普利实验",
        "卡托普利试验": "卡托普利实验",
    }
    return normalize_experiment_type(mapping.get(value, value))


def clean_external_cell(value: Any) -> Any:
    value = clean_string_cell(value)
    if pd.isna(value):
        return np.nan
    if str(value).strip() in {"/", "\\", "／"}:
        return np.nan
    return value


def normalize_external_binary_label(value: Any) -> float:
    value = clean_external_cell(value)
    if pd.isna(value):
        return float("nan")
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return float("nan")
    if numeric in {0, 1}:
        return float(numeric)
    return float("nan")


def prepare_external_dataset(
    file_path: Path,
    reference_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    raw_df = pd.read_excel(file_path).copy()
    mapped_source = raw_df.rename(columns=EXTERNAL_TEST_COLUMN_MAP).copy()
    mapped_source = mapped_source.applymap(clean_external_cell)

    for col in mapped_source.columns:
        if col == "是否确诊":
            continue
        if mapped_source[col].dtype == "object":
            if col == EXPERIMENT_TYPE_COLUMN:
                mapped_source[col] = mapped_source[col].map(normalize_external_experiment_type)
            else:
                mapped_source[col] = maybe_convert_object_to_numeric(mapped_source[col])

    aligned = pd.DataFrame(index=mapped_source.index)
    for col in reference_columns:
        aligned[col] = mapped_source[col] if col in mapped_source.columns else np.nan

    for col in ZERO_FILL_COLUMNS:
        if col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)

    med_cols = [col for col in ZERO_FILL_COLUMNS if col != "联合用药_总数" and col in aligned.columns]
    if "联合用药_总数" in aligned.columns and med_cols:
        med_df = pd.DataFrame(
            {
                col: pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)
                for col in med_cols
            },
            index=aligned.index,
        )
        aligned["联合用药_总数"] = (med_df > 0).sum(axis=1).astype(float)

    if EXPERIMENT_TYPE_COLUMN in aligned.columns:
        aligned[EXPERIMENT_TYPE_COLUMN] = aligned[EXPERIMENT_TYPE_COLUMN].map(normalize_external_experiment_type)

    if "是否确诊" in raw_df.columns:
        y_external = raw_df["是否确诊"].apply(normalize_external_binary_label)
    else:
        y_external = pd.Series(np.nan, index=raw_df.index, name="是否确诊")

    mapping_report = {
        "source_shape": list(raw_df.shape),
        "source_columns": raw_df.columns.tolist(),
        "rename_map_used": {
            src: dst for src, dst in EXTERNAL_TEST_COLUMN_MAP.items() if src in raw_df.columns
        },
        "missing_reference_columns": [col for col in reference_columns if col not in mapped_source.columns],
        "available_reference_columns": [col for col in reference_columns if col in mapped_source.columns],
        "labeled_samples": int(y_external.notna().sum()),
        "unlabeled_samples": int(y_external.isna().sum()),
        "label_distribution_binary": {
            str(int(k)): int(v)
            for k, v in y_external.dropna().astype(int).value_counts().sort_index().items()
        },
        "column_missing_counts_after_alignment": {
            col: int(aligned[col].isna().sum()) for col in aligned.columns
        },
    }
    return raw_df, aligned, y_external, mapping_report


def collapse_multiclass_proba_to_confirmed_binary(proba: np.ndarray) -> np.ndarray:
    binary = np.column_stack([proba[:, 0] + proba[:, 2], proba[:, 1]])
    binary = np.clip(binary, 1e-6, 1.0)
    return binary / binary.sum(axis=1, keepdims=True)


def evaluate_external_binary_from_multiclass(
    y_true_binary: pd.Series,
    proba: np.ndarray,
) -> dict[str, Any]:
    binary_proba = collapse_multiclass_proba_to_confirmed_binary(proba)
    metrics = evaluate_predictions(
        y_true_binary.astype(int),
        binary_proba,
        {0: "非确诊或灰区", 1: "确诊"},
        [0, 1],
    )
    pred_three = np.argmax(proba, axis=1)
    metrics["predicted_class_0_count"] = int((pred_three == 0).sum())
    metrics["predicted_class_1_count"] = int((pred_three == 1).sum())
    metrics["predicted_class_2_count"] = int((pred_three == 2).sum())
    metrics["gray_prediction_rate"] = float(np.mean(pred_three == 2))
    metrics["mean_confirmed_probability"] = float(np.mean(proba[:, 1]))
    return metrics


def save_model_bundles(
    model_dir: Path,
    bundles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, bundle in bundles.items():
        output_path = model_dir / f"{variant_name}.joblib"
        record = {
            "variant": variant_name,
            "file_path": str(output_path),
            "status": "saved",
        }
        try:
            joblib.dump(bundle, output_path)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        manifest.append(record)
    return manifest


def run_external_full_training_evaluation(
    prepared,
    external_input_path: Path,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not external_input_path.exists():
        return None

    external_dir = output_dir / "external_test_evaluation"
    tables_dir = external_dir / "tables"
    artifacts_dir = external_dir / "artifacts"
    model_dir = output_dir / "saved_models"
    external_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_full_raw = prepared.X.copy()
    y_full = prepared.y.copy()
    external_raw_df, X_external_raw, y_external, mapping_report = prepare_external_dataset(
        external_input_path,
        list(X_full_raw.columns),
    )

    external_probas: dict[str, np.ndarray] = {}
    external_severity: dict[str, np.ndarray] = {}
    external_artifacts: dict[str, Any] = {}
    model_bundles: dict[str, dict[str, Any]] = {}

    # Baseline
    base_preprocessor, X_full_base_df, X_ext_base_df = fit_preprocessor_and_transform(X_full_raw, X_external_raw)
    base_discrete = get_discrete_features(X_full_raw)
    baseline_model, baseline_proba = fit_baseline_xgb(
        X_full_base_df,
        y_full,
        X_ext_base_df,
        discrete_numeric_features=base_discrete,
    )
    external_probas["baseline_xgboost"] = baseline_proba
    external_severity["baseline_xgboost"] = np.clip((baseline_proba[:, 2] + 2.0 * baseline_proba[:, 1]) / 2.0, 0.0, 1.0)
    model_bundles["baseline_xgboost"] = {
        "variant": "baseline_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_raw.columns),
        "preprocessor": base_preprocessor,
        "model": baseline_model,
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
    }

    # DML baseline
    X_full_dml_raw, X_ext_dml_raw, dml_artifacts = build_dml_features(X_full_raw, X_external_raw)
    dml_preprocessor, X_full_dml_df, X_ext_dml_df = fit_preprocessor_and_transform(X_full_dml_raw, X_ext_dml_raw)
    dml_discrete = get_discrete_features(X_full_dml_raw)
    dml_model, dml_proba = fit_baseline_xgb(
        X_full_dml_df,
        y_full,
        X_ext_dml_df,
        discrete_numeric_features=dml_discrete,
    )
    external_probas["dml_xgboost"] = dml_proba
    external_severity["dml_xgboost"] = np.clip((dml_proba[:, 2] + 2.0 * dml_proba[:, 1]) / 2.0, 0.0, 1.0)
    external_artifacts["dml_xgboost"] = dml_artifacts
    model_bundles["dml_xgboost"] = {
        "variant": "dml_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_dml_raw.columns),
        "preprocessor": dml_preprocessor,
        "model": dml_model,
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
        "dml_feature_artifacts": dml_artifacts,
    }

    # Adaptive-CaRe
    feature_weights = build_adaptive_feature_weights(list(X_full_base_df.columns))
    care_model, care_proba = fit_weighted_xgb_booster(
        X_full_base_df,
        y_full,
        X_ext_base_df,
        feature_weights=feature_weights,
        adaptive=True,
        discrete_numeric_features=base_discrete,
    )
    external_probas["adaptive_care_xgboost"] = care_proba
    external_severity["adaptive_care_xgboost"] = np.clip((care_proba[:, 2] + 2.0 * care_proba[:, 1]) / 2.0, 0.0, 1.0)
    external_artifacts["adaptive_care_xgboost"] = {
        "feature_weight_summary": {
            "min": float(feature_weights.min()),
            "max": float(feature_weights.max()),
            "mean": float(feature_weights.mean()),
        }
    }
    model_bundles["adaptive_care_xgboost"] = {
        "variant": "adaptive_care_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_raw.columns),
        "preprocessor": base_preprocessor,
        "model": care_model,
        "feature_weights": feature_weights,
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
    }

    # Ordinal
    ordinal_model, ordinal_proba = fit_ordinal_xgboost(
        X_full_base_df,
        X_ext_base_df,
        y_full,
        discrete_numeric_features=base_discrete,
    )
    external_probas["ordinal_xgboost"] = ordinal_proba
    external_severity["ordinal_xgboost"] = np.clip((ordinal_proba[:, 2] + 2.0 * ordinal_proba[:, 1]) / 2.0, 0.0, 1.0)
    model_bundles["ordinal_xgboost"] = {
        "variant": "ordinal_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_raw.columns),
        "preprocessor": base_preprocessor,
        "model": ordinal_model,
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
    }

    # Ordinal V2
    ordinal_v2_model, ordinal_v2_proba = fit_ordinal_xgboost_v2(
        X_full_base_df,
        X_ext_base_df,
        y_full,
        discrete_numeric_features=base_discrete,
    )
    external_probas["ordinal_xgboost_v2"] = ordinal_v2_proba
    external_severity["ordinal_xgboost_v2"] = np.clip((ordinal_v2_proba[:, 2] + 2.0 * ordinal_v2_proba[:, 1]) / 2.0, 0.0, 1.0)
    external_artifacts["ordinal_xgboost_v2"] = {
        "cost_matrix": ordinal_v2_model["cost_matrix"],
        "blend": ordinal_v2_model["blend"],
        "temperature": ordinal_v2_model["temperature"],
    }
    model_bundles["ordinal_xgboost_v2"] = {
        "variant": "ordinal_xgboost_v2",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_raw.columns),
        "preprocessor": base_preprocessor,
        "model": ordinal_v2_model,
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
    }

    # DML + Ordinal
    dml_ordinal_model, dml_ordinal_proba = fit_ordinal_xgboost_v2(
        X_full_dml_df,
        X_ext_dml_df,
        y_full,
        discrete_numeric_features=dml_discrete,
    )
    external_probas["dml_ordinal_xgboost"] = dml_ordinal_proba
    external_severity["dml_ordinal_xgboost"] = np.clip((dml_ordinal_proba[:, 2] + 2.0 * dml_ordinal_proba[:, 1]) / 2.0, 0.0, 1.0)
    external_artifacts["dml_ordinal_xgboost"] = {
        **dml_artifacts,
        "cost_matrix": dml_ordinal_model["cost_matrix"],
        "blend": dml_ordinal_model["blend"],
        "temperature": dml_ordinal_model["temperature"],
    }
    model_bundles["dml_ordinal_xgboost"] = {
        "variant": "dml_ordinal_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_dml_raw.columns),
        "preprocessor": dml_preprocessor,
        "model": dml_ordinal_model,
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
        "dml_feature_artifacts": dml_artifacts,
    }

    # Physiology-informed
    X_full_phy_raw = add_physiology_features(X_full_raw)
    X_ext_phy_raw = add_physiology_features(X_external_raw)
    phy_preprocessor, X_full_phy_df, X_ext_phy_df = fit_preprocessor_and_transform(X_full_phy_raw, X_ext_phy_raw)
    phy_discrete = get_discrete_features(X_full_phy_raw)
    phy_model, phy_proba = fit_baseline_xgb(
        X_full_phy_df,
        y_full,
        X_ext_phy_df,
        discrete_numeric_features=phy_discrete,
    )
    external_probas["physiology_informed_xgboost"] = phy_proba
    external_severity["physiology_informed_xgboost"] = np.clip((phy_proba[:, 2] + 2.0 * phy_proba[:, 1]) / 2.0, 0.0, 1.0)
    external_artifacts["physiology_informed_xgboost"] = {
        "added_features": [c for c in X_full_phy_raw.columns if c not in X_full_raw.columns]
    }
    model_bundles["physiology_informed_xgboost"] = {
        "variant": "physiology_informed_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_phy_raw.columns),
        "preprocessor": phy_preprocessor,
        "model": phy_model,
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
    }

    # Ordinal EMD
    ordinal_emd_model, ordinal_emd_proba, ordinal_emd_severity = fit_custom_softprob_booster(
        X_full_base_df,
        X_ext_base_df,
        y_full,
        discrete_numeric_features=base_discrete,
        objective_mode="emd",
        use_fw_gpl=False,
        seed=RANDOM_STATE + 120,
    )
    external_probas["ordinal_emd_xgboost"] = ordinal_emd_proba
    external_severity["ordinal_emd_xgboost"] = ordinal_emd_severity
    external_artifacts["ordinal_emd_xgboost"] = ordinal_emd_model
    model_bundles["ordinal_emd_xgboost"] = {
        "variant": "ordinal_emd_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_raw.columns),
        "preprocessor": base_preprocessor,
        "model": ordinal_emd_model["booster"],
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
        "objective_mode": ordinal_emd_model["objective_mode"],
        "use_fw_gpl": ordinal_emd_model["use_fw_gpl"],
    }

    # Ordinal FW-GPL
    ordinal_fw_model, ordinal_fw_proba, ordinal_fw_severity = fit_custom_softprob_booster(
        X_full_base_df,
        X_ext_base_df,
        y_full,
        discrete_numeric_features=base_discrete,
        objective_mode="soft_ce",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 121,
    )
    external_probas["ordinal_fw_gpl_xgboost"] = ordinal_fw_proba
    external_severity["ordinal_fw_gpl_xgboost"] = ordinal_fw_severity
    external_artifacts["ordinal_fw_gpl_xgboost"] = ordinal_fw_model
    model_bundles["ordinal_fw_gpl_xgboost"] = {
        "variant": "ordinal_fw_gpl_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_raw.columns),
        "preprocessor": base_preprocessor,
        "model": ordinal_fw_model["booster"],
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
        "objective_mode": ordinal_fw_model["objective_mode"],
        "use_fw_gpl": ordinal_fw_model["use_fw_gpl"],
    }

    # Ordinal EMD + FW-GPL
    ordinal_emd_fw_model, ordinal_emd_fw_proba, ordinal_emd_fw_severity = fit_custom_softprob_booster(
        X_full_base_df,
        X_ext_base_df,
        y_full,
        discrete_numeric_features=base_discrete,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 122,
    )
    external_probas["ordinal_emd_fw_gpl_xgboost"] = ordinal_emd_fw_proba
    external_severity["ordinal_emd_fw_gpl_xgboost"] = ordinal_emd_fw_severity
    external_artifacts["ordinal_emd_fw_gpl_xgboost"] = ordinal_emd_fw_model
    model_bundles["ordinal_emd_fw_gpl_xgboost"] = {
        "variant": "ordinal_emd_fw_gpl_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_raw.columns),
        "preprocessor": base_preprocessor,
        "model": ordinal_emd_fw_model["booster"],
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
        "objective_mode": ordinal_emd_fw_model["objective_mode"],
        "use_fw_gpl": ordinal_emd_fw_model["use_fw_gpl"],
    }

    # DML + EMD + FW-GPL
    dml_emd_fw_model, dml_emd_fw_proba, dml_emd_fw_severity = fit_custom_softprob_booster(
        X_full_dml_df,
        X_ext_dml_df,
        y_full,
        discrete_numeric_features=dml_discrete,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 123,
    )
    external_probas["dml_ordinal_emd_fw_gpl_xgboost"] = dml_emd_fw_proba
    external_severity["dml_ordinal_emd_fw_gpl_xgboost"] = dml_emd_fw_severity
    external_artifacts["dml_ordinal_emd_fw_gpl_xgboost"] = {
        **dml_artifacts,
        **dml_emd_fw_model,
    }
    model_bundles["dml_ordinal_emd_fw_gpl_xgboost"] = {
        "variant": "dml_ordinal_emd_fw_gpl_xgboost",
        "training_mode": "full_cleaned_training_dataset",
        "raw_feature_columns": list(X_full_dml_raw.columns),
        "preprocessor": dml_preprocessor,
        "model": dml_emd_fw_model["booster"],
        "training_raw_X": X_full_raw,
        "target_column": TARGET_COLUMN,
        "dml_feature_artifacts": dml_artifacts,
        "objective_mode": dml_emd_fw_model["objective_mode"],
        "use_fw_gpl": dml_emd_fw_model["use_fw_gpl"],
    }

    labeled_mask = y_external.notna().to_numpy()
    external_metric_rows: list[dict[str, Any]] = []
    external_metric_detail: dict[str, Any] = {}
    for name, proba in external_probas.items():
        pred = np.argmax(proba, axis=1)
        row = {
            "variant": name,
            "n_external_total": int(len(proba)),
            "n_labeled": int(labeled_mask.sum()),
            "predicted_class_0_count": int((pred == 0).sum()),
            "predicted_class_1_count": int((pred == 1).sum()),
            "predicted_class_2_count": int((pred == 2).sum()),
            "gray_prediction_rate_all": float(np.mean(pred == 2)),
            "mean_confirmed_probability_all": float(np.mean(proba[:, 1])),
        }
        if labeled_mask.any():
            detail = evaluate_external_binary_from_multiclass(
                y_external.loc[y_external.notna()],
                proba[labeled_mask],
            )
            external_metric_detail[name] = detail
            row.update(
                {
                    "accuracy_binary": detail["accuracy"],
                    "balanced_accuracy_binary": detail["balanced_accuracy"],
                    "macro_f1_binary": detail["macro_f1"],
                    "weighted_f1_binary": detail["weighted_f1"],
                    "ovr_roc_auc_binary": detail["ovr_roc_auc_macro"],
                    "gray_prediction_rate_labeled": detail["gray_prediction_rate"],
                }
            )
        external_metric_rows.append(row)

    external_metrics_df = pd.DataFrame(external_metric_rows).sort_values(
        ["macro_f1_binary", "balanced_accuracy_binary", "accuracy_binary"],
        ascending=False,
        na_position="last",
    )
    external_metrics_df.to_csv(
        tables_dir / "external_variant_binary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    external_predictions = external_raw_df.copy()
    external_predictions["_mapped_binary_label"] = y_external
    for name, proba in external_probas.items():
        pred = np.argmax(proba, axis=1)
        binary_proba = collapse_multiclass_proba_to_confirmed_binary(proba)
        external_predictions[f"{name}_pred_3class"] = pred
        external_predictions[f"{name}_pred_3class_name"] = pd.Series(pred, index=external_predictions.index).map(TARGET_LABELS)
        external_predictions[f"{name}_pred_binary_confirmed"] = np.argmax(binary_proba, axis=1)
        external_predictions[f"{name}_prob_0"] = proba[:, 0]
        external_predictions[f"{name}_prob_1"] = proba[:, 1]
        external_predictions[f"{name}_prob_2"] = proba[:, 2]
        external_predictions[f"{name}_severity_score"] = external_severity[name]
    external_predictions.to_excel(external_dir / "external_test_variant_predictions.xlsx", index=False)

    model_manifest = save_model_bundles(model_dir, model_bundles)
    (artifacts_dir / "external_mapping_report.json").write_text(
        json.dumps(mapping_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "external_variant_metric_details.json").write_text(
        json.dumps(external_metric_detail, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "saved_model_manifest.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "external_variant_artifacts.json").write_text(
        json.dumps(external_artifacts, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "external_input_path": str(external_input_path),
        "external_output_dir": str(external_dir),
        "saved_model_dir": str(model_dir),
        "n_external_total": int(len(X_external_raw)),
        "n_external_labeled": int(y_external.notna().sum()),
        "best_external_variant_by_macro_f1_binary": (
            str(external_metrics_df.iloc[0]["variant"]) if not external_metrics_df.empty else None
        ),
    }


def main() -> None:
    input_path, output_dir = parse_args()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    artifacts_dir = output_dir / "artifacts"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(input_path)
    X_train_raw, X_test_raw, y_train, y_test = split_data(prepared.X, prepared.y)

    variant_results: list[dict[str, Any]] = []
    probas_map: dict[str, np.ndarray] = {}
    severity_map: dict[str, np.ndarray] = {}
    artifact_summary: dict[str, Any] = {}
    model_store: dict[str, Any] = {}
    feature_name_store: dict[str, list[str]] = {}

    # Baseline
    X_train_base_df, X_test_base_df = preprocess_pair(X_train_raw, X_test_raw)
    baseline_discrete = get_discrete_features(X_train_raw)
    baseline_model, baseline_proba = fit_baseline_xgb(
        X_train_base_df,
        y_train,
        X_test_base_df,
        discrete_numeric_features=baseline_discrete,
    )
    baseline_metrics = evaluate_predictions(y_test, baseline_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "baseline_xgboost", **baseline_metrics})
    probas_map["baseline_xgboost"] = baseline_proba
    severity_map["baseline_xgboost"] = np.clip(
        (baseline_proba[:, 2] + 2.0 * baseline_proba[:, 1]) / 2.0,
        0.0,
        1.0,
    )
    model_store["baseline_xgboost"] = baseline_model
    feature_name_store["baseline_xgboost"] = list(X_train_base_df.columns)

    # Variant 1: DML-inspired
    X_train_dml_raw, X_test_dml_raw, dml_artifacts = build_dml_features(X_train_raw, X_test_raw)
    X_train_dml_df, X_test_dml_df = preprocess_pair(X_train_dml_raw, X_test_dml_raw)
    dml_discrete = get_discrete_features(X_train_dml_raw)
    dml_model, dml_proba = fit_baseline_xgb(
        X_train_dml_df,
        y_train,
        X_test_dml_df,
        discrete_numeric_features=dml_discrete,
    )
    dml_metrics = evaluate_predictions(y_test, dml_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "dml_xgboost", **dml_metrics})
    probas_map["dml_xgboost"] = dml_proba
    severity_map["dml_xgboost"] = np.clip((dml_proba[:, 2] + 2.0 * dml_proba[:, 1]) / 2.0, 0.0, 1.0)
    artifact_summary["dml_xgboost"] = dml_artifacts
    model_store["dml_xgboost"] = dml_model
    feature_name_store["dml_xgboost"] = list(X_train_dml_df.columns)

    # Variant 2: Adaptive-CaRe inspired
    feature_weights = build_adaptive_feature_weights(list(X_train_base_df.columns))
    care_model, care_proba = fit_weighted_xgb_booster(
        X_train_base_df,
        y_train,
        X_test_base_df,
        feature_weights=feature_weights,
        adaptive=True,
        discrete_numeric_features=baseline_discrete,
    )
    care_metrics = evaluate_predictions(y_test, care_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "adaptive_care_xgboost", **care_metrics})
    probas_map["adaptive_care_xgboost"] = care_proba
    severity_map["adaptive_care_xgboost"] = np.clip((care_proba[:, 2] + 2.0 * care_proba[:, 1]) / 2.0, 0.0, 1.0)
    artifact_summary["adaptive_care_xgboost"] = {
        "feature_weight_summary": {
            "min": float(feature_weights.min()),
            "max": float(feature_weights.max()),
            "mean": float(feature_weights.mean()),
        }
    }
    model_store["adaptive_care_xgboost"] = care_model
    feature_name_store["adaptive_care_xgboost"] = list(X_train_base_df.columns)

    # Variant 3: Ordinal
    ordinal_model, ordinal_proba = fit_ordinal_xgboost(
        X_train_base_df,
        X_test_base_df,
        y_train,
        discrete_numeric_features=baseline_discrete,
    )
    ordinal_metrics = evaluate_predictions(y_test, ordinal_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "ordinal_xgboost", **ordinal_metrics})
    probas_map["ordinal_xgboost"] = ordinal_proba
    severity_map["ordinal_xgboost"] = np.clip((ordinal_proba[:, 2] + 2.0 * ordinal_proba[:, 1]) / 2.0, 0.0, 1.0)
    model_store["ordinal_xgboost"] = ordinal_model
    feature_name_store["ordinal_xgboost"] = list(X_train_base_df.columns)

    # Variant 3b: Ordinal V2 with cost-matrix approximation
    ordinal_v2_model, ordinal_v2_proba = fit_ordinal_xgboost_v2(
        X_train_base_df,
        X_test_base_df,
        y_train,
        discrete_numeric_features=baseline_discrete,
    )
    ordinal_v2_metrics = evaluate_predictions(y_test, ordinal_v2_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "ordinal_xgboost_v2", **ordinal_v2_metrics})
    probas_map["ordinal_xgboost_v2"] = ordinal_v2_proba
    severity_map["ordinal_xgboost_v2"] = np.clip(
        (ordinal_v2_proba[:, 2] + 2.0 * ordinal_v2_proba[:, 1]) / 2.0,
        0.0,
        1.0,
    )
    model_store["ordinal_xgboost_v2"] = ordinal_v2_model
    feature_name_store["ordinal_xgboost_v2"] = list(X_train_base_df.columns)
    artifact_summary["ordinal_xgboost_v2"] = {
        "cost_matrix": ordinal_v2_model["cost_matrix"],
        "blend": ordinal_v2_model["blend"],
        "temperature": ordinal_v2_model["temperature"],
    }

    # Variant 3c: DML + Ordinal
    dml_ordinal_model, dml_ordinal_proba = fit_ordinal_xgboost_v2(
        X_train_dml_df,
        X_test_dml_df,
        y_train,
        discrete_numeric_features=dml_discrete,
    )
    dml_ordinal_metrics = evaluate_predictions(y_test, dml_ordinal_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "dml_ordinal_xgboost", **dml_ordinal_metrics})
    probas_map["dml_ordinal_xgboost"] = dml_ordinal_proba
    severity_map["dml_ordinal_xgboost"] = np.clip(
        (dml_ordinal_proba[:, 2] + 2.0 * dml_ordinal_proba[:, 1]) / 2.0,
        0.0,
        1.0,
    )
    model_store["dml_ordinal_xgboost"] = dml_ordinal_model
    feature_name_store["dml_ordinal_xgboost"] = list(X_train_dml_df.columns)
    artifact_summary["dml_ordinal_xgboost"] = {
        **dml_artifacts,
        "cost_matrix": dml_ordinal_model["cost_matrix"],
        "blend": dml_ordinal_model["blend"],
        "temperature": dml_ordinal_model["temperature"],
    }

    # Variant 4: Physiology-informed
    X_train_phy_raw = add_physiology_features(X_train_raw)
    X_test_phy_raw = add_physiology_features(X_test_raw)
    X_train_phy_df, X_test_phy_df = preprocess_pair(X_train_phy_raw, X_test_phy_raw)
    phy_discrete = get_discrete_features(X_train_phy_raw)
    phy_model, phy_proba = fit_baseline_xgb(
        X_train_phy_df,
        y_train,
        X_test_phy_df,
        discrete_numeric_features=phy_discrete,
    )
    phy_metrics = evaluate_predictions(y_test, phy_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "physiology_informed_xgboost", **phy_metrics})
    probas_map["physiology_informed_xgboost"] = phy_proba
    severity_map["physiology_informed_xgboost"] = np.clip(
        (phy_proba[:, 2] + 2.0 * phy_proba[:, 1]) / 2.0,
        0.0,
        1.0,
    )
    model_store["physiology_informed_xgboost"] = phy_model
    feature_name_store["physiology_informed_xgboost"] = list(X_train_phy_df.columns)
    artifact_summary["physiology_informed_xgboost"] = {
        "added_features": [c for c in X_train_phy_raw.columns if c not in X_train_raw.columns]
    }

    # Variant 5: Ordinal with relaxed EMD
    ordinal_emd_model, ordinal_emd_proba, ordinal_emd_severity = fit_custom_softprob_booster(
        X_train_base_df,
        X_test_base_df,
        y_train,
        discrete_numeric_features=baseline_discrete,
        objective_mode="emd",
        use_fw_gpl=False,
        seed=RANDOM_STATE + 20,
    )
    ordinal_emd_metrics = evaluate_predictions(y_test, ordinal_emd_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "ordinal_emd_xgboost", **ordinal_emd_metrics})
    probas_map["ordinal_emd_xgboost"] = ordinal_emd_proba
    severity_map["ordinal_emd_xgboost"] = ordinal_emd_severity
    model_store["ordinal_emd_xgboost"] = ordinal_emd_model["booster"]
    feature_name_store["ordinal_emd_xgboost"] = list(X_train_base_df.columns)
    artifact_summary["ordinal_emd_xgboost"] = {
        "objective_mode": ordinal_emd_model["objective_mode"],
        "use_fw_gpl": ordinal_emd_model["use_fw_gpl"],
        "train_size": ordinal_emd_model["train_size"],
    }

    # Variant 6: Ordinal with FW-GPL
    ordinal_fw_model, ordinal_fw_proba, ordinal_fw_severity = fit_custom_softprob_booster(
        X_train_base_df,
        X_test_base_df,
        y_train,
        discrete_numeric_features=baseline_discrete,
        objective_mode="soft_ce",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 21,
    )
    ordinal_fw_metrics = evaluate_predictions(y_test, ordinal_fw_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "ordinal_fw_gpl_xgboost", **ordinal_fw_metrics})
    probas_map["ordinal_fw_gpl_xgboost"] = ordinal_fw_proba
    severity_map["ordinal_fw_gpl_xgboost"] = ordinal_fw_severity
    model_store["ordinal_fw_gpl_xgboost"] = ordinal_fw_model["booster"]
    feature_name_store["ordinal_fw_gpl_xgboost"] = list(X_train_base_df.columns)
    artifact_summary["ordinal_fw_gpl_xgboost"] = {
        "objective_mode": ordinal_fw_model["objective_mode"],
        "use_fw_gpl": ordinal_fw_model["use_fw_gpl"],
        "train_size": ordinal_fw_model["train_size"],
    }

    # Variant 7: Ordinal with EMD + FW-GPL
    ordinal_emd_fw_model, ordinal_emd_fw_proba, ordinal_emd_fw_severity = fit_custom_softprob_booster(
        X_train_base_df,
        X_test_base_df,
        y_train,
        discrete_numeric_features=baseline_discrete,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 22,
    )
    ordinal_emd_fw_metrics = evaluate_predictions(y_test, ordinal_emd_fw_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "ordinal_emd_fw_gpl_xgboost", **ordinal_emd_fw_metrics})
    probas_map["ordinal_emd_fw_gpl_xgboost"] = ordinal_emd_fw_proba
    severity_map["ordinal_emd_fw_gpl_xgboost"] = ordinal_emd_fw_severity
    model_store["ordinal_emd_fw_gpl_xgboost"] = ordinal_emd_fw_model["booster"]
    feature_name_store["ordinal_emd_fw_gpl_xgboost"] = list(X_train_base_df.columns)
    artifact_summary["ordinal_emd_fw_gpl_xgboost"] = {
        "objective_mode": ordinal_emd_fw_model["objective_mode"],
        "use_fw_gpl": ordinal_emd_fw_model["use_fw_gpl"],
        "train_size": ordinal_emd_fw_model["train_size"],
    }

    # Variant 8: DML + EMD + FW-GPL + Ordinal
    dml_emd_fw_model, dml_emd_fw_proba, dml_emd_fw_severity = fit_custom_softprob_booster(
        X_train_dml_df,
        X_test_dml_df,
        y_train,
        discrete_numeric_features=dml_discrete,
        objective_mode="emd",
        use_fw_gpl=True,
        seed=RANDOM_STATE + 23,
    )
    dml_emd_fw_metrics = evaluate_predictions(y_test, dml_emd_fw_proba, TARGET_LABELS, [0, 1, 2])
    variant_results.append({"variant": "dml_ordinal_emd_fw_gpl_xgboost", **dml_emd_fw_metrics})
    probas_map["dml_ordinal_emd_fw_gpl_xgboost"] = dml_emd_fw_proba
    severity_map["dml_ordinal_emd_fw_gpl_xgboost"] = dml_emd_fw_severity
    model_store["dml_ordinal_emd_fw_gpl_xgboost"] = dml_emd_fw_model["booster"]
    feature_name_store["dml_ordinal_emd_fw_gpl_xgboost"] = list(X_train_dml_df.columns)
    artifact_summary["dml_ordinal_emd_fw_gpl_xgboost"] = {
        **dml_artifacts,
        "objective_mode": dml_emd_fw_model["objective_mode"],
        "use_fw_gpl": dml_emd_fw_model["use_fw_gpl"],
        "train_size": dml_emd_fw_model["train_size"],
    }

    metrics_df = pd.DataFrame(variant_results).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"], ascending=False
    )
    metrics_df.to_csv(tables_dir / "variant_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(tables_dir / "variant_metrics_v2.csv", index=False, encoding="utf-8-sig")
    plot_variant_comparison(metrics_df, figures_dir / "variant_comparison.png")
    plot_recall_heatmap(metrics_df, figures_dir / "variant_recall_heatmap.png")
    plot_roc_overview(y_test, probas_map, figures_dir / "variant_roc_overview.png")

    calibration_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    for name, proba in probas_map.items():
        calibration_rows.append({"variant": name, **evaluate_calibration_metrics(y_test, proba)})
        severity_summary = evaluate_severity_score(y_test, severity_map[name])
        severity_rows.append({"variant": name, **severity_summary})
    calibration_df = pd.DataFrame(calibration_rows).sort_values("ece")
    severity_summary_df = pd.DataFrame(severity_rows).sort_values("class_1_mean_severity", ascending=False)
    calibration_df.to_csv(tables_dir / "variant_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    severity_summary_df.to_csv(tables_dir / "variant_severity_summary.csv", index=False, encoding="utf-8-sig")
    plot_calibration_comparison(calibration_df, figures_dir / "variant_calibration_comparison.png")

    severity_long_rows: list[dict[str, Any]] = []
    for name, scores in severity_map.items():
        for idx, score in enumerate(scores):
            severity_long_rows.append(
                {
                    "variant": name,
                    "sample_index": int(y_test.index[idx]),
                    "true_label": int(y_test.iloc[idx]),
                    "severity_score": float(score),
                }
            )
    severity_long_df = pd.DataFrame(severity_long_rows)
    severity_long_df.to_csv(tables_dir / "variant_severity_long.csv", index=False, encoding="utf-8-sig")
    plot_severity_distributions(
        severity_long_df,
        figures_dir / "severity_score_violin.png",
        figures_dir / "severity_score_boxplot.png",
    )
    (artifacts_dir / "clinical_cost_matrix.json").write_text(
        json.dumps(
            {
                "ordered_classes": ORDERED_CLASSES,
                "cost_matrix": CHAIN_COST_MATRIX.tolist(),
                "fw_gpl_templates": {str(k): v.tolist() for k, v in FW_GPL_SOFT_LABELS.items()},
                "boundary_weights": {str(k): v.tolist() for k, v in BOUNDARY_WEIGHT_LOOKUP.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "ordinal_loss_config.json").write_text(
        json.dumps(
            {
                "objective_modes": ["soft_ce", "emd"],
                "dml_combined_variant": "dml_ordinal_emd_fw_gpl_xgboost",
                "continuous_severity_definition": "E[rank]/2 with ordered rank [0,2,1]",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_row = metrics_df.iloc[0]
    best_name = str(best_row["variant"])
    best_proba = probas_map[best_name]
    best_pred = np.argmax(best_proba, axis=1)
    plot_confusion_heatmap(y_test, best_pred, figures_dir / "best_variant_confusion_heatmap.png")
    plot_multiclass_roc(y_test, best_proba, figures_dir / "best_variant_multiclass_roc.png")
    build_best_variant_feature_importance(
        best_name,
        model_store[best_name],
        feature_name_store[best_name],
        output_dir,
    )

    patient_predictions = X_test_raw.copy()
    patient_predictions["true_label"] = y_test
    patient_predictions["true_label_name"] = y_test.map(TARGET_LABELS)
    for name, proba in probas_map.items():
        patient_predictions[f"{name}_pred"] = np.argmax(proba, axis=1)
        patient_predictions[f"{name}_prob_0"] = proba[:, 0]
        patient_predictions[f"{name}_prob_1"] = proba[:, 1]
        patient_predictions[f"{name}_prob_2"] = proba[:, 2]
        patient_predictions[f"{name}_severity_score"] = severity_map[name]
    patient_predictions.to_excel(output_dir / "test_set_variant_predictions.xlsx", index=True)

    mainline_calibration_summary = run_mainline_calibration_experiment(
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        output_dir,
    )
    strict_prospective_summary = run_strict_prospective_experiment(
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        output_dir,
    )

    summary = {
        "n_samples": int(len(prepared.X)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "output_dir": str(output_dir),
        "external_test_evaluation_enabled": False,
        "best_variant": best_name,
        "best_metrics": {
            key: value
            for key, value in best_row.to_dict().items()
            if key != "classification_report" and key != "confusion_matrix"
        },
        "variant_rank": metrics_df[["variant", "accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"]].to_dict(
            orient="records"
        ),
        "calibration_rank": calibration_df.to_dict(orient="records"),
        "severity_summary": severity_summary_df.to_dict(orient="records"),
        "variant_artifacts": artifact_summary,
        "mainline_calibration_experiment": mainline_calibration_summary,
        "strict_prospective_experiment": strict_prospective_summary,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("XGBoost 因果/机制变体实验完成")
    print(metrics_df[["variant", "accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"]].to_string(index=False))
    print(f"最佳方案: {best_name}")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
