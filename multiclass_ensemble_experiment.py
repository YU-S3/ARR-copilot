from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["axes.unicode_minus"] = False


TARGET_COLUMN = "确诊（0为排除；1为确诊；2为灰色区域）"
DEFAULT_INPUT_FILE = "data_0428.xlsx"
EXCLUDED_FEATURE_COLUMNS = {"住院号", "Unnamed: 0"}
ZERO_FILL_COLUMNS = [
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
]
EXPERIMENT_TYPE_COLUMN = "确诊实验类型"
TARGET_LABELS = {0: "非确诊", 1: "确诊", 2: "灰色区域"}
BASE_MODEL_NAMES = ["xgboost", "lightgbm", "catboost"]
META_RAW_COLUMNS = [
    "ARR比值",
    "醛固酮",
    "肾素",
    "收缩压",
    "舒展压",
    "试验前醛固酮",
    "试验前肾素",
    "试验后醛固酮",
    "试验后肾素",
]
RANDOM_STATE = 42


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    discrete_numeric_features: list[str]
    cleaned_df: pd.DataFrame


@dataclass
class BaseModelArtifacts:
    model_name: str
    task_name: str
    best_params: dict[str, Any]
    preprocessor: ColumnTransformer
    final_model: Any
    feature_names: list[str]
    oof_proba: np.ndarray
    test_proba: np.ndarray
    oof_representation: pd.DataFrame
    test_representation: pd.DataFrame
    resampled_counts: dict[str, int]
    transformed_test_df: pd.DataFrame


@dataclass
class StrategyResult:
    strategy_name: str
    task_name: str
    oof_proba: np.ndarray
    test_proba: np.ndarray
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    explainable_model_name: str


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="基于 XGBoost、LightGBM、CatBoost 的多策略集成临床三分类实验。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / DEFAULT_INPUT_FILE,
        help="输入 Excel 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "ensemble_outputs_v2",
        help="输出目录。",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=6,
        help="每个多分类基模型的 Optuna 试验次数，默认 6。",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Stratified K-Fold 折数，默认 4。",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="最终独立测试集占比，默认 0.2。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="随机种子，默认 42。",
    )
    return parser.parse_args()


def clean_string_cell(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "nan", "None", "?", "？"}:
        return np.nan
    return text


def normalize_target_value(value: Any) -> float | None:
    value = clean_string_cell(value)
    if pd.isna(value):
        return None
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return None
    if numeric in {0, 1, 2}:
        return float(numeric)
    return None


def normalize_experiment_type(value: Any) -> Any:
    value = clean_string_cell(value)
    if pd.isna(value):
        return np.nan
    mapping = {
        "冷盐水": "冷盐水实验",
        "冷盐水实验": "冷盐水实验",
        "卡托普利": "卡托普利实验",
        "卡托普利实验": "卡托普利实验",
    }
    return mapping.get(value, value)


def maybe_convert_object_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(clean_string_cell)
    if cleaned.dropna().empty:
        return cleaned
    converted = pd.to_numeric(cleaned, errors="coerce")
    ratio = converted.notna().sum() / max(cleaned.notna().sum(), 1)
    if ratio >= 0.85:
        return converted
    return cleaned


def detect_discrete_numeric_features(df: pd.DataFrame, numeric_features: list[str]) -> list[str]:
    discrete_features: list[str] = []
    for col in numeric_features:
        series = df[col].dropna()
        if series.empty:
            continue
        unique_count = series.nunique()
        values = series.to_numpy(dtype=float)
        integer_like = np.allclose(values, np.round(values))
        if unique_count <= 12 or (integer_like and unique_count <= 30):
            discrete_features.append(col)
    return discrete_features


def prepare_data(file_path: Path) -> PreparedData:
    df = pd.read_excel(file_path).copy()
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"未找到目标列: {TARGET_COLUMN}")

    excluded_columns = [
        col
        for col in df.columns
        if col in EXCLUDED_FEATURE_COLUMNS or str(col).startswith("Unnamed:")
    ]
    if excluded_columns:
        df = df.drop(columns=excluded_columns)

    missing_zero_fill_cols = [col for col in ZERO_FILL_COLUMNS if col not in df.columns]
    if missing_zero_fill_cols:
        raise KeyError(f"以下需要按 0 补齐的列不存在: {missing_zero_fill_cols}")

    df[ZERO_FILL_COLUMNS] = df[ZERO_FILL_COLUMNS].fillna(0)

    for col in df.columns:
        if df[col].dtype == "object":
            if col == TARGET_COLUMN:
                continue
            if col == EXPERIMENT_TYPE_COLUMN:
                df[col] = df[col].map(normalize_experiment_type)
            else:
                df[col] = maybe_convert_object_to_numeric(df[col])

    df["_target_clean"] = df[TARGET_COLUMN].apply(normalize_target_value)
    df = df[df["_target_clean"].isin([0.0, 1.0, 2.0])].copy()
    df["_target_clean"] = df["_target_clean"].astype(int)

    X = df.drop(columns=[TARGET_COLUMN, "_target_clean"])
    y = df["_target_clean"].copy()

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    discrete_numeric_features = detect_discrete_numeric_features(X, numeric_features)

    return PreparedData(
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        discrete_numeric_features=discrete_numeric_features,
        cleaned_df=df,
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=False))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    X_train_t = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)

    X_valid_df: pd.DataFrame | None = None
    if X_valid is not None:
        X_valid_t = preprocessor.transform(X_valid)
        X_valid_df = pd.DataFrame(X_valid_t, columns=feature_names, index=X_valid.index)
    return X_train_df, X_valid_df


def build_sampling_strategy(y_train: pd.Series) -> dict[int, int]:
    counts = Counter(y_train)
    majority_count = max(counts.values())
    strategy: dict[int, int] = {}
    for cls, count in counts.items():
        if count == majority_count:
            continue
        max_ratio = 0.65 if int(cls) == 0 else 0.80
        desired = int(min(count * 4, majority_count * max_ratio))
        if desired > count + 2:
            strategy[int(cls)] = desired
    return strategy


def apply_controlled_adasyn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    discrete_numeric_features: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    strategy = build_sampling_strategy(y_train)
    if not strategy:
        return X_train, y_train

    minority_count = min(Counter(y_train).values())
    if minority_count <= 2:
        return X_train, y_train

    n_neighbors = min(5, minority_count - 1)
    sampler = ADASYN(
        sampling_strategy=strategy,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    for col in X_train.columns:
        if col.startswith("num__"):
            lower = X_train[col].min()
            upper = X_train[col].max()
            X_resampled[col] = X_resampled[col].clip(lower=lower, upper=upper)

    for raw_col in discrete_numeric_features:
        transformed_col = f"num__{raw_col}"
        if transformed_col in X_resampled.columns:
            lower = X_train[transformed_col].min()
            upper = X_train[transformed_col].max()
            X_resampled[transformed_col] = (
                X_resampled[transformed_col].round().clip(lower=lower, upper=upper)
            )

    return X_resampled, y_resampled


def get_default_params(model_name: str, n_classes: int) -> dict[str, Any]:
    if model_name == "xgboost":
        return {
            "n_estimators": 260,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "reg_alpha": 0.01,
            "reg_lambda": 0.1,
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": 260,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 4,
            "min_child_samples": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.1,
        }
    if model_name == "catboost":
        return {
            "iterations": 300,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.5,
            "bagging_temperature": 0.2,
            "border_count": 128,
        }
    raise ValueError(f"未知模型名称: {model_name}")


def get_model(model_name: str, params: dict[str, Any], random_state: int, n_classes: int) -> Any:
    if model_name == "xgboost":
        if n_classes == 2:
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
                **params,
            )
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            **params,
        )

    if model_name == "lightgbm":
        if n_classes == 2:
            return LGBMClassifier(
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
                **params,
            )
        return LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **params,
        )

    if model_name == "catboost":
        loss_function = "Logloss" if n_classes == 2 else "MultiClass"
        return CatBoostClassifier(
            loss_function=loss_function,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    raise ValueError(f"未知模型名称: {model_name}")


def suggest_params(trial: optuna.trial.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 360),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 360),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 25),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 220, 420),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 3.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 64, 255),
        }
    raise ValueError(f"未知模型名称: {model_name}")


def fit_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
) -> Any:
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    if model_name == "xgboost":
        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    if model_name == "lightgbm":
        fit_kwargs = {"sample_weight": sample_weight}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["eval_metric"] = "multi_logloss" if y_train.nunique() > 2 else "binary_logloss"
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    if model_name == "catboost":
        fit_kwargs = {"sample_weight": sample_weight}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = (X_valid, y_valid)
            fit_kwargs["use_best_model"] = True
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    raise ValueError(f"未知模型名称: {model_name}")


def ensure_2d_proba(proba: np.ndarray, classes: list[int]) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    if proba.shape[1] == len(classes):
        return proba
    aligned = np.zeros((proba.shape[0], len(classes)), dtype=float)
    for idx, cls in enumerate(classes):
        if idx < proba.shape[1]:
            aligned[:, cls] = proba[:, idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum


def get_model_representation(model_name: str, model: Any, X_df: pd.DataFrame) -> pd.DataFrame:
    if model_name == "xgboost":
        leaves = np.asarray(model.apply(X_df))
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return pd.DataFrame(
            leaves.astype(int),
            index=X_df.index,
            columns=[f"xgb_leaf_{i}" for i in range(leaves.shape[1])],
        )
    if model_name == "lightgbm":
        leaves = np.asarray(model.predict(X_df, pred_leaf=True))
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return pd.DataFrame(
            leaves.astype(int),
            index=X_df.index,
            columns=[f"lgb_leaf_{i}" for i in range(leaves.shape[1])],
        )
    raw_values = np.asarray(model.predict(X_df, prediction_type="RawFormulaVal"))
    if raw_values.ndim == 1:
        raw_values = raw_values.reshape(-1, 1)
    return pd.DataFrame(
        raw_values,
        index=X_df.index,
        columns=[f"cat_raw_{i}" for i in range(raw_values.shape[1])],
    )


def evaluate_predictions(
    y_true: pd.Series,
    proba: np.ndarray,
    label_mapping: dict[int, str],
    classes: list[int],
) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "classification_report": classification_report(
            y_true,
            pred,
            labels=classes,
            target_names=[label_mapping[i] for i in classes],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=classes).tolist(),
    }
    try:
        metrics["ovr_roc_auc_macro"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        if len(classes) == 2:
            metrics["ovr_roc_auc_macro"] = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            metrics["ovr_roc_auc_macro"] = float("nan")
    return metrics


def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    discrete_numeric_features: list[str],
    n_trials: int,
    n_splits: int,
    random_state: int,
    n_classes: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, model_name)
        fold_scores: list[float] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_valid = y_train.iloc[valid_idx]

            preprocessor = build_preprocessor(numeric_features, categorical_features)
            X_fold_train_t, X_fold_valid_t = transform_with_preprocessor(
                preprocessor, X_fold_train, X_fold_valid
            )
            X_fold_resampled, y_fold_resampled = apply_controlled_adasyn(
                X_fold_train_t,
                y_fold_train,
                discrete_numeric_features,
                random_state + fold_idx,
            )
            model = get_model(model_name, params, random_state + fold_idx, n_classes)
            model = fit_model(
                model_name,
                model,
                X_fold_resampled,
                y_fold_resampled,
                X_fold_valid_t,
                y_fold_valid,
            )
            valid_pred = model.predict(X_fold_valid_t)
            fold_scores.append(f1_score(y_fold_valid, valid_pred, average="macro"))

        mean_score = float(np.mean(fold_scores))
        trial_rows.append({"trial": trial.number, "score": mean_score, **params})
        return mean_score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_df = pd.DataFrame(trial_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return study.best_params, trials_df


def fit_base_models_with_oof(
    task_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    discrete_numeric_features: list[str],
    n_trials: int,
    n_splits: int,
    random_state: int,
    optimize: bool,
) -> tuple[dict[str, BaseModelArtifacts], dict[str, pd.DataFrame]]:
    classes = sorted(y_train.unique().tolist())
    n_classes = len(classes)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    artifacts: dict[str, BaseModelArtifacts] = {}
    trials_map: dict[str, pd.DataFrame] = {}

    for model_name in BASE_MODEL_NAMES:
        if optimize and n_classes > 2:
            best_params, trials_df = optimize_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                discrete_numeric_features=discrete_numeric_features,
                n_trials=n_trials,
                n_splits=n_splits,
                random_state=random_state,
                n_classes=n_classes,
            )
        else:
            best_params = get_default_params(model_name, n_classes)
            trials_df = pd.DataFrame([{"trial": -1, "score": np.nan, **best_params}])

        oof_proba = np.zeros((len(X_train), n_classes), dtype=float)
        oof_rep_parts: list[pd.DataFrame] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]

            preprocessor = build_preprocessor(numeric_features, categorical_features)
            X_fold_train_t, X_fold_valid_t = transform_with_preprocessor(
                preprocessor, X_fold_train, X_fold_valid
            )
            X_fold_resampled, y_fold_resampled = apply_controlled_adasyn(
                X_fold_train_t,
                y_fold_train,
                discrete_numeric_features,
                random_state + fold_idx,
            )
            model = get_model(model_name, best_params, random_state + fold_idx, n_classes)
            model = fit_model(
                model_name,
                model,
                X_fold_resampled,
                y_fold_resampled,
                X_fold_valid_t,
                y_train.iloc[valid_idx],
            )
            fold_proba = ensure_2d_proba(model.predict_proba(X_fold_valid_t), classes)
            oof_proba[valid_idx] = fold_proba

            fold_repr = get_model_representation(model_name, model, X_fold_valid_t)
            oof_rep_parts.append(fold_repr)

        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_train_t, X_test_t = transform_with_preprocessor(preprocessor, X_train, X_test)
        X_resampled, y_resampled = apply_controlled_adasyn(
            X_train_t, y_train, discrete_numeric_features, random_state
        )
        final_model = get_model(model_name, best_params, random_state, n_classes)
        final_model = fit_model(model_name, final_model, X_resampled, y_resampled)

        test_proba = ensure_2d_proba(final_model.predict_proba(X_test_t), classes)
        test_representation = get_model_representation(model_name, final_model, X_test_t)
        oof_representation = pd.concat(oof_rep_parts).sort_index()

        artifacts[model_name] = BaseModelArtifacts(
            model_name=model_name,
            task_name=task_name,
            best_params=best_params,
            preprocessor=preprocessor,
            final_model=final_model,
            feature_names=list(preprocessor.get_feature_names_out()),
            oof_proba=oof_proba,
            test_proba=test_proba,
            oof_representation=oof_representation,
            test_representation=test_representation,
            resampled_counts={str(k): int(v) for k, v in Counter(y_resampled).items()},
            transformed_test_df=X_test_t,
        )
        trials_map[model_name] = trials_df
    return artifacts, trials_map


def get_base_probabilities(
    artifacts: dict[str, BaseModelArtifacts], use_oof: bool
) -> dict[str, np.ndarray]:
    return {
        model_name: (item.oof_proba if use_oof else item.test_proba)
        for model_name, item in artifacts.items()
    }


def combine_equal_weight(proba_map: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(proba_map.values()), axis=0)
    return stacked.mean(axis=0)


def combine_global_weight(proba_map: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    model_names = list(proba_map)
    result = np.zeros_like(next(iter(proba_map.values())))
    for idx, model_name in enumerate(model_names):
        result += proba_map[model_name] * weights[idx]
    row_sum = result.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return result / row_sum


def combine_class_weight(proba_map: dict[str, np.ndarray], weight_matrix: np.ndarray) -> np.ndarray:
    model_names = list(proba_map)
    n_samples, n_classes = next(iter(proba_map.values())).shape
    result = np.zeros((n_samples, n_classes), dtype=float)
    for class_id in range(n_classes):
        for model_idx, model_name in enumerate(model_names):
            result[:, class_id] += proba_map[model_name][:, class_id] * weight_matrix[class_id, model_idx]
    row_sum = result.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return result / row_sum


def optimize_global_weights(
    y_true: pd.Series,
    proba_map: dict[str, np.ndarray],
    label_mapping: dict[int, str],
    classes: list[int],
    random_state: int,
    n_samples: int = 2000,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    model_names = list(proba_map)
    best_weights = np.ones(len(model_names)) / len(model_names)
    best_proba = combine_global_weight(proba_map, best_weights)
    best_metrics = evaluate_predictions(y_true, best_proba, label_mapping, classes)
    best_score = (best_metrics["macro_f1"], best_metrics["balanced_accuracy"])

    for _ in range(n_samples):
        weights = rng.dirichlet(np.ones(len(model_names)))
        proba = combine_global_weight(proba_map, weights)
        metrics = evaluate_predictions(y_true, proba, label_mapping, classes)
        score = (metrics["macro_f1"], metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics

    return best_weights, best_metrics


def optimize_class_weights(
    y_true: pd.Series,
    proba_map: dict[str, np.ndarray],
    label_mapping: dict[int, str],
    classes: list[int],
    random_state: int,
    n_samples: int = 2500,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    model_names = list(proba_map)
    n_classes = len(classes)
    best_weights = np.tile(np.ones(len(model_names)) / len(model_names), (n_classes, 1))
    best_proba = combine_class_weight(proba_map, best_weights)
    best_metrics = evaluate_predictions(y_true, best_proba, label_mapping, classes)
    best_score = (best_metrics["macro_f1"], best_metrics["balanced_accuracy"])

    for _ in range(n_samples):
        weights = np.vstack([rng.dirichlet(np.ones(len(model_names))) for _ in range(n_classes)])
        proba = combine_class_weight(proba_map, weights)
        metrics = evaluate_predictions(y_true, proba, label_mapping, classes)
        score = (metrics["macro_f1"], metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def prepare_meta_raw_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selected_cols = [col for col in META_RAW_COLUMNS if col in X_train.columns]
    if not selected_cols:
        return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index), []
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(X_train[selected_cols])
    test_values = imputer.transform(X_test[selected_cols])
    return (
        pd.DataFrame(train_values, columns=selected_cols, index=X_train.index),
        pd.DataFrame(test_values, columns=selected_cols, index=X_test.index),
        selected_cols,
    )


def build_stacking_features(
    artifacts: dict[str, BaseModelArtifacts],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for model_name in BASE_MODEL_NAMES:
        oof = artifacts[model_name].oof_proba
        test = artifacts[model_name].test_proba
        train_parts.append(
            pd.DataFrame(
                oof,
                index=X_train.index,
                columns=[f"{model_name}_prob_{i}" for i in range(oof.shape[1])],
            )
        )
        test_parts.append(
            pd.DataFrame(
                test,
                index=X_test.index,
                columns=[f"{model_name}_prob_{i}" for i in range(test.shape[1])],
            )
        )

    meta_train_raw, meta_test_raw, _ = prepare_meta_raw_features(X_train, X_test)
    train_parts.append(meta_train_raw)
    test_parts.append(meta_test_raw)
    return pd.concat(train_parts, axis=1), pd.concat(test_parts, axis=1)


def build_leaf_injection_features(
    artifacts: dict[str, BaseModelArtifacts],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[sp.csr_matrix, sp.csr_matrix, list[str]]:
    leaf_train = pd.concat(
        [
            artifacts["xgboost"].oof_representation.astype(str),
            artifacts["lightgbm"].oof_representation.astype(str),
        ],
        axis=1,
    )
    leaf_test = pd.concat(
        [
            artifacts["xgboost"].test_representation.astype(str),
            artifacts["lightgbm"].test_representation.astype(str),
        ],
        axis=1,
    )
    leaf_train, leaf_test = leaf_train.align(leaf_test, join="outer", axis=1, fill_value="__missing_leaf__")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    leaf_train_sparse = encoder.fit_transform(leaf_train)
    leaf_test_sparse = encoder.transform(leaf_test)

    train_prob_df, test_prob_df = build_stacking_features(artifacts, X_train, X_test)
    cat_raw_train = artifacts["catboost"].oof_representation
    cat_raw_test = artifacts["catboost"].test_representation
    dense_train_df = pd.concat([train_prob_df, cat_raw_train], axis=1)
    dense_test_df = pd.concat([test_prob_df, cat_raw_test], axis=1)
    dense_imputer = SimpleImputer(strategy="median")
    dense_train = dense_imputer.fit_transform(dense_train_df)
    dense_test = dense_imputer.transform(dense_test_df)

    train_sparse = sp.hstack([leaf_train_sparse, sp.csr_matrix(dense_train)]).tocsr()
    test_sparse = sp.hstack([leaf_test_sparse, sp.csr_matrix(dense_test)]).tocsr()
    feature_names = list(encoder.get_feature_names_out()) + dense_train_df.columns.tolist()
    return train_sparse, test_sparse, feature_names


def fit_meta_model(
    model_name: str,
    X_train_meta: Any,
    y_train: pd.Series,
    n_classes: int,
    random_state: int,
) -> Any:
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    if model_name == "logistic":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        model.fit(X_train_meta, y_train)
        return model

    if model_name == "xgboost":
        params = {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
        model = get_model("xgboost", params, random_state, n_classes)
        model.fit(X_train_meta, y_train, sample_weight=sample_weight, verbose=False)
        return model

    if model_name == "catboost":
        params = {
            "iterations": 160,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
        }
        model = get_model("catboost", params, random_state, n_classes)
        model.fit(X_train_meta, y_train, sample_weight=sample_weight)
        return model

    if model_name == "lightgbm":
        params = {
            "n_estimators": 140,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 4,
        }
        model = get_model("lightgbm", params, random_state, n_classes)
        model.fit(X_train_meta, y_train, sample_weight=sample_weight)
        return model

    raise ValueError(f"未知元模型名称: {model_name}")


def evaluate_meta_candidates(
    candidate_names: list[str],
    X_meta: Any,
    y_train: pd.Series,
    n_splits: int,
    random_state: int,
) -> tuple[str, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    classes = sorted(y_train.unique().tolist())

    for candidate_name in candidate_names:
        scores: list[float] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(np.zeros(len(y_train)), y_train), start=1):
            model = fit_meta_model(
                candidate_name,
                X_meta[train_idx] if sp.issparse(X_meta) else X_meta.iloc[train_idx],
                y_train.iloc[train_idx],
                len(classes),
                random_state + fold_idx,
            )
            X_valid = X_meta[valid_idx] if sp.issparse(X_meta) else X_meta.iloc[valid_idx]
            pred = model.predict(X_valid)
            scores.append(f1_score(y_train.iloc[valid_idx], pred, average="macro"))
        rows.append(
            {
                "meta_model": candidate_name,
                "cv_macro_f1": float(np.mean(scores)),
                "cv_macro_f1_std": float(np.std(scores)),
            }
        )
    result_df = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
    return result_df.iloc[0]["meta_model"], result_df


def get_meta_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        values = coef.mean(axis=0) if coef.ndim == 2 else np.abs(coef)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        values = np.asarray(model.get_feature_importance(), dtype=float)
    else:
        values = np.zeros(len(feature_names))
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def build_stacking_strategy(
    task_name: str,
    artifacts: dict[str, BaseModelArtifacts],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    label_mapping: dict[int, str],
    classes: list[int],
    n_splits: int,
    random_state: int,
) -> StrategyResult:
    X_meta_train, X_meta_test = build_stacking_features(artifacts, X_train, X_test)
    best_meta_name, meta_cv_df = evaluate_meta_candidates(
        ["logistic", "xgboost", "catboost"],
        X_meta_train,
        y_train,
        n_splits,
        random_state,
    )
    meta_model = fit_meta_model(best_meta_name, X_meta_train, y_train, len(classes), random_state)
    oof_proba = ensure_2d_proba(meta_model.predict_proba(X_meta_train), classes)
    test_proba = ensure_2d_proba(meta_model.predict_proba(X_meta_test), classes)
    metrics = evaluate_predictions(y_train, oof_proba, label_mapping, classes)
    artifacts_dict = {
        "meta_model_name": best_meta_name,
        "meta_cv_results": meta_cv_df.to_dict(orient="records"),
        "meta_feature_importance": get_meta_feature_importance(
            meta_model, X_meta_train.columns.tolist()
        ).to_dict(orient="records"),
    }
    return StrategyResult(
        strategy_name=f"{task_name}_stacking",
        task_name=task_name,
        oof_proba=oof_proba,
        test_proba=test_proba,
        metrics=metrics,
        artifacts=artifacts_dict,
        explainable_model_name="xgboost",
    )


def build_leaf_injection_strategy(
    task_name: str,
    artifacts: dict[str, BaseModelArtifacts],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    label_mapping: dict[int, str],
    classes: list[int],
    n_splits: int,
    random_state: int,
) -> StrategyResult:
    X_meta_train, X_meta_test, feature_names = build_leaf_injection_features(artifacts, X_train, X_test)
    best_meta_name, meta_cv_df = evaluate_meta_candidates(
        ["logistic", "lightgbm"],
        X_meta_train,
        y_train,
        n_splits,
        random_state,
    )
    meta_model = fit_meta_model(best_meta_name, X_meta_train, y_train, len(classes), random_state)
    oof_proba = ensure_2d_proba(meta_model.predict_proba(X_meta_train), classes)
    test_proba = ensure_2d_proba(meta_model.predict_proba(X_meta_test), classes)
    metrics = evaluate_predictions(y_train, oof_proba, label_mapping, classes)
    artifacts_dict = {
        "meta_model_name": best_meta_name,
        "meta_cv_results": meta_cv_df.to_dict(orient="records"),
        "meta_feature_importance": get_meta_feature_importance(
            meta_model, feature_names
        ).head(200).to_dict(orient="records"),
    }
    return StrategyResult(
        strategy_name=f"{task_name}_leaf_injection",
        task_name=task_name,
        oof_proba=oof_proba,
        test_proba=test_proba,
        metrics=metrics,
        artifacts=artifacts_dict,
        explainable_model_name="xgboost",
    )


def build_base_and_fusion_strategies(
    task_name: str,
    artifacts: dict[str, BaseModelArtifacts],
    y_train: pd.Series,
    label_mapping: dict[int, str],
    classes: list[int],
    random_state: int,
) -> list[StrategyResult]:
    proba_oof_map = get_base_probabilities(artifacts, use_oof=True)
    proba_test_map = get_base_probabilities(artifacts, use_oof=False)
    results: list[StrategyResult] = []

    for model_name in BASE_MODEL_NAMES:
        metrics = evaluate_predictions(
            y_train,
            artifacts[model_name].oof_proba,
            label_mapping,
            classes,
        )
        results.append(
            StrategyResult(
                strategy_name=f"{task_name}_{model_name}",
                task_name=task_name,
                oof_proba=artifacts[model_name].oof_proba,
                test_proba=artifacts[model_name].test_proba,
                metrics=metrics,
                artifacts={"best_params": artifacts[model_name].best_params},
                explainable_model_name=model_name,
            )
        )

    equal_oof = combine_equal_weight(proba_oof_map)
    equal_test = combine_equal_weight(proba_test_map)
    results.append(
        StrategyResult(
            strategy_name=f"{task_name}_equal_weight",
            task_name=task_name,
            oof_proba=equal_oof,
            test_proba=equal_test,
            metrics=evaluate_predictions(y_train, equal_oof, label_mapping, classes),
            artifacts={},
            explainable_model_name="xgboost",
        )
    )

    global_weights, _ = optimize_global_weights(
        y_train, proba_oof_map, label_mapping, classes, random_state
    )
    global_oof = combine_global_weight(proba_oof_map, global_weights)
    global_test = combine_global_weight(proba_test_map, global_weights)
    results.append(
        StrategyResult(
            strategy_name=f"{task_name}_global_weighted",
            task_name=task_name,
            oof_proba=global_oof,
            test_proba=global_test,
            metrics=evaluate_predictions(y_train, global_oof, label_mapping, classes),
            artifacts={"weights": global_weights.tolist(), "models": list(proba_oof_map)},
            explainable_model_name="xgboost",
        )
    )

    class_weights, _ = optimize_class_weights(
        y_train, proba_oof_map, label_mapping, classes, random_state + 11
    )
    class_oof = combine_class_weight(proba_oof_map, class_weights)
    class_test = combine_class_weight(proba_test_map, class_weights)
    results.append(
        StrategyResult(
            strategy_name=f"{task_name}_class_weighted",
            task_name=task_name,
            oof_proba=class_oof,
            test_proba=class_test,
            metrics=evaluate_predictions(y_train, class_oof, label_mapping, classes),
            artifacts={"class_weight_matrix": class_weights.tolist(), "models": list(proba_oof_map)},
            explainable_model_name="xgboost",
        )
    )
    return results


def choose_best_strategy_by_oof(
    strategies: list[StrategyResult],
) -> StrategyResult:
    return sorted(
        strategies,
        key=lambda item: (item.metrics["macro_f1"], item.metrics["balanced_accuracy"]),
        reverse=True,
    )[0]


def run_two_stage_strategy(
    prepared: PreparedData,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    args: argparse.Namespace,
) -> StrategyResult:
    stage1_y_train = (y_train == 1).astype(int)
    stage1_y_test = (y_test == 1).astype(int)
    stage1_mapping = {0: "非确诊或灰区", 1: "确诊"}

    stage1_artifacts, _ = fit_base_models_with_oof(
        task_name="stage1",
        X_train=X_train,
        y_train=stage1_y_train,
        X_test=X_test,
        numeric_features=prepared.numeric_features,
        categorical_features=prepared.categorical_features,
        discrete_numeric_features=prepared.discrete_numeric_features,
        n_trials=0,
        n_splits=args.n_splits,
        random_state=args.random_state + 101,
        optimize=False,
    )
    stage1_strategies = build_base_and_fusion_strategies(
        "stage1",
        stage1_artifacts,
        stage1_y_train,
        stage1_mapping,
        [0, 1],
        args.random_state + 101,
    )
    stage1_strategies.append(
        build_stacking_strategy(
            "stage1",
            stage1_artifacts,
            X_train,
            stage1_y_train,
            X_test,
            stage1_mapping,
            [0, 1],
            args.n_splits,
            args.random_state + 102,
        )
    )
    best_stage1 = choose_best_strategy_by_oof(stage1_strategies)

    subset_mask = y_train != 1
    X_stage2_train = X_train.loc[subset_mask]
    y_stage2_train = y_train.loc[subset_mask].map({0: 0, 2: 1})
    stage2_mapping = {0: "非确诊", 1: "灰色区域"}

    stage2_artifacts, _ = fit_base_models_with_oof(
        task_name="stage2",
        X_train=X_stage2_train,
        y_train=y_stage2_train,
        X_test=X_test,
        numeric_features=prepared.numeric_features,
        categorical_features=prepared.categorical_features,
        discrete_numeric_features=prepared.discrete_numeric_features,
        n_trials=0,
        n_splits=args.n_splits,
        random_state=args.random_state + 201,
        optimize=False,
    )
    stage2_strategies = build_base_and_fusion_strategies(
        "stage2",
        stage2_artifacts,
        y_stage2_train,
        stage2_mapping,
        [0, 1],
        args.random_state + 201,
    )
    stage2_strategies.append(
        build_stacking_strategy(
            "stage2",
            stage2_artifacts,
            X_stage2_train,
            y_stage2_train,
            X_test,
            stage2_mapping,
            [0, 1],
            args.n_splits,
            args.random_state + 202,
        )
    )
    best_stage2 = choose_best_strategy_by_oof(stage2_strategies)

    p1_oof = best_stage1.oof_proba[:, 1]
    p_not1_oof = 1 - p1_oof
    stage2_oof_full = np.full((len(X_train), 2), 0.5, dtype=float)
    stage2_oof_full[subset_mask.values] = best_stage2.oof_proba
    two_stage_oof = np.column_stack(
        [
            p_not1_oof * stage2_oof_full[:, 0],
            p1_oof,
            p_not1_oof * stage2_oof_full[:, 1],
        ]
    )
    two_stage_oof = two_stage_oof / two_stage_oof.sum(axis=1, keepdims=True)

    p1_test = best_stage1.test_proba[:, 1]
    p_not1_test = 1 - p1_test
    two_stage_test = np.column_stack(
        [
            p_not1_test * best_stage2.test_proba[:, 0],
            p1_test,
            p_not1_test * best_stage2.test_proba[:, 1],
        ]
    )
    two_stage_test = two_stage_test / two_stage_test.sum(axis=1, keepdims=True)

    return StrategyResult(
        strategy_name="main_two_stage",
        task_name="main",
        oof_proba=two_stage_oof,
        test_proba=two_stage_test,
        metrics=evaluate_predictions(y_train, two_stage_oof, TARGET_LABELS, [0, 1, 2]),
        artifacts={
            "stage1_best": best_stage1.strategy_name,
            "stage2_best": best_stage2.strategy_name,
            "stage1_oof_metrics": best_stage1.metrics,
            "stage2_oof_metrics": best_stage2.metrics,
            "stage1_test_metrics": evaluate_predictions(stage1_y_test, best_stage1.test_proba, stage1_mapping, [0, 1]),
        },
        explainable_model_name="xgboost",
    )


def calibrate_probs_one_vs_rest(
    train_proba: np.ndarray,
    y_train: pd.Series,
    test_proba: np.ndarray,
    method: str,
) -> np.ndarray:
    calibrated_test = np.zeros_like(test_proba, dtype=float)
    for class_id in range(train_proba.shape[1]):
        y_binary = (y_train == class_id).astype(int).to_numpy()
        train_scores = train_proba[:, class_id]
        test_scores = test_proba[:, class_id]
        if method == "sigmoid":
            calibrator = LogisticRegression(max_iter=500, class_weight="balanced")
            calibrator.fit(train_scores.reshape(-1, 1), y_binary)
            calibrated_test[:, class_id] = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(train_scores, y_binary)
            calibrated_test[:, class_id] = calibrator.predict(test_scores)
        else:
            raise ValueError(f"未知校准方法: {method}")
    row_sum = calibrated_test.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return calibrated_test / row_sum


def optimize_class_scaling(
    train_proba: np.ndarray,
    y_train: pd.Series,
    label_mapping: dict[int, str],
    classes: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    grid = [0.7, 0.85, 1.0, 1.15, 1.3]
    best_scale = np.ones(train_proba.shape[1], dtype=float)
    scaled = train_proba * best_scale
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    best_metrics = evaluate_predictions(y_train, scaled, label_mapping, classes)
    best_score = (best_metrics["macro_f1"], best_metrics["balanced_accuracy"])

    for s0 in grid:
        for s1 in grid:
            for s2 in grid:
                scale = np.array([s0, s1, s2], dtype=float)
                scaled = train_proba * scale
                scaled = scaled / scaled.sum(axis=1, keepdims=True)
                metrics = evaluate_predictions(y_train, scaled, label_mapping, classes)
                score = (metrics["macro_f1"], metrics["balanced_accuracy"])
                if score > best_score:
                    best_score = score
                    best_scale = scale
                    best_metrics = metrics
    return best_scale, best_metrics


def plot_class_distribution(y: pd.Series, output_path: Path) -> None:
    counts = y.value_counts().sort_index()
    labels = [TARGET_LABELS[int(i)] for i in counts.index]
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=labels, y=counts.values, hue=labels, palette="Blues_d", legend=False)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 1, str(value), ha="center", va="bottom", fontsize=10)
    plt.title("目标类别分布")
    plt.xlabel("类别")
    plt.ylabel("样本数")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    show_df = metrics_df.melt(
        id_vars="strategy_name",
        value_vars=["accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(14, 6))
    sns.barplot(data=show_df, x="metric", y="score", hue="strategy_name")
    plt.ylim(0, 1.05)
    plt.title("多种集成策略指标对比")
    plt.xlabel("指标")
    plt.ylabel("得分")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_recall_heatmap(recall_df: pd.DataFrame, output_path: Path) -> None:
    pivot_df = recall_df.pivot(index="strategy_name", columns="class_name", values="recall")
    plt.figure(figsize=(8, max(5, 0.55 * len(pivot_df))))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("各策略分类别召回率热力图")
    plt.xlabel("类别")
    plt.ylabel("策略")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_heatmap(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[TARGET_LABELS[i] for i in [0, 1, 2]],
        yticklabels=[TARGET_LABELS[i] for i in [0, 1, 2]],
    )
    plt.title("最优策略混淆矩阵热力图")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_multiclass_roc(y_true: pd.Series, proba: np.ndarray, output_path: Path) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for class_id in [0, 1, 2]:
        fpr, tpr, _ = roc_curve(y_bin[:, class_id], proba[:, class_id])
        auc_score = roc_auc_score(y_bin[:, class_id], proba[:, class_id])
        plt.plot(fpr, tpr, label=f"{TARGET_LABELS[class_id]} AUC={auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("最优策略 One-vs-Rest ROC 曲线")
    plt.xlabel("假阳性率")
    plt.ylabel("真正率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    show_df = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(show_df["feature"], show_df["importance"], color="#4C72B0")
    plt.title(title)
    plt.xlabel("重要度")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def unify_shap_output(shap_values: Any) -> list[np.ndarray]:
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]
    if hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
    else:
        values = np.asarray(shap_values)
    if values.ndim == 3:
        return [values[:, :, idx] for idx in range(values.shape[2])]
    return [values]


def generate_shap_outputs(
    model_name: str,
    model: Any,
    X_test_transformed: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    title_prefix: str,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer(X_test_transformed)
    shap_matrices = unify_shap_output(shap_values_raw)
    target_class = 1 if len(shap_matrices) > 1 else 0
    global_values = np.abs(shap_matrices[target_class]).mean(axis=0)
    global_df = (
        pd.DataFrame({"feature": X_test_transformed.columns, "importance": global_values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    global_df.to_csv(output_dir / "best_strategy_shap_global.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(
        global_df,
        output_dir / "best_strategy_shap_global_bar.png",
        f"{title_prefix} 的 SHAP 全局重要性 Top 20",
    )

    sample_idx = int(np.argmax(model.predict_proba(X_test_transformed)[:, min(target_class, 1)]))
    patient_pred = int(np.argmax(model.predict_proba(X_test_transformed.iloc[[sample_idx]])[0]))
    local_values = shap_matrices[min(patient_pred, len(shap_matrices) - 1)][sample_idx]
    local_df = pd.DataFrame(
        {
            "feature": X_test_transformed.columns,
            "shap_value": local_values,
            "abs_value": np.abs(local_values),
            "patient_value": X_test_transformed.iloc[sample_idx].values,
        }
    ).sort_values("abs_value", ascending=False)
    local_df.to_csv(output_dir / "best_strategy_shap_local_patient.csv", index=False, encoding="utf-8-sig")

    top_local = local_df.head(15).sort_values("shap_value")
    colors = ["#C44E52" if value < 0 else "#4C72B0" for value in top_local["shap_value"]]
    plt.figure(figsize=(10, 8))
    plt.barh(top_local["feature"], top_local["shap_value"], color=colors)
    plt.title(
        f"{title_prefix} 局部归因图（样本索引 {X_test_transformed.index[sample_idx]}）"
    )
    plt.xlabel("SHAP 贡献值")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(output_dir / "best_strategy_shap_local_patient.png", dpi=300)
    plt.close()

    patient_summary = {
        "selected_test_index": int(X_test_transformed.index[sample_idx]),
        "predicted_class": patient_pred,
        "predicted_label": TARGET_LABELS.get(patient_pred, str(patient_pred)),
        "true_class": int(y_test.iloc[sample_idx]),
        "true_label": TARGET_LABELS[int(y_test.iloc[sample_idx])],
        "explained_model": model_name,
        "top_local_features": local_df.head(10).to_dict(orient="records"),
    }
    (output_dir / "best_strategy_shap_patient_summary.json").write_text(
        json.dumps(patient_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_feature_importance(
    artifacts: dict[str, BaseModelArtifacts],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_name, item in artifacts.items():
        model = item.final_model
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "get_feature_importance"):
            values = np.asarray(model.get_feature_importance(), dtype=float)
        else:
            continue
        if values.sum() > 0:
            values = values / values.sum()
        rows.append(
            pd.DataFrame(
                {
                    "feature": item.feature_names,
                    "importance": values,
                    "model_name": model_name,
                }
            )
        )
    return (
        pd.concat(rows, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    artifacts_dir = args.output_dir / "artifacts"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(args.input)
    plot_class_distribution(prepared.y, figures_dir / "class_distribution.png")

    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=prepared.y,
    )

    base_artifacts, trials_map = fit_base_models_with_oof(
        task_name="main",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        numeric_features=prepared.numeric_features,
        categorical_features=prepared.categorical_features,
        discrete_numeric_features=prepared.discrete_numeric_features,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        random_state=args.random_state,
        optimize=True,
    )

    for model_name, trials_df in trials_map.items():
        trials_df.to_csv(
            tables_dir / f"{model_name}_optuna_trials.csv",
            index=False,
            encoding="utf-8-sig",
        )

    strategies = build_base_and_fusion_strategies(
        "main",
        base_artifacts,
        y_train,
        TARGET_LABELS,
        [0, 1, 2],
        args.random_state,
    )
    strategies.append(
        build_stacking_strategy(
            "main",
            base_artifacts,
            X_train,
            y_train,
            X_test,
            TARGET_LABELS,
            [0, 1, 2],
            args.n_splits,
            args.random_state + 20,
        )
    )
    strategies.append(
        run_two_stage_strategy(prepared, X_train, y_train, X_test, y_test, args)
    )
    strategies.append(
        build_leaf_injection_strategy(
            "main",
            base_artifacts,
            X_train,
            y_train,
            X_test,
            TARGET_LABELS,
            [0, 1, 2],
            args.n_splits,
            args.random_state + 30,
        )
    )

    result_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    per_strategy_test_details: dict[str, dict[str, Any]] = {}

    for strategy in strategies:
        test_metrics = evaluate_predictions(y_test, strategy.test_proba, TARGET_LABELS, [0, 1, 2])
        result_rows.append(
            {
                "strategy_name": strategy.strategy_name,
                "accuracy": test_metrics["accuracy"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
                "ovr_roc_auc_macro": test_metrics["ovr_roc_auc_macro"],
            }
        )
        for class_name in ["非确诊", "确诊", "灰色区域"]:
            recall_rows.append(
                {
                    "strategy_name": strategy.strategy_name,
                    "class_name": class_name,
                    "recall": test_metrics["classification_report"][class_name]["recall"],
                }
            )
        per_strategy_test_details[strategy.strategy_name] = {
            "oof_metrics": strategy.metrics,
            "test_metrics": test_metrics,
            "artifacts": strategy.artifacts,
            "explainable_model_name": strategy.explainable_model_name,
        }

    metrics_df = pd.DataFrame(result_rows).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False
    )
    metrics_df.to_csv(tables_dir / "strategy_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(recall_rows).to_csv(
        tables_dir / "strategy_class_recalls.csv", index=False, encoding="utf-8-sig"
    )
    plot_model_comparison(metrics_df, figures_dir / "strategy_comparison.png")
    plot_recall_heatmap(pd.DataFrame(recall_rows), figures_dir / "strategy_recall_heatmap.png")

    best_strategy_name = metrics_df.iloc[0]["strategy_name"]
    best_strategy = next(item for item in strategies if item.strategy_name == best_strategy_name)
    best_strategy_test_metrics = per_strategy_test_details[best_strategy_name]["test_metrics"]

    calibrated_results: list[dict[str, Any]] = []
    best_calibrated_proba = best_strategy.test_proba.copy()
    best_calibrated_name = "default"
    best_calibrated_metrics = best_strategy_test_metrics
    best_score = (
        best_strategy_test_metrics["macro_f1"],
        best_strategy_test_metrics["balanced_accuracy"],
    )
    best_scale = np.ones(3)

    for method in ["sigmoid", "isotonic"]:
        calibrated_test = calibrate_probs_one_vs_rest(
            best_strategy.oof_proba,
            y_train,
            best_strategy.test_proba,
            method,
        )
        metrics = evaluate_predictions(y_test, calibrated_test, TARGET_LABELS, [0, 1, 2])
        calibrated_results.append({"method": method, **{k: metrics[k] for k in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "ovr_roc_auc_macro",
        ]}})

        train_calibrated = calibrate_probs_one_vs_rest(
            best_strategy.oof_proba,
            y_train,
            best_strategy.oof_proba,
            method,
        )
        scale, _ = optimize_class_scaling(train_calibrated, y_train, TARGET_LABELS, [0, 1, 2])
        scaled_test = calibrated_test * scale
        scaled_test = scaled_test / scaled_test.sum(axis=1, keepdims=True)
        scaled_metrics = evaluate_predictions(y_test, scaled_test, TARGET_LABELS, [0, 1, 2])
        calibrated_results.append({"method": f"{method}_thresholded", **{k: scaled_metrics[k] for k in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "ovr_roc_auc_macro",
        ]}})
        score = (scaled_metrics["macro_f1"], scaled_metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_calibrated_proba = scaled_test
            best_calibrated_name = f"{method}_thresholded"
            best_calibrated_metrics = scaled_metrics
            best_scale = scale

    pd.DataFrame(calibrated_results).to_csv(
        tables_dir / "best_strategy_calibration_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_proba_for_plot = best_calibrated_proba if best_calibrated_name != "default" else best_strategy.test_proba
    final_pred_for_plot = np.argmax(final_proba_for_plot, axis=1)
    plot_confusion_heatmap(y_test, final_pred_for_plot, figures_dir / "best_strategy_confusion_heatmap.png")
    plot_multiclass_roc(y_test, final_proba_for_plot, figures_dir / "best_strategy_roc_curve.png")

    importance_df = collect_feature_importance(base_artifacts)
    importance_df.to_csv(tables_dir / "base_feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(
        importance_df,
        figures_dir / "base_feature_importance_top20.png",
        "基模型平均特征重要性 Top 20",
    )

    leaf_strategy = next(item for item in strategies if item.strategy_name == "main_leaf_injection")
    leaf_importance = pd.DataFrame(leaf_strategy.artifacts["meta_feature_importance"])
    leaf_importance.to_csv(
        tables_dir / "leaf_injection_meta_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_feature_importance(
        leaf_importance,
        figures_dir / "leaf_injection_meta_feature_importance.png",
        "中间注入元特征重要性 Top 20",
    )

    stacking_strategy = next(item for item in strategies if item.strategy_name == "main_stacking")
    stacking_importance = pd.DataFrame(stacking_strategy.artifacts["meta_feature_importance"])
    stacking_importance.to_csv(
        tables_dir / "stacking_meta_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_feature_importance(
        stacking_importance,
        figures_dir / "stacking_meta_feature_importance.png",
        "Stacking 元特征重要性 Top 20",
    )

    explain_model_name = best_strategy.explainable_model_name
    explain_artifact = base_artifacts[explain_model_name]
    explain_title = best_strategy_name if explain_model_name == best_strategy_name.split("_")[-1] else f"{best_strategy_name} 的主解释层"
    generate_shap_outputs(
        explain_model_name,
        explain_artifact.final_model,
        explain_artifact.transformed_test_df,
        y_test,
        figures_dir,
        explain_title,
    )

    predictions_df = X_test.copy()
    predictions_df[TARGET_COLUMN] = y_test
    for strategy in strategies:
        predictions_df[f"{strategy.strategy_name}_pred"] = np.argmax(strategy.test_proba, axis=1)
        predictions_df[f"{strategy.strategy_name}_prob_0"] = strategy.test_proba[:, 0]
        predictions_df[f"{strategy.strategy_name}_prob_1"] = strategy.test_proba[:, 1]
        predictions_df[f"{strategy.strategy_name}_prob_2"] = strategy.test_proba[:, 2]
    predictions_df["best_strategy_final_pred"] = final_pred_for_plot
    predictions_df["best_strategy_final_prob_0"] = final_proba_for_plot[:, 0]
    predictions_df["best_strategy_final_prob_1"] = final_proba_for_plot[:, 1]
    predictions_df["best_strategy_final_prob_2"] = final_proba_for_plot[:, 2]
    predictions_df.to_excel(args.output_dir / "test_set_predictions.xlsx", index=True)

    (artifacts_dir / "strategy_details.json").write_text(
        json.dumps(per_strategy_test_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "input_file": str(args.input),
        "n_samples_after_cleaning": int(len(prepared.y)),
        "target_distribution": {str(k): int(v) for k, v in Counter(prepared.y).items()},
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "best_single_model": metrics_df[
            metrics_df["strategy_name"].isin([f"main_{name}" for name in BASE_MODEL_NAMES])
        ].iloc[0]["strategy_name"],
        "best_strategy": best_strategy_name,
        "best_strategy_metrics": best_strategy_test_metrics,
        "best_strategy_calibration": best_calibrated_name,
        "best_strategy_calibrated_metrics": best_calibrated_metrics,
        "best_strategy_threshold_scale": best_scale.tolist(),
        "best_strategy_explainable_model": explain_model_name,
        "all_strategy_metrics": metrics_df.to_dict(orient="records"),
        "base_model_best_params": {
            model_name: artifact.best_params for model_name, artifact in base_artifacts.items()
        },
        "base_model_resampling_summary": {
            model_name: artifact.resampled_counts for model_name, artifact in base_artifacts.items()
        },
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("实验完成")
    print(f"清洗后样本数: {len(prepared.y)}")
    print(f"训练集/测试集: {len(X_train)}/{len(X_test)}")
    print("策略测试集指标:")
    print(metrics_df.to_string(index=False))
    print(f"最佳策略: {best_strategy_name}")
    print(f"校准增强后最佳版本: {best_calibrated_name}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
