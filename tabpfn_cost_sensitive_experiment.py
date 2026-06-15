from __future__ import annotations

import json
import os
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
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tabpfn_client import TabPFNClassifier, set_access_token

from env_utils import get_tabpfn_token
from multiclass_ensemble_experiment import (
    TARGET_LABELS,
    apply_controlled_adasyn,
    build_preprocessor,
    evaluate_predictions,
    plot_confusion_heatmap,
    plot_multiclass_roc,
    prepare_data,
)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42


@dataclass
class SplitData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_df: pd.DataFrame
    X_test_df: pd.DataFrame
    X_train_dense: np.ndarray
    X_test_dense: np.ndarray
    discrete_numeric_features: list[str]


def transform_to_dataframe(
    preprocessor: Any, x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(x_train_t, columns=feature_names, index=x_train.index),
        pd.DataFrame(x_test_t, columns=feature_names, index=x_test.index),
    )


def prepare_split_data(random_state: int = RANDOM_STATE) -> SplitData:
    project_dir = Path(__file__).resolve().parent
    prepared = prepare_data(project_dir / "数据表格测试.xlsx")
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=random_state,
        stratify=prepared.y,
    )
    preprocessor = build_preprocessor(prepared.numeric_features, prepared.categorical_features)
    x_train_df, x_test_df = transform_to_dataframe(preprocessor, x_train_raw, x_test_raw)
    scaler = StandardScaler()
    x_train_dense = scaler.fit_transform(x_train_df)
    x_test_dense = scaler.transform(x_test_df)
    return SplitData(
        X_train_raw=x_train_raw,
        X_test_raw=x_test_raw,
        y_train=y_train,
        y_test=y_test,
        X_train_df=x_train_df,
        X_test_df=x_test_df,
        X_train_dense=x_train_dense,
        X_test_dense=x_test_dense,
        discrete_numeric_features=prepared.discrete_numeric_features,
    )


def get_token() -> str:
    token = get_tabpfn_token(Path(__file__).resolve().parent)
    if not token:
        raise RuntimeError("未找到 TabPFN access token。请设置 TABPFN_API_TOKEN 或 TABPFN_TOKEN。")
    return token


def build_think_mode_sampling_strategy(y_train: pd.Series, target_total: int = 520) -> dict[int, int]:
    counts = Counter(y_train)
    majority_class, majority_count = counts.most_common(1)[0]
    n_classes = len(counts)
    target_each = max(int(np.ceil(target_total / n_classes)), majority_count)
    strategy: dict[int, int] = {}
    for cls, count in counts.items():
        if cls == majority_class:
            continue
        cap = int(majority_count * (0.85 if int(cls) == 0 else 0.95))
        desired = min(max(target_each, count + 2), cap)
        if desired > count:
            strategy[int(cls)] = desired
    return strategy


def apply_tabpfn_adasyn(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    discrete_numeric_features: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    strategy = build_think_mode_sampling_strategy(y_train)
    if not strategy:
        return X_train_df, y_train

    minority_count = min(Counter(y_train).values())
    if minority_count <= 2:
        return X_train_df, y_train

    n_neighbors = min(5, minority_count - 1)
    sampler = ADASYN(
        sampling_strategy=strategy,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train_df, y_train)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train_df.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    for col in X_train_df.columns:
        if col.startswith("num__") and not col.startswith("num__missingindicator"):
            lower = X_train_df[col].min()
            upper = X_train_df[col].max()
            X_resampled[col] = X_resampled[col].clip(lower=lower, upper=upper)

    for raw_col in discrete_numeric_features:
        transformed_col = f"num__{raw_col}"
        if transformed_col in X_resampled.columns:
            lower = X_train_df[transformed_col].min()
            upper = X_train_df[transformed_col].max()
            X_resampled[transformed_col] = (
                X_resampled[transformed_col].round().clip(lower=lower, upper=upper)
            )

    return X_resampled, y_resampled


def run_tabpfn_model(
    X_train_dense: np.ndarray,
    y_train: pd.Series,
    X_test_dense: np.ndarray,
    thinking_mode: bool = False,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    set_access_token(get_token())
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            model = TabPFNClassifier(
                random_state=random_state,
                balance_probabilities=True,
                thinking_mode=thinking_mode,
                thinking_timeout_s=240 if thinking_mode else None,
            )
            model.fit(X_train_dense, y_train.to_numpy())
            return np.asarray(model.predict_proba(X_test_dense))
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2 * attempt)
    if last_error is None:
        raise RuntimeError("TabPFN 运行失败，但未捕获到具体异常。")
    raise last_error


def plot_model_roc_overview(y_true: pd.Series, probas: dict[str, np.ndarray], output_path: Path) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for model_name, proba in probas.items():
        auc_score = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin[:, 1], proba[:, 1])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("TabPFN 不同训练模式 ROC 对比（确诊类 OVR）")
    plt.xlabel("假阳性率")
    plt.ylabel("真正率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    melted = metrics_df.melt(
        id_vars="model_name",
        value_vars=["accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model_name")
    plt.ylim(0, 1.05)
    plt.title("TabPFN 基础版、ADASYN版与 think-mode 对比")
    plt.xlabel("指标")
    plt.ylabel("得分")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "tabpfn_cost_sensitive_outputs"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    split_data = prepare_split_data(RANDOM_STATE)

    baseline_summary = json.loads(
        (project_dir / "ensemble_outputs_v2" / "experiment_summary.json").read_text(encoding="utf-8")
    )
    xgb_baseline_metrics = baseline_summary["best_strategy_metrics"]

    results: list[dict[str, Any]] = []
    proba_map: dict[str, np.ndarray] = {}
    artifacts: dict[str, Any] = {}

    # 1. Original base model
    base_proba = run_tabpfn_model(
        split_data.X_train_dense,
        split_data.y_train,
        split_data.X_test_dense,
        thinking_mode=False,
        random_state=RANDOM_STATE,
    )
    base_metrics = evaluate_predictions(split_data.y_test, base_proba, TARGET_LABELS, [0, 1, 2])
    results.append(
        {
            "model_name": "tabpfn_base_original",
            "train_size": int(len(split_data.y_train)),
            "accuracy": base_metrics["accuracy"],
            "balanced_accuracy": base_metrics["balanced_accuracy"],
            "macro_f1": base_metrics["macro_f1"],
            "weighted_f1": base_metrics["weighted_f1"],
            "ovr_roc_auc_macro": base_metrics["ovr_roc_auc_macro"],
        }
    )
    proba_map["base_original"] = base_proba

    # 2. Controlled ADASYN enhanced base
    X_resampled_df, y_resampled = apply_tabpfn_adasyn(
        split_data.X_train_df,
        split_data.y_train,
        split_data.discrete_numeric_features,
        RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_resampled_dense = scaler.fit_transform(X_resampled_df)
    X_test_dense_rescaled = scaler.transform(split_data.X_test_df)
    resampled_counts = {str(k): int(v) for k, v in Counter(y_resampled).items()}

    adasyn_base_proba = run_tabpfn_model(
        X_resampled_dense,
        y_resampled,
        X_test_dense_rescaled,
        thinking_mode=False,
        random_state=RANDOM_STATE + 1,
    )
    adasyn_base_metrics = evaluate_predictions(split_data.y_test, adasyn_base_proba, TARGET_LABELS, [0, 1, 2])
    results.append(
        {
            "model_name": "tabpfn_adasyn_base",
            "train_size": int(len(y_resampled)),
            "accuracy": adasyn_base_metrics["accuracy"],
            "balanced_accuracy": adasyn_base_metrics["balanced_accuracy"],
            "macro_f1": adasyn_base_metrics["macro_f1"],
            "weighted_f1": adasyn_base_metrics["weighted_f1"],
            "ovr_roc_auc_macro": adasyn_base_metrics["ovr_roc_auc_macro"],
        }
    )
    proba_map["adasyn_base"] = adasyn_base_proba

    # 3. ADASYN enhanced think-mode
    think_status = "success"
    think_metrics: dict[str, Any] | None = None
    think_proba: np.ndarray | None = None
    think_error: str | None = None
    try:
        think_proba = run_tabpfn_model(
            X_resampled_dense,
            y_resampled,
            X_test_dense_rescaled,
            thinking_mode=True,
            random_state=RANDOM_STATE + 2,
        )
        think_metrics = evaluate_predictions(split_data.y_test, think_proba, TARGET_LABELS, [0, 1, 2])
        results.append(
            {
                "model_name": "tabpfn_adasyn_think_mode",
                "train_size": int(len(y_resampled)),
                "accuracy": think_metrics["accuracy"],
                "balanced_accuracy": think_metrics["balanced_accuracy"],
                "macro_f1": think_metrics["macro_f1"],
                "weighted_f1": think_metrics["weighted_f1"],
                "ovr_roc_auc_macro": think_metrics["ovr_roc_auc_macro"],
            }
        )
        proba_map["adasyn_think_mode"] = think_proba
    except Exception as exc:  # pragma: no cover
        think_status = "failed"
        think_error = str(exc)

    results.append(
        {
            "model_name": "xgboost_baseline_reference",
            "train_size": int(len(split_data.y_train)),
            "accuracy": xgb_baseline_metrics["accuracy"],
            "balanced_accuracy": xgb_baseline_metrics["balanced_accuracy"],
            "macro_f1": xgb_baseline_metrics["macro_f1"],
            "weighted_f1": xgb_baseline_metrics["weighted_f1"],
            "ovr_roc_auc_macro": xgb_baseline_metrics["ovr_roc_auc_macro"],
        }
    )

    metrics_df = pd.DataFrame(results).sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    metrics_df.to_csv(tables_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    plot_comparison(metrics_df, figures_dir / "comparison_metrics.png")

    if proba_map:
        plot_model_roc_overview(
            split_data.y_test,
            {
                "Base": base_proba,
                "ADASYN Base": adasyn_base_proba,
                **({"ADASYN Think": think_proba} if think_proba is not None else {}),
            },
            figures_dir / "comparison_roc_overview.png",
        )

    best_tabpfn_name = metrics_df[metrics_df["model_name"].str.contains("tabpfn")].iloc[0]["model_name"]
    best_proba_lookup = {
        "tabpfn_base_original": base_proba,
        "tabpfn_adasyn_base": adasyn_base_proba,
        "tabpfn_adasyn_think_mode": think_proba,
    }
    best_proba = best_proba_lookup.get(best_tabpfn_name)
    if best_proba is not None:
        plot_confusion_heatmap(
            split_data.y_test,
            np.argmax(best_proba, axis=1),
            figures_dir / "best_tabpfn_confusion_heatmap.png",
        )
        plot_multiclass_roc(
            split_data.y_test,
            best_proba,
            figures_dir / "best_tabpfn_multiclass_roc.png",
        )

    prediction_df = split_data.X_test_raw.copy()
    prediction_df["true_label"] = split_data.y_test
    prediction_df["true_label_name"] = split_data.y_test.map(TARGET_LABELS)
    prediction_df["base_pred"] = np.argmax(base_proba, axis=1)
    prediction_df["base_prob_0"] = base_proba[:, 0]
    prediction_df["base_prob_1"] = base_proba[:, 1]
    prediction_df["base_prob_2"] = base_proba[:, 2]
    prediction_df["adasyn_base_pred"] = np.argmax(adasyn_base_proba, axis=1)
    prediction_df["adasyn_base_prob_0"] = adasyn_base_proba[:, 0]
    prediction_df["adasyn_base_prob_1"] = adasyn_base_proba[:, 1]
    prediction_df["adasyn_base_prob_2"] = adasyn_base_proba[:, 2]
    if think_proba is not None:
        prediction_df["adasyn_think_pred"] = np.argmax(think_proba, axis=1)
        prediction_df["adasyn_think_prob_0"] = think_proba[:, 0]
        prediction_df["adasyn_think_prob_1"] = think_proba[:, 1]
        prediction_df["adasyn_think_prob_2"] = think_proba[:, 2]
    prediction_df.to_excel(output_dir / "test_predictions.xlsx", index=True)

    summary = {
        "train_size_original": int(len(split_data.y_train)),
        "train_size_after_adasyn": int(len(y_resampled)),
        "resampled_counts": resampled_counts,
        "tabpfn_base_metrics": base_metrics,
        "tabpfn_adasyn_base_metrics": adasyn_base_metrics,
        "tabpfn_adasyn_think_mode_status": think_status,
        "tabpfn_adasyn_think_mode_metrics": think_metrics,
        "tabpfn_adasyn_think_mode_error": think_error,
        "xgboost_reference_metrics": xgb_baseline_metrics,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("TabPFN 代价敏感/ADASYN/think-mode 对比实验完成")
    print(metrics_df.to_string(index=False))
    print(f"重采样后训练样本数: {len(y_resampled)}")
    print(f"重采样后类别分布: {resampled_counts}")
    print(f"think-mode 状态: {think_status}")
    if think_error:
        print(f"think-mode 错误: {think_error}")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
