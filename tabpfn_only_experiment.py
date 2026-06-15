from __future__ import annotations

import json
import os
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tabpfn_client import TabPFNClassifier, set_access_token

from env_utils import get_tabpfn_token
from multiclass_ensemble_experiment import (
    TARGET_LABELS,
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
    X_train_dense: np.ndarray
    X_test_dense: np.ndarray
    feature_names: list[str]


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
        X_train_dense=x_train_dense,
        X_test_dense=x_test_dense,
        feature_names=list(x_train_df.columns),
    )


def run_tabpfn(split_data: SplitData, random_state: int = RANDOM_STATE) -> np.ndarray:
    token = get_tabpfn_token(Path(__file__).resolve().parent)
    if not token:
        raise RuntimeError("未找到 TabPFN access token。请设置 TABPFN_API_TOKEN 或 TABPFN_TOKEN。")

    set_access_token(token)
    model = TabPFNClassifier(random_state=random_state)
    model.fit(split_data.X_train_dense, split_data.y_train.to_numpy())
    return np.asarray(model.predict_proba(split_data.X_test_dense))


def plot_model_roc_overview(y_true: pd.Series, probas: dict[str, np.ndarray], output_path: Path) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for model_name, proba in probas.items():
        auc_score = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin[:, 1], proba[:, 1])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("TabPFN-3 与 XGBoost ROC 对比（确诊类 OVR）")
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
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model_name")
    plt.ylim(0, 1.05)
    plt.title("TabPFN-3 与 XGBoost 指标对比")
    plt.xlabel("指标")
    plt.ylabel("得分")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "tabpfn_only_outputs"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    split_data = prepare_split_data(RANDOM_STATE)
    baseline_summary = json.loads((project_dir / "ensemble_outputs_v2" / "experiment_summary.json").read_text(encoding="utf-8"))
    baseline_predictions = pd.read_excel(project_dir / "ensemble_outputs_v2" / "test_set_predictions.xlsx", index_col=0)
    baseline_metrics = baseline_summary["best_strategy_metrics"]
    baseline_proba = baseline_predictions[
        ["main_xgboost_prob_0", "main_xgboost_prob_1", "main_xgboost_prob_2"]
    ].to_numpy()

    tabpfn_proba = run_tabpfn(split_data, RANDOM_STATE)
    tabpfn_metrics = evaluate_predictions(split_data.y_test, tabpfn_proba, TARGET_LABELS, [0, 1, 2])

    comparison_df = pd.DataFrame(
        [
            {
                "model_name": "tabpfn_3",
                "accuracy": tabpfn_metrics["accuracy"],
                "balanced_accuracy": tabpfn_metrics["balanced_accuracy"],
                "macro_f1": tabpfn_metrics["macro_f1"],
                "weighted_f1": tabpfn_metrics["weighted_f1"],
                "ovr_roc_auc_macro": tabpfn_metrics["ovr_roc_auc_macro"],
            },
            {
                "model_name": "xgboost_baseline",
                "accuracy": baseline_metrics["accuracy"],
                "balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "macro_f1": baseline_metrics["macro_f1"],
                "weighted_f1": baseline_metrics["weighted_f1"],
                "ovr_roc_auc_macro": baseline_metrics["ovr_roc_auc_macro"],
            },
        ]
    ).sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    comparison_df.to_csv(tables_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")

    plot_comparison(comparison_df, figures_dir / "comparison_metrics.png")
    plot_model_roc_overview(
        split_data.y_test,
        {"TabPFN-3": tabpfn_proba, "XGBoost": baseline_proba},
        figures_dir / "comparison_roc_overview.png",
    )
    plot_confusion_heatmap(
        split_data.y_test,
        np.argmax(tabpfn_proba, axis=1),
        figures_dir / "tabpfn_confusion_heatmap.png",
    )
    plot_multiclass_roc(split_data.y_test, tabpfn_proba, figures_dir / "tabpfn_roc_curve.png")

    summary = {
        "tabpfn_metrics": tabpfn_metrics,
        "baseline_xgboost_metrics": baseline_metrics,
        "test_size": int(len(split_data.y_test)),
        "train_size": int(len(split_data.y_train)),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions_df = split_data.X_test_raw.copy()
    predictions_df["true_label"] = split_data.y_test
    predictions_df["tabpfn_pred"] = np.argmax(tabpfn_proba, axis=1)
    predictions_df["tabpfn_prob_0"] = tabpfn_proba[:, 0]
    predictions_df["tabpfn_prob_1"] = tabpfn_proba[:, 1]
    predictions_df["tabpfn_prob_2"] = tabpfn_proba[:, 2]
    predictions_df.to_excel(output_dir / "test_predictions.xlsx", index=True)

    print("TabPFN 单独实验完成")
    print(comparison_df.to_string(index=False))
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
