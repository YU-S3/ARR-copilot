from __future__ import annotations

import json
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from causal_xgboost_variants_experiment import XGB_BEST_PARAMS
from frontier_augmentation import SCMMixAugmentor, TapInspiredInpaintingAugmentor
from multiclass_ensemble_experiment import (
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

DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]
EXPERIMENT_GROUPS = [
    "xgb_reference_raw",
    "xgb_reference_adasyn",
    "xgb_tap_only",
    "xgb_tap_plus_adasyn",
    "xgb_scm_only",
    "xgb_scm_plus_adasyn",
]


@dataclass
class SplitData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    discrete_numeric_features: list[str]


@dataclass
class TrainResult:
    proba: np.ndarray
    train_size: int
    used_adasyn: bool
    resampled_train_size: int


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


def prepare_split_data(project_dir: Path, seed: int) -> SplitData:
    prepared = prepare_data(project_dir / "数据表格测试.xlsx")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=seed,
        stratify=prepared.y,
    )
    return SplitData(
        X_train_raw=X_train_raw.reset_index(drop=True),
        X_test_raw=X_test_raw.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        discrete_numeric_features=prepared.discrete_numeric_features,
    )


def fit_xgb_from_raw(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    discrete_numeric_features: list[str],
    seed: int,
    use_adasyn: bool,
) -> TrainResult:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, X_test_df = transform_with_preprocessor(preprocessor, X_train_raw, X_test_raw)

    resampled_train_size = int(len(X_train_df))
    if use_adasyn:
        X_train_df, y_train = apply_controlled_adasyn(
            X_train_df,
            y_train,
            discrete_numeric_features,
            seed,
        )
        resampled_train_size = int(len(X_train_df))

    model = make_xgb_classifier(seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_df, y_train, sample_weight=sample_weight, verbose=False)
    proba = np.asarray(model.predict_proba(X_test_df))
    return TrainResult(
        proba=proba,
        train_size=int(len(X_train_raw)),
        used_adasyn=use_adasyn,
        resampled_train_size=resampled_train_size,
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


def plot_metric_boxplot(metrics_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 6))
    order = (
        metrics_df.groupby("model_name")[metric]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    sns.boxplot(data=metrics_df, x="model_name", y=metric, order=order)
    sns.stripplot(
        data=metrics_df,
        x="model_name",
        y=metric,
        order=order,
        color="black",
        alpha=0.55,
        size=4,
    )
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.xlabel("方案")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_recall_errorbar(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.sort_values("class0_recall_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df))
    lower = plot_df["class0_recall_mean"] - plot_df["class0_recall_ci_low"]
    upper = plot_df["class0_recall_ci_high"] - plot_df["class0_recall_mean"]

    plt.figure(figsize=(11, 6))
    plt.errorbar(
        x,
        plot_df["class0_recall_mean"],
        yerr=[lower, upper],
        fmt="o",
        capsize=4,
        color="#4C72B0",
    )
    plt.xticks(x, plot_df["model_name"], rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.title("各方案 0 类 Recall 均值与 95% Bootstrap CI")
    plt.xlabel("方案")
    plt.ylabel("class0_recall")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "frontier_augmentation_outputs_v2"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []
    metadata_rows: list[pd.DataFrame] = []

    for seed in DEFAULT_SEEDS:
        split_data = prepare_split_data(project_dir, seed)
        base_train_result = fit_xgb_from_raw(
            split_data.X_train_raw,
            split_data.y_train,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=False,
        )
        base_adasyn_result = fit_xgb_from_raw(
            split_data.X_train_raw,
            split_data.y_train,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=True,
        )

        tap_aug = TapInspiredInpaintingAugmentor(random_state=seed).generate(
            split_data.X_train_raw,
            split_data.y_train,
        )
        tap_train_X = pd.concat(
            [split_data.X_train_raw, tap_aug.X_aug],
            axis=0,
            ignore_index=True,
        )
        tap_train_y = pd.concat(
            [split_data.y_train, tap_aug.y_aug],
            axis=0,
            ignore_index=True,
        )
        tap_only_result = fit_xgb_from_raw(
            tap_train_X,
            tap_train_y,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=False,
        )
        tap_adasyn_result = fit_xgb_from_raw(
            tap_train_X,
            tap_train_y,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=True,
        )

        scm_aug = SCMMixAugmentor(random_state=seed).generate(
            split_data.X_train_raw,
            split_data.y_train,
        )
        scm_train_X = pd.concat(
            [split_data.X_train_raw, scm_aug.X_aug],
            axis=0,
            ignore_index=True,
        )
        scm_train_y = pd.concat(
            [split_data.y_train, scm_aug.y_aug],
            axis=0,
            ignore_index=True,
        )
        scm_only_result = fit_xgb_from_raw(
            scm_train_X,
            scm_train_y,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=False,
        )
        scm_adasyn_result = fit_xgb_from_raw(
            scm_train_X,
            scm_train_y,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=True,
        )

        seed_result_map = {
            "xgb_reference_raw": base_train_result,
            "xgb_reference_adasyn": base_adasyn_result,
            "xgb_tap_only": tap_only_result,
            "xgb_tap_plus_adasyn": tap_adasyn_result,
            "xgb_scm_only": scm_only_result,
            "xgb_scm_plus_adasyn": scm_adasyn_result,
        }

        for model_name, train_result in seed_result_map.items():
            metrics = evaluate_predictions(
                split_data.y_test,
                train_result.proba,
                TARGET_LABELS,
                [0, 1, 2],
            )
            metric_rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "train_size": train_result.train_size,
                    "resampled_train_size": train_result.resampled_train_size,
                    "used_adasyn": train_result.used_adasyn,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "ovr_roc_auc_macro": metrics["ovr_roc_auc_macro"],
                    "class0_recall": extract_class_recall(metrics, "非确诊"),
                    "class1_recall": extract_class_recall(metrics, "确诊"),
                    "class2_recall": extract_class_recall(metrics, "灰色区域"),
                }
            )
            for class_name in ["非确诊", "确诊", "灰色区域"]:
                row = metrics["classification_report"].get(class_name, {})
                recall_rows.append(
                    {
                        "seed": seed,
                        "model_name": model_name,
                        "class_name": class_name,
                        "precision": row.get("precision", np.nan),
                        "recall": row.get("recall", np.nan),
                        "f1_score": row.get("f1-score", np.nan),
                        "support": row.get("support", np.nan),
                    }
                )

        audit_rows.append(tap_aug.audit.assign(seed=seed, method="tap_proxy"))
        audit_rows.append(scm_aug.audit.assign(seed=seed, method="scm_proxy"))
        metadata_rows.append(tap_aug.metadata.assign(seed=seed, method="tap_proxy"))
        metadata_rows.append(scm_aug.metadata.assign(seed=seed, method="scm_proxy"))

    metrics_df = pd.DataFrame(metric_rows)
    recall_df = pd.DataFrame(recall_rows)
    metrics_df.to_csv(tables_dir / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    recall_df.to_csv(tables_dir / "class_recalls_by_seed.csv", index=False, encoding="utf-8-sig")

    if audit_rows:
        pd.concat(audit_rows, ignore_index=True).to_csv(
            tables_dir / "augmentation_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if metadata_rows:
        pd.concat(metadata_rows, ignore_index=True).to_csv(
            tables_dir / "augmentation_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_rows: list[dict[str, Any]] = []
    for model_name in EXPERIMENT_GROUPS:
        model_df = metrics_df[metrics_df["model_name"] == model_name].copy()
        row: dict[str, Any] = {"model_name": model_name}
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "ovr_roc_auc_macro",
            "class0_recall",
            "class1_recall",
            "class2_recall",
        ]:
            values = model_df[metric].to_numpy(dtype=float)
            ci_low, ci_high = compute_bootstrap_ci(values, seed=12345 + len(summary_rows))
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        row["train_size_mean"] = float(model_df["train_size"].mean())
        row["resampled_train_size_mean"] = float(model_df["resampled_train_size"].mean())
        row["used_adasyn"] = bool(model_df["used_adasyn"].iloc[0])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=False,
    )
    summary_df.to_csv(tables_dir / "metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    ablation_rows: list[dict[str, Any]] = []
    summary_map = summary_df.set_index("model_name")
    ablation_rows.append(
        {
            "comparison": "reference_adasyn_minus_reference_raw",
            "balanced_accuracy_delta": float(
                summary_map.loc["xgb_reference_adasyn", "balanced_accuracy_mean"]
                - summary_map.loc["xgb_reference_raw", "balanced_accuracy_mean"]
            ),
            "macro_f1_delta": float(
                summary_map.loc["xgb_reference_adasyn", "macro_f1_mean"]
                - summary_map.loc["xgb_reference_raw", "macro_f1_mean"]
            ),
            "class0_recall_delta": float(
                summary_map.loc["xgb_reference_adasyn", "class0_recall_mean"]
                - summary_map.loc["xgb_reference_raw", "class0_recall_mean"]
            ),
        }
    )
    ablation_rows.append(
        {
            "comparison": "tap_only_minus_reference_raw",
            "balanced_accuracy_delta": float(
                summary_map.loc["xgb_tap_only", "balanced_accuracy_mean"]
                - summary_map.loc["xgb_reference_raw", "balanced_accuracy_mean"]
            ),
            "macro_f1_delta": float(
                summary_map.loc["xgb_tap_only", "macro_f1_mean"]
                - summary_map.loc["xgb_reference_raw", "macro_f1_mean"]
            ),
            "class0_recall_delta": float(
                summary_map.loc["xgb_tap_only", "class0_recall_mean"]
                - summary_map.loc["xgb_reference_raw", "class0_recall_mean"]
            ),
        }
    )
    ablation_rows.append(
        {
            "comparison": "tap_plus_adasyn_minus_reference_adasyn",
            "balanced_accuracy_delta": float(
                summary_map.loc["xgb_tap_plus_adasyn", "balanced_accuracy_mean"]
                - summary_map.loc["xgb_reference_adasyn", "balanced_accuracy_mean"]
            ),
            "macro_f1_delta": float(
                summary_map.loc["xgb_tap_plus_adasyn", "macro_f1_mean"]
                - summary_map.loc["xgb_reference_adasyn", "macro_f1_mean"]
            ),
            "class0_recall_delta": float(
                summary_map.loc["xgb_tap_plus_adasyn", "class0_recall_mean"]
                - summary_map.loc["xgb_reference_adasyn", "class0_recall_mean"]
            ),
        }
    )
    ablation_rows.append(
        {
            "comparison": "scm_only_minus_reference_raw",
            "balanced_accuracy_delta": float(
                summary_map.loc["xgb_scm_only", "balanced_accuracy_mean"]
                - summary_map.loc["xgb_reference_raw", "balanced_accuracy_mean"]
            ),
            "macro_f1_delta": float(
                summary_map.loc["xgb_scm_only", "macro_f1_mean"]
                - summary_map.loc["xgb_reference_raw", "macro_f1_mean"]
            ),
            "class0_recall_delta": float(
                summary_map.loc["xgb_scm_only", "class0_recall_mean"]
                - summary_map.loc["xgb_reference_raw", "class0_recall_mean"]
            ),
        }
    )
    ablation_rows.append(
        {
            "comparison": "scm_plus_adasyn_minus_reference_adasyn",
            "balanced_accuracy_delta": float(
                summary_map.loc["xgb_scm_plus_adasyn", "balanced_accuracy_mean"]
                - summary_map.loc["xgb_reference_adasyn", "balanced_accuracy_mean"]
            ),
            "macro_f1_delta": float(
                summary_map.loc["xgb_scm_plus_adasyn", "macro_f1_mean"]
                - summary_map.loc["xgb_reference_adasyn", "macro_f1_mean"]
            ),
            "class0_recall_delta": float(
                summary_map.loc["xgb_scm_plus_adasyn", "class0_recall_mean"]
                - summary_map.loc["xgb_reference_adasyn", "class0_recall_mean"]
            ),
        }
    )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(tables_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    plot_metric_boxplot(
        metrics_df,
        "balanced_accuracy",
        figures_dir / "balanced_accuracy_boxplot.png",
        "各方案 Balanced Accuracy 分布",
    )
    plot_metric_boxplot(
        metrics_df,
        "macro_f1",
        figures_dir / "macro_f1_boxplot.png",
        "各方案 Macro F1 分布",
    )
    plot_recall_errorbar(summary_df, figures_dir / "class0_recall_errorbar.png")

    best_overall = summary_df.iloc[0]["model_name"]
    best_minority = summary_df.sort_values(
        ["class0_recall_mean", "balanced_accuracy_mean", "macro_f1_mean"],
        ascending=False,
    ).iloc[0]["model_name"]

    summary = {
        "seeds": DEFAULT_SEEDS,
        "test_size_ratio": 0.2,
        "experiment_groups": EXPERIMENT_GROUPS,
        "best_overall_by_mean_macro_f1": best_overall,
        "best_minority_by_mean_class0_recall": best_minority,
        "metrics_mean_std": summary_rows,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("前沿增广解耦消融实验完成")
    print(summary_df.to_string(index=False))
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
