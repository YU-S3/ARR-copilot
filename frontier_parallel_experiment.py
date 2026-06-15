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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from tabpfn_client import TabPFNClassifier, set_access_token
from xgboost import XGBClassifier

from causal_xgboost_variants_experiment import XGB_BEST_PARAMS
from env_utils import get_tabpfn_token
from frontier_augmentation import SCMMixAugmentor, TapInspiredInpaintingAugmentor
from multiclass_ensemble_experiment import (
    TARGET_LABELS,
    apply_controlled_adasyn,
    build_preprocessor,
    evaluate_predictions,
    plot_confusion_heatmap,
    plot_multiclass_roc,
    prepare_data,
    transform_with_preprocessor,
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


def prepare_split_data(project_dir: Path) -> SplitData:
    prepared = prepare_data(project_dir / "数据表格测试.xlsx")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=prepared.y,
    )
    preprocessor = build_preprocessor(prepared.numeric_features, prepared.categorical_features)
    X_train_df, X_test_df = transform_with_preprocessor(preprocessor, X_train_raw, X_test_raw)
    scaler = StandardScaler()
    X_train_dense = scaler.fit_transform(X_train_df)
    X_test_dense = scaler.transform(X_test_df)
    return SplitData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        X_train_dense=X_train_dense,
        X_test_dense=X_test_dense,
        discrete_numeric_features=prepared.discrete_numeric_features,
    )


def get_token() -> str | None:
    return get_tabpfn_token(Path(__file__).resolve().parent)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_augmented_xgb(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    discrete_numeric_features: list[str],
) -> np.ndarray:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, X_test_df = transform_with_preprocessor(preprocessor, X_train_raw, X_test_raw)
    X_train_df, y_train = apply_controlled_adasyn(
        X_train_df,
        y_train,
        discrete_numeric_features,
        RANDOM_STATE,
    )
    model = make_xgb_classifier(RANDOM_STATE)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_df, y_train, sample_weight=sample_weight, verbose=False)
    return np.asarray(model.predict_proba(X_test_df))


def run_tabpfn(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
) -> np.ndarray:
    token = get_token()
    if not token:
        raise RuntimeError("未找到 TabPFN access token。")

    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, X_test_df = transform_with_preprocessor(preprocessor, X_train_raw, X_test_raw)
    scaler = StandardScaler()
    X_train_dense = scaler.fit_transform(X_train_df)
    X_test_dense = scaler.transform(X_test_df)

    set_access_token(token)
    model = TabPFNClassifier(random_state=RANDOM_STATE, balance_probabilities=True)
    model.fit(X_train_dense, y_train.to_numpy())
    return np.asarray(model.predict_proba(X_test_dense))


def extract_class0_recall(metrics: dict[str, Any]) -> float:
    report = metrics.get("classification_report", {})
    return float(report.get("非确诊", {}).get("recall", 0.0))


def plot_model_roc_overview(
    y_true: pd.Series,
    probas: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(9, 6))
    for model_name, proba in probas.items():
        auc_score = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin[:, 1], proba[:, 1])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("前沿并行实验 ROC 对比（确诊类 OVR）")
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
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model_name")
    plt.ylim(0, 1.05)
    plt.title("前沿并行实验主要指标对比")
    plt.xlabel("指标")
    plt.ylabel("得分")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_augmentation_shift(audit_df: pd.DataFrame, output_path: Path) -> None:
    stat_df = audit_df[audit_df["row_type"] == "feature_stats"].copy()
    if stat_df.empty:
        return
    stat_df["abs_mean_shift"] = stat_df["mean_shift"].abs().fillna(0.0)
    stat_df["method_feature"] = stat_df["method"].astype(str) + " | " + stat_df["feature"].astype(str)
    show_df = stat_df.sort_values("abs_mean_shift", ascending=False).head(16).iloc[::-1]
    plt.figure(figsize=(11, 8))
    plt.barh(show_df["method_feature"], show_df["abs_mean_shift"], color="#4C72B0")
    plt.title("结构化增广的特征分布偏移 Top 16")
    plt.xlabel("绝对均值偏移")
    plt.ylabel("方法 | 特征")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_reference_predictions(project_dir: Path, split_data: SplitData) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    references: dict[str, dict[str, Any]] = {}
    probas: dict[str, np.ndarray] = {}

    ensemble_summary = load_json(project_dir / "ensemble_outputs_v2" / "experiment_summary.json")
    ensemble_pred = pd.read_excel(project_dir / "ensemble_outputs_v2" / "test_set_predictions.xlsx", index_col=0)
    ensemble_pred = ensemble_pred.loc[split_data.y_test.index]
    references["xgb_reference"] = ensemble_summary["best_strategy_metrics"]
    probas["xgb_reference"] = ensemble_pred[
        ["main_xgboost_prob_0", "main_xgboost_prob_1", "main_xgboost_prob_2"]
    ].to_numpy()

    tabpfn_summary = load_json(project_dir / "tabpfn_only_outputs" / "experiment_summary.json")
    tabpfn_pred = pd.read_excel(project_dir / "tabpfn_only_outputs" / "test_predictions.xlsx", index_col=0)
    tabpfn_pred = tabpfn_pred.loc[split_data.y_test.index]
    references["tabpfn_reference"] = tabpfn_summary["tabpfn_metrics"]
    probas["tabpfn_reference"] = tabpfn_pred[
        ["tabpfn_prob_0", "tabpfn_prob_1", "tabpfn_prob_2"]
    ].to_numpy()

    tabpfn_adasyn_summary = load_json(project_dir / "tabpfn_cost_sensitive_outputs" / "experiment_summary.json")
    tabpfn_adasyn_pred = pd.read_excel(project_dir / "tabpfn_cost_sensitive_outputs" / "test_predictions.xlsx", index_col=0)
    tabpfn_adasyn_pred = tabpfn_adasyn_pred.loc[split_data.y_test.index]
    references["tabpfn_adasyn_reference"] = tabpfn_adasyn_summary["tabpfn_adasyn_base_metrics"]
    probas["tabpfn_adasyn_reference"] = tabpfn_adasyn_pred[
        ["adasyn_base_prob_0", "adasyn_base_prob_1", "adasyn_base_prob_2"]
    ].to_numpy()
    return references, probas


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "frontier_parallel_outputs"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    split_data = prepare_split_data(project_dir)
    references, proba_map = load_reference_predictions(project_dir, split_data)

    metrics_map: dict[str, dict[str, Any]] = dict(references)
    audit_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    run_notes: dict[str, Any] = {"token_available": bool(get_token())}

    tap_result = TapInspiredInpaintingAugmentor(RANDOM_STATE).generate(
        split_data.X_train_raw,
        split_data.y_train,
    )
    tap_train_X = pd.concat([split_data.X_train_raw, tap_result.X_aug], axis=0, ignore_index=True)
    tap_train_y = pd.concat([split_data.y_train.reset_index(drop=True), tap_result.y_aug], axis=0, ignore_index=True)
    tap_proba_xgb = fit_augmented_xgb(
        tap_train_X,
        tap_train_y,
        split_data.X_test_raw,
        split_data.discrete_numeric_features,
    )
    metrics_map["xgb_tap_proxy"] = evaluate_predictions(
        split_data.y_test,
        tap_proba_xgb,
        TARGET_LABELS,
        [0, 1, 2],
    )
    proba_map["xgb_tap_proxy"] = tap_proba_xgb
    audit_frames.append(tap_result.audit.assign(method="tap_proxy"))
    metadata_frames.append(tap_result.metadata.assign(method="tap_proxy"))

    scm_result = SCMMixAugmentor(RANDOM_STATE).generate(
        split_data.X_train_raw,
        split_data.y_train,
    )
    scm_train_X = pd.concat([split_data.X_train_raw, scm_result.X_aug], axis=0, ignore_index=True)
    scm_train_y = pd.concat([split_data.y_train.reset_index(drop=True), scm_result.y_aug], axis=0, ignore_index=True)
    scm_proba_xgb = fit_augmented_xgb(
        scm_train_X,
        scm_train_y,
        split_data.X_test_raw,
        split_data.discrete_numeric_features,
    )
    metrics_map["xgb_scm_proxy"] = evaluate_predictions(
        split_data.y_test,
        scm_proba_xgb,
        TARGET_LABELS,
        [0, 1, 2],
    )
    proba_map["xgb_scm_proxy"] = scm_proba_xgb
    audit_frames.append(scm_result.audit.assign(method="scm_proxy"))
    metadata_frames.append(scm_result.metadata.assign(method="scm_proxy"))

    if get_token():
        try:
            tabpfn_tap_proba = run_tabpfn(tap_train_X, tap_train_y, split_data.X_test_raw)
            metrics_map["tabpfn_tap_proxy"] = evaluate_predictions(
                split_data.y_test,
                tabpfn_tap_proba,
                TARGET_LABELS,
                [0, 1, 2],
            )
            proba_map["tabpfn_tap_proxy"] = tabpfn_tap_proba
        except Exception as exc:
            run_notes["tabpfn_tap_proxy_error"] = str(exc)
        try:
            tabpfn_scm_proba = run_tabpfn(scm_train_X, scm_train_y, split_data.X_test_raw)
            metrics_map["tabpfn_scm_proxy"] = evaluate_predictions(
                split_data.y_test,
                tabpfn_scm_proba,
                TARGET_LABELS,
                [0, 1, 2],
            )
            proba_map["tabpfn_scm_proxy"] = tabpfn_scm_proba
        except Exception as exc:
            run_notes["tabpfn_scm_proxy_error"] = str(exc)
    else:
        run_notes["tabpfn_skip_reason"] = "未提供 TabPFN token，仅保留引用基线并完成 XGBoost 新方案。"

    comparison_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    for model_name, metrics in metrics_map.items():
        comparison_rows.append(
            {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "ovr_roc_auc_macro": metrics["ovr_roc_auc_macro"],
                "class0_recall": extract_class0_recall(metrics),
            }
        )
        for class_name in ["非确诊", "确诊", "灰色区域"]:
            row = metrics["classification_report"].get(class_name, {})
            recall_rows.append(
                {
                    "model_name": model_name,
                    "class_name": class_name,
                    "precision": row.get("precision", np.nan),
                    "recall": row.get("recall", np.nan),
                    "f1_score": row.get("f1-score", np.nan),
                    "support": row.get("support", np.nan),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    )
    recall_df = pd.DataFrame(recall_rows)
    comparison_df.to_csv(tables_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    recall_df.to_csv(tables_dir / "class_recalls.csv", index=False, encoding="utf-8-sig")

    audit_df = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    if not audit_df.empty:
        audit_df.to_csv(tables_dir / "augmentation_audit.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"method": "none", "row_type": "summary", "feature": "no_audit"}]).to_csv(
            tables_dir / "augmentation_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if metadata_frames:
        pd.concat(metadata_frames, ignore_index=True).to_csv(
            tables_dir / "augmentation_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )

    plot_comparison(comparison_df, figures_dir / "comparison_metrics.png")
    plot_model_roc_overview(split_data.y_test, proba_map, figures_dir / "comparison_roc_overview.png")
    if not audit_df.empty:
        plot_augmentation_shift(audit_df, figures_dir / "augmentation_distribution_shift.png")

    best_overall_name = comparison_df.iloc[0]["model_name"]
    best_minority_name = comparison_df.sort_values(
        ["class0_recall", "balanced_accuracy", "macro_f1"],
        ascending=False,
    ).iloc[0]["model_name"]

    plot_confusion_heatmap(
        split_data.y_test,
        np.argmax(proba_map[best_overall_name], axis=1),
        figures_dir / "best_overall_confusion_heatmap.png",
    )
    plot_confusion_heatmap(
        split_data.y_test,
        np.argmax(proba_map[best_minority_name], axis=1),
        figures_dir / "best_minority_confusion_heatmap.png",
    )
    plot_multiclass_roc(
        split_data.y_test,
        proba_map[best_overall_name],
        figures_dir / "best_overall_multiclass_roc.png",
    )

    summary = {
        "train_size": int(len(split_data.y_train)),
        "test_size": int(len(split_data.y_test)),
        "reference_models": ["xgb_reference", "tabpfn_reference", "tabpfn_adasyn_reference"],
        "new_models": [name for name in metrics_map if name.endswith("_proxy")],
        "best_overall": best_overall_name,
        "best_minority": best_minority_name,
        "comparison_metrics": comparison_rows,
        "run_notes": run_notes,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("前沿并行实验完成")
    print(comparison_df.to_string(index=False))
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
