from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from frontier_scm_v2_experiment import (
    DEFAULT_SEEDS,
    ProgressTracker,
    SCMMixV2Augmentor,
    build_hard_case_index_map,
    make_xgb_classifier,
)
from multiclass_ensemble_experiment import (
    DEFAULT_INPUT_FILE,
    apply_controlled_adasyn,
    build_preprocessor,
    prepare_data,
)
from scm_v3_augmentation import SCMMixV3Augmentor
from tabddpm_prototype_experiment import generate_tabddpm_rows


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "frontier_scm_v3_outputs" / "generalization_diagnostics"
DATA_PATH = PROJECT_DIR / DEFAULT_INPUT_FILE
CLASSES = [0, 1, 2]

SCM_V2_REFERENCE_CONFIG = {
    "seed_strategy": "hard_case_seed",
    "target_classes": "class0_only",
    "treat_mix_prob": 0.4,
    "residual_scale": 0.5,
    "teacher_mode": "single",
}

SCM_V3_BEST_CONFIG = {
    "seed_strategy": "hard_case_seed",
    "target_classes": "class0_only",
    "treat_mix_prob": 0.4,
    "sampler_strength": 0.5,
    "node_sampler": "residual",
    "rule_filter_mode": "off",
    "teacher_filter_mode": "single",
    "counterfactual_mode": "off",
}


@dataclass
class ThreeWaySplit:
    X_train_raw: pd.DataFrame
    X_valid_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    discrete_numeric_features: list[str]


@dataclass
class DiagnosticFit:
    proba_train: np.ndarray
    proba_valid: np.ndarray
    proba_test: np.ndarray
    train_size: int
    augmented_size: int
    resampled_train_size: int
    used_adasyn: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCM_v3 generalization and calibration diagnostics.")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--valid-size", type=float, default=0.25)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tabddpm-epochs", type=int, default=120)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-tabddpm", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild report files from existing tables.")
    return parser.parse_args()


def prepare_three_way_split(
    seed: int,
    valid_size: float,
    test_size: float,
    input_path: Path | None = None,
) -> ThreeWaySplit:
    resolved_input = Path(input_path or DATA_PATH)
    prepared = prepare_data(resolved_input)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=test_size,
        random_state=seed,
        stratify=prepared.y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_size,
        random_state=seed + 1009,
        stratify=y_train_valid,
    )
    return ThreeWaySplit(
        X_train_raw=X_train.reset_index(drop=True),
        X_valid_raw=X_valid.reset_index(drop=True),
        X_test_raw=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        discrete_numeric_features=prepared.discrete_numeric_features,
    )


def safe_apply_adasyn(
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
    except ValueError as exc:
        if "No samples will be generated with the provided ratio settings" not in str(exc):
            raise
        return X_train_df, y_train, False
    return X_resampled, y_resampled, len(X_resampled) != len(X_train_df)


def fit_xgb_diagnostic(
    X_fit_raw: pd.DataFrame,
    y_fit: pd.Series,
    X_train_eval_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    discrete_numeric_features: list[str],
    seed: int,
    augmented_size: int,
) -> DiagnosticFit:
    numeric_features = X_fit_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_fit_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_fit_t = preprocessor.fit_transform(X_fit_raw)
    feature_names = preprocessor.get_feature_names_out()
    X_fit_df = pd.DataFrame(X_fit_t, columns=feature_names, index=X_fit_raw.index)
    X_train_eval_df = pd.DataFrame(
        preprocessor.transform(X_train_eval_raw),
        columns=feature_names,
        index=X_train_eval_raw.index,
    )
    X_valid_df = pd.DataFrame(
        preprocessor.transform(X_valid_raw),
        columns=feature_names,
        index=X_valid_raw.index,
    )
    X_test_df = pd.DataFrame(
        preprocessor.transform(X_test_raw),
        columns=feature_names,
        index=X_test_raw.index,
    )

    X_model_df, y_model, used_adasyn = safe_apply_adasyn(
        X_fit_df,
        y_fit,
        discrete_numeric_features,
        seed,
    )
    model = make_xgb_classifier(seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_model)
    model.fit(X_model_df, y_model, sample_weight=sample_weight, verbose=False)
    return DiagnosticFit(
        proba_train=np.asarray(model.predict_proba(X_train_eval_df)),
        proba_valid=np.asarray(model.predict_proba(X_valid_df)),
        proba_test=np.asarray(model.predict_proba(X_test_df)),
        train_size=int(len(X_fit_raw)),
        augmented_size=augmented_size,
        resampled_train_size=int(len(X_model_df)),
        used_adasyn=used_adasyn,
    )


def make_augmented_training_data(
    model_name: str,
    split_data: ThreeWaySplit,
    seed: int,
    tabddpm_epochs: int,
    smoke_test: bool,
) -> tuple[pd.DataFrame, pd.Series, int]:
    if model_name == "xgb_reference_adasyn":
        return split_data.X_train_raw, split_data.y_train, 0

    if model_name == "xgb_scm_v2_best":
        hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)
        augmentor = SCMMixV2Augmentor(random_state=seed, **SCM_V2_REFERENCE_CONFIG)
        aug_result = augmentor.generate(split_data.X_train_raw, split_data.y_train, hard_case_map)
        X_fit = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
        y_fit = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)
        return X_fit, y_fit, int(len(aug_result.X_aug))

    if model_name == "scm_v3_res_norule_single_flat_nocf":
        hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)
        augmentor = SCMMixV3Augmentor(
            random_state=seed,
            project_dir=PROJECT_DIR,
            use_remote_teacher=False,
            **SCM_V3_BEST_CONFIG,
        )
        aug_result = augmentor.generate(split_data.X_train_raw, split_data.y_train, hard_case_map)
        X_fit = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
        y_fit = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)
        return X_fit, y_fit, int(len(aug_result.X_aug))

    if model_name == "xgb_tabddpm_proto":
        X_aug, y_aug = generate_tabddpm_rows(
            split_data.X_train_raw,
            split_data.y_train,
            seed=seed,
            epochs=tabddpm_epochs,
            smoke_test=smoke_test,
        )
        X_fit = pd.concat([split_data.X_train_raw, X_aug], axis=0, ignore_index=True)
        y_fit = pd.concat([split_data.y_train, y_aug], axis=0, ignore_index=True)
        return X_fit, y_fit, int(len(X_aug))

    raise ValueError(f"Unknown model_name: {model_name}")


def expected_calibration_error(y_true: pd.Series, proba: np.ndarray, n_bins: int = 10) -> float:
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    correct = (pred == y_true.to_numpy()).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return ece


def multiclass_brier(y_true: pd.Series, proba: np.ndarray) -> float:
    y_bin = label_binarize(y_true, classes=CLASSES)
    return float(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))


def compute_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    y_array = y_true.to_numpy()
    row: dict[str, float] = {
        "accuracy": float(accuracy_score(y_array, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, pred)),
        "macro_f1": float(f1_score(y_array, pred, average="macro")),
        "weighted_f1": float(f1_score(y_array, pred, average="weighted")),
        "brier_multiclass": multiclass_brier(y_true, proba),
        "ece_confidence": expected_calibration_error(y_true, proba),
        "class0_average_precision": float(average_precision_score((y_array == 0).astype(int), proba[:, 0])),
    }
    for cls in CLASSES:
        mask = y_array == cls
        row[f"class{cls}_support"] = float(mask.sum())
        row[f"class{cls}_recall"] = float(((pred == cls) & mask).sum() / max(mask.sum(), 1))
    try:
        row["ovr_roc_auc_macro"] = float(roc_auc_score(y_array, proba, multi_class="ovr", average="macro"))
    except ValueError:
        row["ovr_roc_auc_macro"] = float("nan")
    return row


def metric_row(
    seed: int,
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    proba: np.ndarray,
    fit: DiagnosticFit,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "model_name": model_name,
        "split": split_name,
        "train_size": fit.train_size,
        "augmented_size": fit.augmented_size,
        "resampled_train_size": fit.resampled_train_size,
        "used_adasyn": fit.used_adasyn,
        **compute_metrics(y_true, proba),
    }


def bootstrap_test_ci(
    predictions: list[dict[str, Any]],
    n_iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_idx, model_name in enumerate(sorted({p["model_name"] for p in predictions})):
        model_predictions = [p for p in predictions if p["model_name"] == model_name and p["split"] == "test"]
        y = np.concatenate([p["y_true"].to_numpy() for p in model_predictions])
        proba = np.vstack([p["proba"] for p in model_predictions])
        rng = np.random.default_rng(7000 + model_idx)
        metric_samples: dict[str, list[float]] = {
            "accuracy": [],
            "balanced_accuracy": [],
            "macro_f1": [],
            "class0_recall": [],
            "class0_average_precision": [],
            "brier_multiclass": [],
            "ece_confidence": [],
        }
        for _ in range(n_iterations):
            idx = rng.integers(0, len(y), size=len(y))
            sample_metrics = compute_metrics(pd.Series(y[idx]), proba[idx])
            for metric in metric_samples:
                metric_samples[metric].append(sample_metrics[metric])
        for metric, values in metric_samples.items():
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "mean": float(np.nanmean(arr)),
                    "ci_low": float(np.nanpercentile(arr, 2.5)),
                    "ci_high": float(np.nanpercentile(arr, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def summarize_split_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "ovr_roc_auc_macro",
        "class0_recall",
        "class1_recall",
        "class2_recall",
        "brier_multiclass",
        "ece_confidence",
        "class0_average_precision",
    ]
    return (
        metrics_df.groupby(["model_name", "split"], as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )


def build_gap_tables(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = metrics_df.pivot_table(
        index=["seed", "model_name"],
        columns="split",
        values=["macro_f1", "balanced_accuracy", "brier_multiclass", "ece_confidence"],
    )
    wide.columns = [f"{metric}_{split}" for metric, split in wide.columns]
    gap_df = wide.reset_index()
    gap_df["macro_f1_train_test_gap"] = gap_df["macro_f1_train"] - gap_df["macro_f1_test"]
    gap_df["balanced_accuracy_train_test_gap"] = (
        gap_df["balanced_accuracy_train"] - gap_df["balanced_accuracy_test"]
    )
    gap_df["macro_f1_train_valid_gap"] = gap_df["macro_f1_train"] - gap_df["macro_f1_valid"]
    gap_df["balanced_accuracy_train_valid_gap"] = (
        gap_df["balanced_accuracy_train"] - gap_df["balanced_accuracy_valid"]
    )
    summary = (
        gap_df.groupby("model_name", as_index=False)[
            [
                "macro_f1_train_test_gap",
                "balanced_accuracy_train_test_gap",
                "macro_f1_train_valid_gap",
                "balanced_accuracy_train_valid_gap",
                "brier_multiclass_test",
                "ece_confidence_test",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    return gap_df, summary


def write_curves(predictions: list[dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    test_predictions = [p for p in predictions if p["split"] == "test"]

    plt.figure(figsize=(7.5, 6))
    for model_name in sorted({p["model_name"] for p in test_predictions}):
        model_predictions = [p for p in test_predictions if p["model_name"] == model_name]
        y = np.concatenate([p["y_true"].to_numpy() for p in model_predictions])
        proba = np.vstack([p["proba"] for p in model_predictions])
        precision, recall, _ = precision_recall_curve((y == 0).astype(int), proba[:, 0])
        ap = average_precision_score((y == 0).astype(int), proba[:, 0])
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    plt.xlabel("Class 0 recall")
    plt.ylabel("Class 0 precision")
    plt.title("Class 0 Precision-Recall Curve")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "class0_precision_recall_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.5, 6))
    bins = np.linspace(0.0, 1.0, 11)
    for model_name in sorted({p["model_name"] for p in test_predictions}):
        model_predictions = [p for p in test_predictions if p["model_name"] == model_name]
        y = np.concatenate([p["y_true"].to_numpy() for p in model_predictions])
        proba = np.vstack([p["proba"] for p in model_predictions])
        pred = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
        correct = (pred == y).astype(float)
        xs: list[float] = []
        ys: list[float] = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                xs.append(float(conf[mask].mean()))
                ys.append(float(correct[mask].mean()))
        plt.plot(xs, ys, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Test Calibration Curve")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "test_calibration_curve.png", dpi=300)
    plt.close()


def write_gap_plot(gap_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = gap_df.melt(
        id_vars=["seed", "model_name"],
        value_vars=["macro_f1_train_test_gap", "balanced_accuracy_train_test_gap"],
        var_name="gap_metric",
        value_name="gap",
    )
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=plot_df, x="model_name", y="gap", hue="gap_metric")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Train-Test Generalization Gap")
    plt.tight_layout()
    plt.savefig(figures_dir / "generalization_gap_boxplot.png", dpi=300)
    plt.close()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flattened = df.copy()
    flattened.columns = [
        "_".join(str(part) for part in col if str(part) and str(part) != "nan")
        if isinstance(col, tuple)
        else str(col)
        for col in flattened.columns
    ]
    return flattened


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display_df = flatten_columns(df).copy()
    display_df = display_df.reset_index(drop=True)
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = [str(col) for col in display_df.columns]
    rows = display_df.astype(str).values.tolist()
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def write_report(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    gap_summary: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    test_summary = (
        test_df.groupby("model_name", as_index=False)[
            ["accuracy", "balanced_accuracy", "macro_f1", "class0_recall", "brier_multiclass", "ece_confidence"]
        ]
        .mean()
        .sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    )
    best_gap = gap_summary.copy()
    best_gap = flatten_columns(best_gap)
    report = [
        "# SCM_v3 Generalization Diagnostics",
        "",
        f"- Seeds: {', '.join(map(str, args.seeds[:1] if args.smoke_test else args.seeds))}",
        f"- Split: train/valid/test = {1 - args.test_size - (1 - args.test_size) * args.valid_size:.2f}/"
        f"{(1 - args.test_size) * args.valid_size:.2f}/{args.test_size:.2f}",
        "- Compared models: xgb_reference_adasyn, xgb_scm_v2_best, scm_v3_res_norule_single_flat_nocf, xgb_tabddpm_proto",
        "",
        "## Test Mean Metrics",
        "",
        dataframe_to_markdown(test_summary),
        "",
        "## Gap Summary",
        "",
        dataframe_to_markdown(best_gap),
        "",
        "## Bootstrap Test CI",
        "",
        dataframe_to_markdown(bootstrap_df),
        "",
    ]
    (output_dir / "generalization_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    bootstrap_iterations = 200 if args.smoke_test else args.bootstrap_iterations
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_names = [
        "xgb_reference_adasyn",
        "xgb_scm_v2_best",
        "scm_v3_res_norule_single_flat_nocf",
    ]
    if not args.skip_tabddpm:
        model_names.append("xgb_tabddpm_proto")

    if args.report_only:
        metrics_path = tables_dir / "split_metrics_by_seed.csv"
        bootstrap_path = tables_dir / "test_bootstrap_ci.csv"
        if not metrics_path.exists() or not bootstrap_path.exists():
            raise FileNotFoundError(
                "report-only requires split_metrics_by_seed.csv and test_bootstrap_ci.csv in the tables directory."
            )
        metrics_df = pd.read_csv(metrics_path)
        bootstrap_df = pd.read_csv(bootstrap_path)
        _, gap_summary = build_gap_tables(metrics_df)
        gap_summary.to_csv(
            tables_dir / "generalization_gap_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_report(output_dir, metrics_df, gap_summary, bootstrap_df, args)
        summary = {
            "seeds": sorted(metrics_df["seed"].dropna().astype(int).unique().tolist()),
            "valid_size_within_train": args.valid_size,
            "test_size": args.test_size,
            "model_names": model_names,
            "bootstrap_iterations": bootstrap_iterations,
            "input_file": str(args.input),
            "output_dir": str(output_dir),
            "report_only": True,
        }
        (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    tracker = ProgressTracker(output_dir, total_steps=len(seeds) * len(model_names))
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for seed in seeds:
        split_data = prepare_three_way_split(seed, args.valid_size, args.test_size, args.input)
        for model_name in model_names:
            tracker.log(f"Running {model_name}, seed={seed}", stage="fit", context={"seed": seed, "model": model_name})
            X_fit, y_fit, augmented_size = make_augmented_training_data(
                model_name,
                split_data,
                seed,
                args.tabddpm_epochs,
                args.smoke_test,
            )
            fit = fit_xgb_diagnostic(
                X_fit,
                y_fit,
                split_data.X_train_raw,
                split_data.X_valid_raw,
                split_data.X_test_raw,
                split_data.discrete_numeric_features,
                seed,
                augmented_size,
            )
            for split_name, y_true, proba in [
                ("train", split_data.y_train, fit.proba_train),
                ("valid", split_data.y_valid, fit.proba_valid),
                ("test", split_data.y_test, fit.proba_test),
            ]:
                rows.append(metric_row(seed, model_name, split_name, y_true, proba, fit))
                predictions.append(
                    {
                        "seed": seed,
                        "model_name": model_name,
                        "split": split_name,
                        "y_true": y_true,
                        "proba": proba,
                    }
                )
            tracker.log(f"Finished {model_name}, seed={seed}", stage="done", advance=1)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "split_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    split_summary = summarize_split_metrics(metrics_df)
    split_summary.to_csv(tables_dir / "split_metrics_summary.csv", index=True, encoding="utf-8-sig")
    gap_df, gap_summary = build_gap_tables(metrics_df)
    gap_df.to_csv(tables_dir / "generalization_gaps_by_seed.csv", index=False, encoding="utf-8-sig")
    flatten_columns(gap_summary).to_csv(
        tables_dir / "generalization_gap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_df = bootstrap_test_ci(predictions, bootstrap_iterations)
    bootstrap_df.to_csv(tables_dir / "test_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    write_curves(predictions, figures_dir)
    write_gap_plot(gap_df, figures_dir)
    write_report(output_dir, metrics_df, gap_summary, bootstrap_df, args)
    summary = {
        "seeds": seeds,
        "valid_size_within_train": args.valid_size,
        "test_size": args.test_size,
        "model_names": model_names,
        "bootstrap_iterations": bootstrap_iterations,
        "input_file": str(args.input),
        "output_dir": str(output_dir),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
