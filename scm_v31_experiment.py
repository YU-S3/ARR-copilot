from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from frontier_augmentation import clip_numeric_value, infer_discrete_numeric_features
from frontier_scm_v2_experiment import DEFAULT_SEEDS, ProgressTracker
from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE
from scm_v3_generalization_experiment import (
    PROJECT_DIR,
    compute_metrics,
    dataframe_to_markdown,
    fit_xgb_diagnostic,
    make_augmented_training_data,
    prepare_three_way_split,
)
from tabddpm_prototype_experiment import build_numeric_generation_columns, fit_tabddpm_proto


OUTPUT_DIR = PROJECT_DIR / "frontier_scm_v31_outputs"
CLASSES = [0, 1, 2]
DUAL_MAINLINES = ["xgb_scm_v2_best", "scm_v3_res_norule_single_flat_nocf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCM V3.1 calibration and TabDDPM follow-up experiment.")
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--valid-size", type=float, default=0.25)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tabddpm-sample-sizes", type=int, nargs="*", default=[12, 24, 48])
    parser.add_argument("--tabddpm-epochs-grid", type=int, nargs="*", default=[60, 120])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-tabddpm", action="store_true")
    return parser.parse_args()


def normalize_probability_rows(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=float), 1e-8, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def multiclass_log_loss(y_true: pd.Series, proba: np.ndarray) -> float:
    y_arr = y_true.to_numpy(dtype=int)
    clipped = np.clip(proba, 1e-8, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(y_arr)), y_arr])))


def apply_probability_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-8, 1.0))
    scaled = np.exp(logits / max(float(temperature), 1e-8))
    return normalize_probability_rows(scaled)


def fit_temperature(valid_proba: np.ndarray, y_valid: pd.Series) -> dict[str, float]:
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.5, 2.0, 16),
                np.linspace(2.25, 4.0, 8),
            ]
        )
    )
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in candidates:
        calibrated = apply_probability_temperature(valid_proba, float(temperature))
        loss = multiclass_log_loss(y_valid, calibrated)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return {"temperature": best_temperature, "valid_log_loss": best_loss}


def fit_ovr_sigmoid(valid_proba: np.ndarray, y_valid: pd.Series, seed: int) -> list[LogisticRegression]:
    y_arr = y_valid.to_numpy(dtype=int)
    calibrators: list[LogisticRegression] = []
    for cls in CLASSES:
        target = (y_arr == cls).astype(int)
        if target.min() == target.max():
            raise ValueError(f"Class {cls} is missing positive or negative samples for sigmoid calibration.")
        calibrator = LogisticRegression(random_state=seed, solver="lbfgs")
        calibrator.fit(valid_proba[:, [cls]], target)
        calibrators.append(calibrator)
    return calibrators


def apply_ovr_sigmoid(calibrators: list[LogisticRegression], proba: np.ndarray) -> np.ndarray:
    columns = [calibrator.predict_proba(proba[:, [idx]])[:, 1] for idx, calibrator in enumerate(calibrators)]
    return normalize_probability_rows(np.column_stack(columns))


def fit_ovr_isotonic(valid_proba: np.ndarray, y_valid: pd.Series) -> list[IsotonicRegression]:
    y_arr = y_valid.to_numpy(dtype=int)
    calibrators: list[IsotonicRegression] = []
    for cls in CLASSES:
        target = (y_arr == cls).astype(int)
        if target.min() == target.max():
            raise ValueError(f"Class {cls} is missing positive or negative samples for isotonic calibration.")
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(valid_proba[:, cls], target)
        calibrators.append(calibrator)
    return calibrators


def apply_ovr_isotonic(calibrators: list[IsotonicRegression], proba: np.ndarray) -> np.ndarray:
    columns = [calibrator.predict(proba[:, idx]) for idx, calibrator in enumerate(calibrators)]
    return normalize_probability_rows(np.column_stack(columns))


def collect_test_metric_row(
    *,
    seed: int,
    variant: str,
    model_name: str,
    family: str,
    y_test: pd.Series,
    test_proba: np.ndarray,
    augmented_size: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "family": family,
        "variant": variant,
        "model_name": model_name,
        "augmented_size": augmented_size,
        **compute_metrics(y_test, test_proba),
    }
    if extra:
        row.update(extra)
    return row


def run_mainline_calibration(
    *,
    seed: int,
    split_data: Any,
    model_name: str,
    smoke_test: bool,
) -> list[dict[str, Any]]:
    x_fit, y_fit, augmented_size = make_augmented_training_data(
        model_name,
        split_data,
        seed,
        tabddpm_epochs=20 if smoke_test else 120,
        smoke_test=smoke_test,
    )
    fit = fit_xgb_diagnostic(
        x_fit,
        y_fit,
        split_data.X_train_raw,
        split_data.X_valid_raw,
        split_data.X_test_raw,
        split_data.discrete_numeric_features,
        seed,
        augmented_size,
    )
    rows = [
        collect_test_metric_row(
            seed=seed,
            variant=f"{model_name}__uncalibrated",
            model_name=model_name,
            family="mainline_calibration",
            y_test=split_data.y_test,
            test_proba=fit.proba_test,
            augmented_size=augmented_size,
            extra={"calibration_method": "uncalibrated"},
        )
    ]

    temp_artifact = fit_temperature(fit.proba_valid, split_data.y_valid)
    temp_proba = apply_probability_temperature(fit.proba_test, temp_artifact["temperature"])
    rows.append(
        collect_test_metric_row(
            seed=seed,
            variant=f"{model_name}__temperature",
            model_name=model_name,
            family="mainline_calibration",
            y_test=split_data.y_test,
            test_proba=temp_proba,
            augmented_size=augmented_size,
            extra={"calibration_method": "temperature", **temp_artifact},
        )
    )

    try:
        sigmoid = fit_ovr_sigmoid(fit.proba_valid, split_data.y_valid, seed)
        sigmoid_proba = apply_ovr_sigmoid(sigmoid, fit.proba_test)
        rows.append(
            collect_test_metric_row(
                seed=seed,
                variant=f"{model_name}__ovr_sigmoid",
                model_name=model_name,
                family="mainline_calibration",
                y_test=split_data.y_test,
                test_proba=sigmoid_proba,
                augmented_size=augmented_size,
                extra={"calibration_method": "ovr_sigmoid"},
            )
        )
    except Exception as exc:
        rows.append(
            {
                "seed": seed,
                "family": "mainline_calibration",
                "variant": f"{model_name}__ovr_sigmoid",
                "model_name": model_name,
                "calibration_method": "ovr_sigmoid",
                "status": "failed",
                "error": str(exc),
            }
        )

    try:
        isotonic = fit_ovr_isotonic(fit.proba_valid, split_data.y_valid)
        isotonic_proba = apply_ovr_isotonic(isotonic, fit.proba_test)
        rows.append(
            collect_test_metric_row(
                seed=seed,
                variant=f"{model_name}__ovr_isotonic",
                model_name=model_name,
                family="mainline_calibration",
                y_test=split_data.y_test,
                test_proba=isotonic_proba,
                augmented_size=augmented_size,
                extra={"calibration_method": "ovr_isotonic"},
            )
        )
    except Exception as exc:
        rows.append(
            {
                "seed": seed,
                "family": "mainline_calibration",
                "variant": f"{model_name}__ovr_isotonic",
                "model_name": model_name,
                "calibration_method": "ovr_isotonic",
                "status": "failed",
                "error": str(exc),
            }
        )

    return rows


def generate_tabddpm_rows_fixed(
    x_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int,
    epochs: int,
    n_samples: int,
    smoke_test: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    class_zero = x_train_raw.loc[y_train == 0].copy()
    if len(class_zero) < 6 or n_samples <= 0:
        return pd.DataFrame(columns=x_train_raw.columns), pd.Series(dtype=int, name=y_train.name)

    feature_columns = build_numeric_generation_columns(x_train_raw)
    if not feature_columns:
        return pd.DataFrame(columns=x_train_raw.columns), pd.Series(dtype=int, name=y_train.name)

    prototype = fit_tabddpm_proto(
        class_zero,
        feature_columns,
        epochs=20 if smoke_test else epochs,
        seed=seed,
    )
    sample_steps = 12 if smoke_test else 24
    sampled_numeric = prototype.sample(n_samples=n_samples, steps=sample_steps, seed=seed + 77)
    sampled_df = pd.DataFrame(sampled_numeric, columns=feature_columns)
    support_cache = {col: pd.to_numeric(x_train_raw[col], errors="coerce") for col in feature_columns}
    discrete_features = set(infer_discrete_numeric_features(x_train_raw))

    rows: list[pd.Series] = []
    rng = np.random.default_rng(seed + 99)
    for idx in range(n_samples):
        base_row = class_zero.iloc[int(rng.integers(0, len(class_zero)))].copy()
        candidate = base_row.copy()
        for col in feature_columns:
            value = float(sampled_df.iloc[idx][col])
            value = clip_numeric_value(value, support_cache[col])
            if col in discrete_features:
                value = float(np.round(value))
            candidate[col] = value
        rows.append(candidate)

    augmented_df = pd.DataFrame(rows).reset_index(drop=True)
    y_aug = pd.Series(np.zeros(len(augmented_df), dtype=int), name=y_train.name)
    return augmented_df, y_aug


def run_tabddpm_grid(
    *,
    seed: int,
    split_data: Any,
    n_samples: int,
    epochs: int,
    smoke_test: bool,
) -> dict[str, Any]:
    x_aug, y_aug = generate_tabddpm_rows_fixed(
        split_data.X_train_raw,
        split_data.y_train,
        seed=seed,
        epochs=epochs,
        n_samples=n_samples,
        smoke_test=smoke_test,
    )
    x_fit = pd.concat([split_data.X_train_raw, x_aug], axis=0, ignore_index=True)
    y_fit = pd.concat([split_data.y_train, y_aug], axis=0, ignore_index=True)
    fit = fit_xgb_diagnostic(
        x_fit,
        y_fit,
        split_data.X_train_raw,
        split_data.X_valid_raw,
        split_data.X_test_raw,
        split_data.discrete_numeric_features,
        seed,
        int(len(x_aug)),
    )
    variant = f"xgb_tabddpm_n{n_samples}_e{epochs}"
    return collect_test_metric_row(
        seed=seed,
        variant=variant,
        model_name="xgb_tabddpm_grid",
        family="tabddpm_grid",
        y_test=split_data.y_test,
        test_proba=fit.proba_test,
        augmented_size=int(len(x_aug)),
        extra={"tabddpm_n_samples": n_samples, "tabddpm_epochs": epochs},
    )


def summarize_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    status = rows_df["status"].fillna("success") if "status" in rows_df.columns else pd.Series("success", index=rows_df.index)
    valid_df = rows_df[status != "failed"].copy()
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "ovr_roc_auc_macro",
        "class0_recall",
        "class1_recall",
        "class2_recall",
        "class0_average_precision",
        "brier_multiclass",
        "ece_confidence",
        "augmented_size",
    ]
    present_metrics = [metric for metric in metrics if metric in valid_df.columns]
    summary = valid_df.groupby(["family", "variant", "model_name"], as_index=False)[present_metrics].agg(["mean", "std"])
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary.reset_index(drop=True)


def plot_v31_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.copy()
    needed = {"variant", "balanced_accuracy_mean", "macro_f1_mean", "class0_recall_mean"}
    if not needed.issubset(set(plot_df.columns)):
        return
    plot_df = plot_df.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=False).head(16)
    melted = plot_df.melt(
        id_vars=["variant"],
        value_vars=["balanced_accuracy_mean", "macro_f1_mean", "class0_recall_mean"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(13, 6))
    sns.barplot(data=melted, x="variant", y="value", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.0, 1.02)
    plt.title("SCM V3.1 top variants")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_report(output_dir: Path, rows_df: pd.DataFrame, summary_df: pd.DataFrame, args: argparse.Namespace) -> None:
    valid_summary = summary_df.copy()
    if "balanced_accuracy_mean" in valid_summary.columns:
        valid_summary = valid_summary.sort_values(
            ["balanced_accuracy_mean", "macro_f1_mean", "class0_recall_mean"],
            ascending=False,
        )
    top_cols = [
        col
        for col in [
            "family",
            "variant",
            "model_name",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "class0_recall_mean",
            "brier_multiclass_mean",
            "ece_confidence_mean",
            "augmented_size_mean",
        ]
        if col in valid_summary.columns
    ]
    report = [
        "# SCM V3.1 Experiment Report",
        "",
        f"- Seeds: {', '.join(map(str, args.seeds[:1] if args.smoke_test else args.seeds))}",
        f"- Split: train/valid/test = {1 - args.test_size - (1 - args.test_size) * args.valid_size:.2f}/"
        f"{(1 - args.test_size) * args.valid_size:.2f}/{args.test_size:.2f}",
        "- Mainline calibration: SCM-v2 best and SCM_v3 best with uncalibrated, temperature, OVR sigmoid, OVR isotonic.",
        "- TabDDPM grid: fixed generated class-0 sample sizes crossed with epoch settings.",
        "",
        "## Ranked Summary",
        "",
        dataframe_to_markdown(valid_summary[top_cols]),
        "",
        "## Notes",
        "",
        "- V3.1 is a follow-up validation experiment, not a replacement for the original V3 report.",
        "- Calibration methods are fit on the validation split and evaluated only on the test split.",
        "- TabDDPM grid variants are compared against the dual mainlines using the same split protocol.",
        "",
    ]
    (output_dir / "SCM_v31_experiment_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    sample_sizes = [] if args.skip_tabddpm else (args.tabddpm_sample_sizes[:1] if args.smoke_test else args.tabddpm_sample_sizes)
    epoch_grid = [] if args.skip_tabddpm else (args.tabddpm_epochs_grid[:1] if args.smoke_test else args.tabddpm_epochs_grid)

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    total_steps = len(seeds) * (len(DUAL_MAINLINES) + len(sample_sizes) * len(epoch_grid))
    tracker = ProgressTracker(output_dir, total_steps=max(total_steps, 1))
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        split_data = prepare_three_way_split(seed, args.valid_size, args.test_size, args.input)
        for model_name in DUAL_MAINLINES:
            tracker.log(f"Running calibration for {model_name}, seed={seed}", stage="mainline_calibration")
            rows.extend(
                run_mainline_calibration(
                    seed=seed,
                    split_data=split_data,
                    model_name=model_name,
                    smoke_test=args.smoke_test,
                )
            )
            pd.DataFrame(rows).to_csv(tables_dir / "v31_results_by_seed.partial.csv", index=False, encoding="utf-8-sig")
            tracker.log(f"Finished calibration for {model_name}, seed={seed}", stage="mainline_done", advance=1)

        for n_samples in sample_sizes:
            for epochs in epoch_grid:
                tracker.log(
                    f"Running TabDDPM grid n={n_samples}, epochs={epochs}, seed={seed}",
                    stage="tabddpm_grid",
                )
                rows.append(
                    run_tabddpm_grid(
                        seed=seed,
                        split_data=split_data,
                        n_samples=n_samples,
                        epochs=epochs,
                        smoke_test=args.smoke_test,
                    )
                )
                pd.DataFrame(rows).to_csv(
                    tables_dir / "v31_results_by_seed.partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                tracker.log(
                    f"Finished TabDDPM grid n={n_samples}, epochs={epochs}, seed={seed}",
                    stage="tabddpm_done",
                    advance=1,
                )

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(tables_dir / "v31_results_by_seed.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize_rows(rows_df)
    summary_df.to_csv(tables_dir / "v31_summary.csv", index=False, encoding="utf-8-sig")
    plot_v31_summary(summary_df, figures_dir / "v31_top_variants.png")
    write_report(output_dir, rows_df, summary_df, args)
    experiment_summary = {
        "seeds": seeds,
        "valid_size_within_train": args.valid_size,
        "test_size": args.test_size,
        "dual_mainlines": DUAL_MAINLINES,
        "tabddpm_sample_sizes": sample_sizes,
        "tabddpm_epochs_grid": epoch_grid,
        "input_file": str(args.input),
        "skip_tabddpm": args.skip_tabddpm,
        "output_dir": str(output_dir),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
