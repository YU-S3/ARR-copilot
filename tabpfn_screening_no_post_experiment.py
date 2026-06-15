from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, build_preprocessor, prepare_data
from screening_0428_experiment import (
    FEATURE_POLICIES,
    ProgressTracker,
    apply_feature_policy,
    build_metric_row,
    build_ranking_summary,
    collect_metrics,
    gap_rows_from_fold_metrics,
    requested_tasks,
    task_classes,
    task_target,
    write_feature_policy_files,
    aggregate_metrics,
)


PROJECT_DIR = Path(__file__).resolve().parent
TABPFN_DIR = PROJECT_DIR / "TabPFN"
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 2027]
DEFAULT_CV_FOLDS = [5, 10]

CHECKPOINTS = {
    "multiclass": "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
    "binary": "tabpfn-v3-classifier-v3_20260417_binary.ckpt",
    "default": "tabpfn-v3-classifier-v3_default.ckpt",
}


@dataclass(frozen=True)
class TabPFNVariant:
    name: str
    task: str
    checkpoint_key: str
    balance_probabilities: bool


VARIANTS = {
    "tabpfn3_multiclass": TabPFNVariant(
        name="tabpfn3_multiclass",
        task="three",
        checkpoint_key="multiclass",
        balance_probabilities=False,
    ),
    "tabpfn3_multiclass_balanced": TabPFNVariant(
        name="tabpfn3_multiclass_balanced",
        task="three",
        checkpoint_key="multiclass",
        balance_probabilities=True,
    ),
    "tabpfn3_default_multiclass_balanced": TabPFNVariant(
        name="tabpfn3_default_multiclass_balanced",
        task="three",
        checkpoint_key="default",
        balance_probabilities=True,
    ),
    "tabpfn3_binary": TabPFNVariant(
        name="tabpfn3_binary",
        task="binary",
        checkpoint_key="binary",
        balance_probabilities=False,
    ),
    "tabpfn3_binary_balanced": TabPFNVariant(
        name="tabpfn3_binary_balanced",
        task="binary",
        checkpoint_key="binary",
        balance_probabilities=True,
    ),
    "tabpfn3_default_binary_balanced": TabPFNVariant(
        name="tabpfn3_default_binary_balanced",
        task="binary",
        checkpoint_key="default",
        balance_probabilities=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local TabPFN-3 screening experiment without post-test features."
    )
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tabpfn-dir", type=Path, default=TABPFN_DIR)
    parser.add_argument("--task-mode", choices=["three", "binary", "both"], default="both")
    parser.add_argument(
        "--feature-policy",
        choices=FEATURE_POLICIES,
        nargs="+",
        default=["screening_no_post"],
        help="Use screening_no_post for the primary experiment; add full_reference/post_mask_stress for leakage audit.",
    )
    parser.add_argument(
        "--include-audit-policies",
        action="store_true",
        help="Run full_reference, screening_no_post, and post_mask_stress using the same TabPFN variants.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--cv-folds", type=int, nargs="*", default=DEFAULT_CV_FOLDS)
    parser.add_argument("--variant-set", choices=["core", "extended"], default="core")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(VARIANTS),
        help="Explicit TabPFN variants to run. Overrides --variant-set.",
    )
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--fit-mode",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
        default="fit_preprocessors",
    )
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    mode = "smoke" if args.smoke_test else "full"
    return PROJECT_DIR / "rerun_0428_outputs" / "tabpfn_screening_no_post" / mode


def variant_names_for(task: str, variant_set: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return [name for name in explicit if VARIANTS[name].task == task]
    if variant_set == "core":
        return ["tabpfn3_multiclass_balanced"] if task == "three" else ["tabpfn3_binary_balanced"]
    if task == "three":
        return [
            "tabpfn3_multiclass",
            "tabpfn3_multiclass_balanced",
            "tabpfn3_default_multiclass_balanced",
        ]
    return [
        "tabpfn3_binary",
        "tabpfn3_binary_balanced",
        "tabpfn3_default_binary_balanced",
    ]


def validate_checkpoints(tabpfn_dir: Path, variant_names: list[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    missing: list[Path] = []
    for name in variant_names:
        spec = VARIANTS[name]
        path = tabpfn_dir / CHECKPOINTS[spec.checkpoint_key]
        paths[name] = str(path)
        if not path.exists():
            missing.append(path)
    if missing:
        raise FileNotFoundError("Missing TabPFN checkpoint(s): " + ", ".join(str(path) for path in missing))
    return paths


def fit_transform_tabpfn_inputs(
    X_train_raw: pd.DataFrame,
    X_eval_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X_train_raw.columns if col not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_t = preprocessor.fit_transform(X_train_raw)
    X_eval_t = preprocessor.transform(X_eval_raw)
    feature_names = list(preprocessor.get_feature_names_out())
    categorical_indices = [
        idx for idx, name in enumerate(feature_names) if str(name).startswith("cat__")
    ]
    return (
        np.asarray(X_train_t, dtype=np.float32),
        np.asarray(X_eval_t, dtype=np.float32),
        categorical_indices,
        feature_names,
    )


def align_proba(model: Any, proba: np.ndarray, classes: list[int]) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    model_classes = getattr(model, "classes_", np.arange(arr.shape[1]))
    aligned = np.zeros((arr.shape[0], len(classes)), dtype=float)
    for col_idx, cls in enumerate(model_classes):
        cls_int = int(cls)
        if cls_int in classes:
            aligned[:, classes.index(cls_int)] = arr[:, col_idx]
    aligned = np.clip(aligned, 1e-8, 1.0)
    return aligned / aligned.sum(axis=1, keepdims=True)


def fit_predict_tabpfn(
    *,
    variant_name: str,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_eval_raw: pd.DataFrame,
    seed: int,
    checkpoint_paths: dict[str, str],
    n_estimators: int,
    device: str,
    fit_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from tabpfn import TabPFNClassifier

    variant = VARIANTS[variant_name]
    X_train, X_eval, categorical_indices, feature_names = fit_transform_tabpfn_inputs(
        X_train_raw,
        X_eval_raw,
    )
    model = TabPFNClassifier(
        n_estimators=n_estimators,
        categorical_features_indices=categorical_indices or None,
        balance_probabilities=variant.balance_probabilities,
        model_path=checkpoint_paths[variant_name],
        device=device,
        fit_mode=fit_mode,
        random_state=seed,
        show_progress_bar=False,
    )
    model.fit(X_train, y_train.to_numpy(dtype=int))
    proba = align_proba(model, model.predict_proba(X_eval), task_classes(variant.task))
    metadata = {
        "train_size": int(len(y_train)),
        "resampled_train_size": int(len(y_train)),
        "feature_count": int(X_train.shape[1]),
        "categorical_feature_count": int(len(categorical_indices)),
        "checkpoint_path": checkpoint_paths[variant_name],
        "feature_names": feature_names,
    }
    return proba, metadata


def estimate_total_steps(
    tasks: list[str],
    policies: list[str],
    seeds: list[int],
    folds: list[int],
    variant_set: str,
    explicit_models: list[str] | None,
    skip_cv: bool,
) -> int:
    per_policy_seed = sum(len(variant_names_for(task, variant_set, explicit_models)) for task in tasks)
    seed_steps = len(seeds) * len(policies) * per_policy_seed
    cv_steps = 0 if skip_cv else len(folds) * len(policies) * per_policy_seed
    return max(seed_steps + cv_steps + 4, 1)


def run_seed_suite(
    prepared_X: pd.DataFrame,
    prepared_y: pd.Series,
    *,
    tasks: list[str],
    policies: list[str],
    seeds: list[int],
    variant_set: str,
    explicit_models: list[str] | None,
    checkpoint_paths: dict[str, str],
    n_estimators: int,
    device: str,
    fit_mode: str,
    tracker: ProgressTracker,
    tables_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
                for variant_name in variant_names_for(task, variant_set, explicit_models):
                    tracker.log(
                        f"seed={seed} {task}/{policy}/{variant_name}",
                        stage="seed_model",
                        context={"seed": seed, "task": task, "feature_policy": policy, "model_name": variant_name},
                    )
                    proba, metadata = fit_predict_tabpfn(
                        variant_name=variant_name,
                        X_train_raw=view.X_train,
                        y_train=y_train,
                        X_eval_raw=view.X_eval,
                        seed=seed,
                        checkpoint_paths=checkpoint_paths,
                        n_estimators=n_estimators,
                        device=device,
                        fit_mode=fit_mode,
                    )
                    rows.append(
                        {
                            **build_metric_row(
                                task=task,
                                feature_policy=policy,
                                seed=seed,
                                model_name=variant_name,
                                y_true=y_test,
                                proba=proba,
                                train_size=metadata["train_size"],
                                eval_size=len(y_test),
                                resampled_train_size=metadata["resampled_train_size"],
                                augmented_size=0,
                                split_type="test",
                            ),
                            "checkpoint_path": metadata["checkpoint_path"],
                            "feature_count": metadata["feature_count"],
                            "categorical_feature_count": metadata["categorical_feature_count"],
                        }
                    )
                    pd.DataFrame(rows).to_csv(
                        tables_dir / "tabpfn_metrics_by_seed.partial.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    tracker.log(
                        f"done seed={seed} {task}/{policy}/{variant_name}",
                        stage="seed_model_done",
                        advance=1,
                        context={"seed": seed, "task": task, "feature_policy": policy, "model_name": variant_name},
                    )
    return pd.DataFrame(rows)


def run_cv_suite(
    prepared_X: pd.DataFrame,
    prepared_y: pd.Series,
    *,
    tasks: list[str],
    policies: list[str],
    folds: list[int],
    variant_set: str,
    explicit_models: list[str] | None,
    checkpoint_paths: dict[str, str],
    n_estimators: int,
    device: str,
    fit_mode: str,
    tracker: ProgressTracker,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for fold_count in folds:
        for policy in policies:
            for task in tasks:
                y_for_split = task_target(prepared_y, task)
                cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42 + fold_count)
                for model_idx, variant_name in enumerate(variant_names_for(task, variant_set, explicit_models)):
                    train_metrics_by_fold: list[dict[str, Any]] = []
                    valid_metrics_by_fold: list[dict[str, Any]] = []
                    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(prepared_X, y_for_split), start=1):
                        X_train_base = prepared_X.iloc[train_idx].reset_index(drop=True)
                        X_valid_base = prepared_X.iloc[valid_idx].reset_index(drop=True)
                        y_train_base = prepared_y.iloc[train_idx].reset_index(drop=True)
                        y_valid_base = prepared_y.iloc[valid_idx].reset_index(drop=True)
                        y_train = task_target(y_train_base, task)
                        y_valid = task_target(y_valid_base, task)
                        view = apply_feature_policy(X_train_base, X_valid_base, policy)
                        seed = 20000 + fold_count * 100 + fold_idx + model_idx * 17
                        tracker.log(
                            f"cv={fold_count} fold={fold_idx} {task}/{policy}/{variant_name}",
                            stage="cross_validation",
                            context={
                                "fold_count": fold_count,
                                "fold_index": fold_idx,
                                "task": task,
                                "feature_policy": policy,
                                "model_name": variant_name,
                            },
                        )
                        X_eval_both = pd.concat([view.X_train_eval, view.X_eval], axis=0, ignore_index=True)
                        proba_both, metadata = fit_predict_tabpfn(
                            variant_name=variant_name,
                            X_train_raw=view.X_train,
                            y_train=y_train,
                            X_eval_raw=X_eval_both,
                            seed=seed,
                            checkpoint_paths=checkpoint_paths,
                            n_estimators=n_estimators,
                            device=device,
                            fit_mode=fit_mode,
                        )
                        train_proba = proba_both[: len(y_train)]
                        valid_proba = proba_both[len(y_train) :]
                        train_metrics = collect_metrics(y_train, train_proba, task)
                        valid_metrics = collect_metrics(y_valid, valid_proba, task)
                        train_metrics_by_fold.append(train_metrics)
                        valid_metrics_by_fold.append(valid_metrics)
                        cv_rows.append(
                            {
                                "task": task,
                                "feature_policy": policy,
                                "fold_count": fold_count,
                                "fold_index": fold_idx,
                                "model_name": variant_name,
                                "split_type": "valid",
                                "train_size": int(len(y_train)),
                                "valid_size": int(len(y_valid)),
                                "augmented_size": 0,
                                "resampled_train_size": metadata["resampled_train_size"],
                                "checkpoint_path": metadata["checkpoint_path"],
                                "feature_count": metadata["feature_count"],
                                "categorical_feature_count": metadata["categorical_feature_count"],
                                **valid_metrics,
                            }
                        )
                        pd.DataFrame(cv_rows).to_csv(
                            tables_dir / "tabpfn_cross_validation_by_fold.partial.csv",
                            index=False,
                            encoding="utf-8-sig",
                        )
                    gap_rows.extend(
                        gap_rows_from_fold_metrics(
                            train_metrics_by_fold,
                            valid_metrics_by_fold,
                            task=task,
                            policy=policy,
                            fold_count=fold_count,
                            model_name=variant_name,
                        )
                    )
                    tracker.log(
                        f"done cv={fold_count} {task}/{policy}/{variant_name}",
                        stage="cross_validation_done",
                        advance=1,
                        context={"fold_count": fold_count, "task": task, "feature_policy": policy, "model_name": variant_name},
                    )
    return pd.DataFrame(cv_rows), pd.DataFrame(gap_rows)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.seeds = [args.seeds[0] if args.seeds else 42]
        args.cv_folds = [2]
        args.variant_set = "core"
        args.n_estimators = min(args.n_estimators, 2)
    if args.include_audit_policies:
        args.feature_policy = ["full_reference", "screening_no_post", "post_mask_stress"]

    tasks = requested_tasks(args.task_mode)
    policies = list(dict.fromkeys(args.feature_policy))
    seeds = list(dict.fromkeys(args.seeds))
    folds = sorted({int(fold) for fold in args.cv_folds if int(fold) >= 2})
    output_dir = resolve_output_dir(args)
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    requested_variant_names: list[str] = []
    for task in tasks:
        requested_variant_names.extend(variant_names_for(task, args.variant_set, args.models))
    requested_variant_names = list(dict.fromkeys(requested_variant_names))
    checkpoint_paths = validate_checkpoints(args.tabpfn_dir, requested_variant_names)

    tracker = ProgressTracker(
        output_dir,
        estimate_total_steps(
            tasks,
            policies,
            seeds,
            folds,
            args.variant_set,
            args.models,
            args.skip_cv,
        ),
    )
    tracker.log(
        "TabPFN screening experiment started",
        stage="startup",
        context={
            "tasks": tasks,
            "feature_policies": policies,
            "seeds": seeds,
            "cv_folds": folds,
            "variant_set": args.variant_set,
            "models": requested_variant_names,
            "n_estimators": args.n_estimators,
            "device": args.device,
            "fit_mode": args.fit_mode,
        },
    )

    prepared = prepare_data(args.input)
    write_feature_policy_files(output_dir, list(prepared.X.columns), policies)
    target_distribution = prepared.y.value_counts().sort_index().to_dict()
    tracker.log(
        "data loaded",
        stage="data",
        advance=1,
        context={
            "shape": list(prepared.X.shape),
            "target_distribution": {str(k): int(v) for k, v in target_distribution.items()},
            "checkpoint_paths": checkpoint_paths,
        },
    )

    seed_df = run_seed_suite(
        prepared.X,
        prepared.y,
        tasks=tasks,
        policies=policies,
        seeds=seeds,
        variant_set=args.variant_set,
        explicit_models=args.models,
        checkpoint_paths=checkpoint_paths,
        n_estimators=args.n_estimators,
        device=args.device,
        fit_mode=args.fit_mode,
        tracker=tracker,
        tables_dir=tables_dir,
    )
    seed_df.to_csv(tables_dir / "tabpfn_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    seed_summary = aggregate_metrics(seed_df, ["task", "feature_policy", "model_name"], seed_base=51000)
    seed_summary = seed_summary.sort_values(
        ["task", "feature_policy", "balanced_accuracy_mean", "macro_f1_mean"],
        ascending=[True, True, False, False],
    )
    seed_summary.to_csv(tables_dir / "tabpfn_metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    tracker.log("seed suite summarized", stage="seed_summary", advance=1)

    cv_df = pd.DataFrame()
    gap_df = pd.DataFrame()
    cv_summary = pd.DataFrame()
    if not args.skip_cv:
        cv_df, gap_df = run_cv_suite(
            prepared.X,
            prepared.y,
            tasks=tasks,
            policies=policies,
            folds=folds,
            variant_set=args.variant_set,
            explicit_models=args.models,
            checkpoint_paths=checkpoint_paths,
            n_estimators=args.n_estimators,
            device=args.device,
            fit_mode=args.fit_mode,
            tracker=tracker,
            tables_dir=tables_dir,
        )
        cv_df.to_csv(tables_dir / "tabpfn_cross_validation_by_fold.csv", index=False, encoding="utf-8-sig")
        cv_summary = aggregate_metrics(
            cv_df,
            ["task", "feature_policy", "fold_count", "model_name"],
            seed_base=61000,
        )
        cv_summary = cv_summary.sort_values(
            ["task", "feature_policy", "fold_count", "balanced_accuracy_mean", "macro_f1_mean"],
            ascending=[True, True, True, False, False],
        )
        cv_summary.to_csv(tables_dir / "tabpfn_cross_validation_mean_var.csv", index=False, encoding="utf-8-sig")
        gap_df.to_csv(tables_dir / "tabpfn_overfitting_indicators.csv", index=False, encoding="utf-8-sig")
        tracker.log("cross validation summarized", stage="cv_summary", advance=1)

    ranking = build_ranking_summary(seed_summary, gap_df)
    ranking.to_csv(tables_dir / "tabpfn_ranking_summary.csv", index=False, encoding="utf-8-sig")
    tracker.log("ranking written", stage="ranking", advance=1)

    summary = {
        "input_file": str(args.input),
        "output_dir": str(output_dir),
        "tabpfn_dir": str(args.tabpfn_dir),
        "smoke_test": bool(args.smoke_test),
        "tasks": tasks,
        "feature_policies": policies,
        "seeds": seeds,
        "cv_folds": [] if args.skip_cv else folds,
        "variant_set": args.variant_set,
        "models": requested_variant_names,
        "n_estimators": int(args.n_estimators),
        "device": args.device,
        "fit_mode": args.fit_mode,
        "checkpoint_paths": checkpoint_paths,
        "target_distribution": {str(k): int(v) for k, v in target_distribution.items()},
        "tables": {
            "metrics_by_seed": "tables/tabpfn_metrics_by_seed.csv",
            "metrics_mean_std": "tables/tabpfn_metrics_mean_std.csv",
            "cross_validation_by_fold": "tables/tabpfn_cross_validation_by_fold.csv" if not args.skip_cv else None,
            "cross_validation_mean_var": "tables/tabpfn_cross_validation_mean_var.csv" if not args.skip_cv else None,
            "overfitting_indicators": "tables/tabpfn_overfitting_indicators.csv" if not args.skip_cv else None,
            "ranking_summary": "tables/tabpfn_ranking_summary.csv",
        },
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tracker.finish()
    print(f"TabPFN screening experiment completed. Output: {output_dir}")
    if not ranking.empty:
        print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
