from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from env_utils import get_tabpfn_token
from frontier_scm_v2_experiment import (
    DEFAULT_SEEDS,
    ProgressTracker,
    SCMMixV2Augmentor,
    aggregate_metrics,
    build_hard_case_index_map,
    fit_xgb_from_raw,
    prepare_split_data,
    run_tabpfn_with_raw,
)
from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, evaluate_predictions
from scm_v3_augmentation import (
    SCMMixV3Augmentor,
    evaluate_model_from_raw,
    summarize_model_metrics,
)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["axes.unicode_minus"] = False

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "frontier_scm_v3_outputs"
TARGET_LABELS = {0: "非确诊", 1: "确诊", 2: "灰色区域"}
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
    "treat_mix_prob": 0.40,
    "sampler_strength": 0.50,
    "node_sampler": "residual",
    "rule_filter_mode": "off",
    "teacher_filter_mode": "single",
    "curriculum_mode": "off",
    "counterfactual_mode": "off",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCM_v3 并行改进实验脚本。")
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE, help="输入 Excel 文件路径。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出目录。")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS, help="随机种子列表。")
    parser.add_argument("--smoke-test", action="store_true", help="只跑极小配置集。")
    parser.add_argument("--best-only", action="store_true", help="只跑此前最优的 SCM-v3 简单残差配置。")
    parser.add_argument("--skip-tabpfn", action="store_true", help="跳过 TabPFN 扩展。")
    parser.add_argument("--enable-remote-teacher", action="store_true", help="允许在异构教师中过滤时调用 TabPFN。")
    return parser.parse_args()


def write_partial_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_boxplot(metrics_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if metrics_df.empty:
        return
    plt.figure(figsize=(12, 6))
    ordered = (
        metrics_df.groupby("model_name")[metric]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(data=metrics_df, x="model_name", y=metric, order=ordered)
    plt.xticks(rotation=35, ha="right")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_phase1_configs(smoke_test: bool, best_only: bool = False) -> list[dict[str, Any]]:
    if best_only:
        return [SCM_V3_BEST_CONFIG.copy()]
    configs: list[dict[str, Any]] = []
    for node_sampler in ["residual", "conditional_kde"]:
        for rule_filter_mode in ["off", "medical_rules"]:
            for teacher_filter_mode in ["single", "hetero_consensus"]:
                for curriculum_mode in ["off", "two_stage"]:
                    configs.append(
                        {
                            "seed_strategy": "hard_case_seed",
                            "target_classes": "class0_only",
                            "treat_mix_prob": 0.40,
                            "sampler_strength": 0.50 if node_sampler == "residual" else 0.65,
                            "node_sampler": node_sampler,
                            "rule_filter_mode": rule_filter_mode,
                            "teacher_filter_mode": teacher_filter_mode,
                            "curriculum_mode": curriculum_mode,
                            "counterfactual_mode": "off",
                        }
                    )
    if smoke_test:
        return [
            configs[0],
            configs[-1],
        ]
    return configs


def config_to_name(config: dict[str, Any]) -> str:
    sampler_tag = "kde" if config["node_sampler"] == "conditional_kde" else "res"
    rule_tag = "rule" if config["rule_filter_mode"] == "medical_rules" else "norule"
    teacher_tag = "hetero" if config["teacher_filter_mode"] == "hetero_consensus" else "single"
    curriculum_tag = "cur" if config["curriculum_mode"] == "two_stage" else "flat"
    cf_tag = "cfdo" if config["counterfactual_mode"] == "treatment_do" else "nocf"
    return f"scm_v3_{sampler_tag}_{rule_tag}_{teacher_tag}_{curriculum_tag}_{cf_tag}"


def fit_xgb_with_safe_adasyn(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    discrete_numeric_features: list[str],
    *,
    seed: int,
) -> Any:
    try:
        return fit_xgb_from_raw(
            X_train_raw,
            y_train,
            X_test_raw,
            discrete_numeric_features,
            seed=seed,
            use_adasyn=True,
        )
    except ValueError as exc:
        if "No samples will be generated with the provided ratio settings" not in str(exc):
            raise
        return fit_xgb_from_raw(
            X_train_raw,
            y_train,
            X_test_raw,
            discrete_numeric_features,
            seed=seed,
            use_adasyn=False,
        )


def fit_scm_v2_reference(split_data: Any, seed: int) -> tuple[Any, int]:
    hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)
    augmentor = SCMMixV2Augmentor(
        random_state=seed,
        seed_strategy=SCM_V2_REFERENCE_CONFIG["seed_strategy"],
        target_classes=SCM_V2_REFERENCE_CONFIG["target_classes"],
        treat_mix_prob=SCM_V2_REFERENCE_CONFIG["treat_mix_prob"],
        residual_scale=SCM_V2_REFERENCE_CONFIG["residual_scale"],
        teacher_mode=SCM_V2_REFERENCE_CONFIG["teacher_mode"],
    )
    aug_result = augmentor.generate(split_data.X_train_raw, split_data.y_train, hard_case_map)
    X_train_aug = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
    y_train_aug = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)
    train_result = fit_xgb_with_safe_adasyn(
        X_train_aug,
        y_train_aug,
        split_data.X_test_raw,
        split_data.discrete_numeric_features,
        seed=seed,
    )
    return train_result, int(len(aug_result.X_aug))


def fit_scm_v3_config(
    split_data: Any,
    seed: int,
    config: dict[str, Any],
    *,
    use_remote_teacher: bool = False,
) -> tuple[Any, int, pd.DataFrame]:
    hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)

    def _run_single_stage(stage_treat_mix_prob: float, stage_sampler_strength: float, counterfactual_mode: str) -> Any:
        augmentor = SCMMixV3Augmentor(
            random_state=seed,
            seed_strategy=config["seed_strategy"],
            target_classes=config["target_classes"],
            treat_mix_prob=stage_treat_mix_prob,
            sampler_strength=stage_sampler_strength,
            node_sampler=config["node_sampler"],
            rule_filter_mode=config["rule_filter_mode"],
            teacher_filter_mode=config["teacher_filter_mode"],
            counterfactual_mode=counterfactual_mode,
            project_dir=PROJECT_DIR,
            use_remote_teacher=use_remote_teacher,
        )
        return augmentor.generate(split_data.X_train_raw, split_data.y_train, hard_case_map)

    if config["curriculum_mode"] == "two_stage":
        stage1 = _run_single_stage(config["treat_mix_prob"] * 0.55, max(0.25, config["sampler_strength"] * 0.55), "off")
        stage2 = _run_single_stage(config["treat_mix_prob"], config["sampler_strength"], config["counterfactual_mode"])
        augmented_df = pd.concat([stage1.X_aug, stage2.X_aug], axis=0, ignore_index=True)
        y_aug = pd.concat([stage1.y_aug, stage2.y_aug], axis=0, ignore_index=True)
        audit_df = pd.concat([stage1.audit, stage2.audit], axis=0, ignore_index=True)
    else:
        stage = _run_single_stage(config["treat_mix_prob"], config["sampler_strength"], config["counterfactual_mode"])
        augmented_df = stage.X_aug
        y_aug = stage.y_aug
        audit_df = stage.audit

    X_train_aug = pd.concat([split_data.X_train_raw, augmented_df], axis=0, ignore_index=True)
    y_train_aug = pd.concat([split_data.y_train, y_aug], axis=0, ignore_index=True)
    train_result = fit_xgb_with_safe_adasyn(
        X_train_aug,
        y_train_aug,
        split_data.X_test_raw,
        split_data.discrete_numeric_features,
        seed=seed,
    )
    return train_result, int(len(augmented_df)), audit_df


def build_leaderboards(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_df = summary_df.sort_values(
        ["accuracy_mean", "macro_f1_mean", "ovr_roc_auc_macro_mean", "balanced_accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)
    minority_df = summary_df.sort_values(
        ["balanced_accuracy_mean", "class0_recall_mean", "macro_f1_mean", "accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)
    return overall_df, minority_df


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configs = make_phase1_configs(args.smoke_test, args.best_only)
    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    run_tabpfn = bool(get_tabpfn_token(PROJECT_DIR)) and not args.skip_tabpfn
    total_steps = len(seeds) * (2 + len(configs) + (2 if run_tabpfn else 0))
    tracker = ProgressTracker(output_dir, total_steps=max(total_steps, 1))
    tracker.log(
        "SCM_v3 实验启动",
        stage="startup",
        context={
            "smoke_test": args.smoke_test,
            "best_only": args.best_only,
            "seed_count": len(seeds),
            "input_file": str(args.input),
        },
    )

    metrics_rows: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []
    phase3_rows: list[dict[str, Any]] = []

    for seed in seeds:
        split_data = prepare_split_data(args.input, seed)
        tracker.log(f"运行基线模型，seed={seed}", stage="baseline_reference")
        reference_result = fit_xgb_with_safe_adasyn(
            split_data.X_train_raw,
            split_data.y_train,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed=seed,
        )
        metrics_rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_reference_adasyn",
                y_test=split_data.y_test,
                proba=reference_result.proba,
                augmented_size=0,
                train_size=reference_result.train_size,
                resampled_train_size=reference_result.resampled_train_size,
            )
        )
        write_partial_table(metrics_rows, tables_dir / "metrics_by_seed.partial.csv")
        tracker.log(
            f"完成基线模型，seed={seed}",
            stage="baseline_reference_done",
            advance=1,
            context={"seed": seed, "model_name": "xgb_reference_adasyn"},
        )

        tracker.log(f"运行 SCM-v2 参照，seed={seed}", stage="baseline_scm_v2")
        scm_v2_result, scm_v2_aug_size = fit_scm_v2_reference(split_data, seed)
        metrics_rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_scm_v2_best",
                y_test=split_data.y_test,
                proba=scm_v2_result.proba,
                augmented_size=scm_v2_aug_size,
                train_size=scm_v2_result.train_size,
                resampled_train_size=scm_v2_result.resampled_train_size,
            )
        )
        write_partial_table(metrics_rows, tables_dir / "metrics_by_seed.partial.csv")
        tracker.log(
            f"完成 SCM-v2 参照，seed={seed}",
            stage="baseline_scm_v2_done",
            advance=1,
            context={"seed": seed, "model_name": "xgb_scm_v2_best"},
        )

        for config_idx, config in enumerate(configs, start=1):
            model_name = config_to_name(config)
            tracker.log(
                f"运行主线配置 {config_idx}/{len(configs)}，seed={seed}，{model_name}",
                stage="phase1_config_running",
                context={"seed": seed, "config_index": config_idx, "config_total": len(configs), "model_name": model_name},
            )
            train_result, aug_size, audit_df = fit_scm_v3_config(
                split_data,
                seed,
                config,
                use_remote_teacher=args.enable_remote_teacher,
            )
            metrics_rows.append(
                evaluate_model_from_raw(
                    seed=seed,
                    model_name=model_name,
                    y_test=split_data.y_test,
                    proba=train_result.proba,
                    augmented_size=aug_size,
                    train_size=train_result.train_size,
                    resampled_train_size=train_result.resampled_train_size,
                )
            )
            audit_df = audit_df.copy()
            audit_df["seed"] = seed
            audit_df["model_name"] = model_name
            audit_rows.append(audit_df)
            write_partial_table(metrics_rows, tables_dir / "metrics_by_seed.partial.csv")
            tracker.log(
                f"完成主线配置 {config_idx}/{len(configs)}，seed={seed}，增广样本={aug_size}",
                stage="phase1_config_done",
                advance=1,
                context={"seed": seed, "model_name": model_name, "augmented_size": aug_size},
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(tables_dir / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize_model_metrics(metrics_df, metrics_df["model_name"].unique().tolist())
    summary_df.to_csv(tables_dir / "metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    overall_df, minority_df = build_leaderboards(summary_df)
    overall_df.to_csv(tables_dir / "overall_leaderboard.csv", index=False, encoding="utf-8-sig")
    minority_df.to_csv(tables_dir / "minority_leaderboard.csv", index=False, encoding="utf-8-sig")
    plot_boxplot(metrics_df, "balanced_accuracy", figures_dir / "balanced_accuracy_boxplot.png", "SCM_v3 Balanced Accuracy")
    plot_boxplot(metrics_df, "macro_f1", figures_dir / "macro_f1_boxplot.png", "SCM_v3 Macro F1")

    if audit_rows:
        pd.concat(audit_rows, axis=0, ignore_index=True).to_csv(
            tables_dir / "augmentation_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

    best_scm_v3_name = minority_df[minority_df["model_name"].str.startswith("scm_v3_")].iloc[0]["model_name"]
    best_config = next(config for config in configs if config_to_name(config) == best_scm_v3_name)

    if run_tabpfn:
        for seed in seeds:
            split_data = prepare_split_data(args.input, seed)
            tracker.log(f"运行 TabPFN 基线，seed={seed}", stage="phase3_tabpfn_reference")
            ref_proba = run_tabpfn_with_raw(
                split_data.X_train_raw,
                split_data.y_train,
                split_data.X_test_raw,
                PROJECT_DIR,
                seed,
                status_callback=lambda msg, seed=seed: tracker.log(msg, stage="phase3_tabpfn_retry", context={"seed": seed}),
            )
            ref_metrics = evaluate_predictions(split_data.y_test, ref_proba, TARGET_LABELS, [0, 1, 2])
            phase3_rows.append(
                {
                    "seed": seed,
                    "model_name": "tabpfn_reference_v3",
                    "augmented_size": 0,
                    "accuracy": ref_metrics["accuracy"],
                    "balanced_accuracy": ref_metrics["balanced_accuracy"],
                    "macro_f1": ref_metrics["macro_f1"],
                    "weighted_f1": ref_metrics["weighted_f1"],
                    "ovr_roc_auc_macro": ref_metrics["ovr_roc_auc_macro"],
                    "class0_recall": float(ref_metrics["classification_report"]["非确诊"]["recall"]),
                    "class1_recall": float(ref_metrics["classification_report"]["确诊"]["recall"]),
                    "class2_recall": float(ref_metrics["classification_report"]["灰色区域"]["recall"]),
                }
            )
            write_partial_table(phase3_rows, tables_dir / "tabpfn_metrics_by_seed.partial.csv")
            tracker.log(
                f"完成 TabPFN 基线，seed={seed}",
                stage="phase3_tabpfn_reference_done",
                advance=1,
                context={"seed": seed, "model_name": "tabpfn_reference_v3"},
            )

            tracker.log(f"运行 TabPFN + SCM_v3，seed={seed}", stage="phase3_tabpfn_scm_v3")
            hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)
            augmentor = SCMMixV3Augmentor(
                random_state=seed,
                seed_strategy=best_config["seed_strategy"],
                target_classes=best_config["target_classes"],
                treat_mix_prob=best_config["treat_mix_prob"],
                sampler_strength=best_config["sampler_strength"],
                node_sampler=best_config["node_sampler"],
                rule_filter_mode=best_config["rule_filter_mode"],
                teacher_filter_mode=best_config["teacher_filter_mode"],
                counterfactual_mode=best_config["counterfactual_mode"],
                project_dir=PROJECT_DIR,
                use_remote_teacher=args.enable_remote_teacher,
            )
            aug_result = augmentor.generate(split_data.X_train_raw, split_data.y_train, hard_case_map)
            scm_train_X = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
            scm_train_y = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)
            scm_proba = run_tabpfn_with_raw(
                scm_train_X,
                scm_train_y,
                split_data.X_test_raw,
                PROJECT_DIR,
                seed,
                status_callback=lambda msg, seed=seed: tracker.log(msg, stage="phase3_tabpfn_retry", context={"seed": seed}),
            )
            scm_metrics = evaluate_predictions(split_data.y_test, scm_proba, TARGET_LABELS, [0, 1, 2])
            phase3_rows.append(
                {
                    "seed": seed,
                    "model_name": "tabpfn_scm_v3_best",
                    "augmented_size": int(len(aug_result.X_aug)),
                    "accuracy": scm_metrics["accuracy"],
                    "balanced_accuracy": scm_metrics["balanced_accuracy"],
                    "macro_f1": scm_metrics["macro_f1"],
                    "weighted_f1": scm_metrics["weighted_f1"],
                    "ovr_roc_auc_macro": scm_metrics["ovr_roc_auc_macro"],
                    "class0_recall": float(scm_metrics["classification_report"]["非确诊"]["recall"]),
                    "class1_recall": float(scm_metrics["classification_report"]["确诊"]["recall"]),
                    "class2_recall": float(scm_metrics["classification_report"]["灰色区域"]["recall"]),
                }
            )
            write_partial_table(phase3_rows, tables_dir / "tabpfn_metrics_by_seed.partial.csv")
            tracker.log(
                f"完成 TabPFN + SCM_v3，seed={seed}",
                stage="phase3_tabpfn_scm_v3_done",
                advance=1,
                context={"seed": seed, "model_name": "tabpfn_scm_v3_best"},
            )

        phase3_df = pd.DataFrame(phase3_rows)
        phase3_df.to_csv(tables_dir / "tabpfn_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
        phase3_summary = aggregate_metrics(phase3_df, phase3_df["model_name"].unique().tolist())
        phase3_summary.to_csv(tables_dir / "tabpfn_metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    else:
        phase3_summary = pd.DataFrame()

    summary = {
        "seeds": seeds,
        "smoke_test": args.smoke_test,
        "best_only": args.best_only,
        "input_file": str(args.input),
        "config_count": len(configs),
        "best_scm_v3_name": best_scm_v3_name,
        "best_scm_v3_config": best_config,
        "overall_leader": overall_df.iloc[0]["model_name"],
        "minority_leader": minority_df.iloc[0]["model_name"],
        "tabpfn_phase_completed": run_tabpfn and not phase3_summary.empty,
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.log("SCM_v3 实验完成", stage="complete", context=summary)


if __name__ == "__main__":
    main()
