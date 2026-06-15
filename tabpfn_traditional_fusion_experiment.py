from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, prepare_data
from screening_0428_experiment import aggregate_metrics, fit_bundle_by_name, predict_bundle, task_target
from screening_tuning_0428_experiment import (
    ModelConfig,
    binary_configs,
    fit_model as fit_tuning_model,
    predict_model as predict_tuning_model,
    three_configs,
)
from tabpfn_screening_no_post_experiment import TABPFN_DIR, validate_checkpoints
from tabpfn_xgb_fusion_paper_experiment import (
    FEATURE_POLICY,
    TASK,
    TABPFN_VARIANT,
    ProbabilitySet,
    SourceProbabilities,
    add_oof_selected_row,
    apply_isotonic,
    apply_sigmoid,
    apply_strict_screening_policy,
    binary_metrics,
    blend_weights,
    build_composite_ranking,
    build_gap_table,
    choose_threshold,
    evaluated_rows_for_source,
    fit_isotonic,
    fit_sigmoid,
    fit_tabpfn_probability,
    format_seconds,
    logit_prob,
    positive_proba,
    write_feature_policy_files,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "rerun_0428_outputs" / "tabpfn_screening_no_post" / "traditional_fusion_paper"
)
SCREENING_BUNDLE_IDS = ["xgb_binary_scm_v2"]
FIRST_BATCH_TUNING_BINARY = ["xgb_bin_d3_l20_bal", "xgb_bin_d2_l10_bal"]
FIRST_BATCH_TUNING_THREE = ["three_catboost_d3_l50_bal"]
FUSION_CORE_TUNING_BINARY = ["xgb_bin_d3_l20_bal"]
FUSION_CORE_TUNING_THREE = ["three_catboost_d3_l50_bal"]
FUSION_CORE_CONSTRAINED = [
    "xgb_eng_d2_l10",
    "xgb_base_d3_l20",
    "soft_vote_xgb_cat",
]
ALL_EXTRA_TUNING_BINARY = ["xgb_bin_d2_l10_bal"]
ALL_EXTRA_CONSTRAINED = [
    "xgb_eng_d3_l20",
    "cat_eng_d3_l50",
    "soft_vote_xgb_cat_brf",
]
CONSTRAINED_CANDIDATE_IDS = FUSION_CORE_CONSTRAINED + ALL_EXTRA_CONSTRAINED
SMOKE_ESTIMATOR_CAP = 40
PRIMARY_SENSITIVITY_FLOOR = 0.90
XGB_SCM_BASELINE_MODEL = "xgb_binary_scm_v2|default_0p50"


@dataclass(frozen=True)
class TraditionalCandidate:
    model_id: str
    source: str
    task: str
    family: str
    config: Any | None = None
    smoke_reduced: bool = False


@dataclass
class InnerOOFMany:
    probs: dict[str, np.ndarray]
    y: pd.Series
    rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested no-post TabPFN + traditional-model fusion experiment for ARR binary screening."
        )
    )
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / DEFAULT_INPUT_FILE)
    parser.add_argument("--tabpfn-dir", type=Path, default=TABPFN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-set",
        choices=["first_batch", "fusion_core", "constrained", "all"],
        default="fusion_core",
        help="fusion_core uses the V2 core traditional candidates; all adds exploratory extras.",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--max-outer-folds", type=int, default=0)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fit-mode",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
        default="fit_preprocessors",
    )
    parser.add_argument("--skip-stacking", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def reduced_params_for_smoke(params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    reduced = dict(params)
    changed = False
    for key in ("n_estimators", "iterations"):
        if key in reduced:
            old_value = int(reduced[key])
            new_value = min(old_value, SMOKE_ESTIMATOR_CAP)
            reduced[key] = new_value
            changed = changed or new_value != old_value
    return reduced, changed


def maybe_reduce_tuning_config(config: ModelConfig, smoke: bool) -> tuple[ModelConfig, bool]:
    if not smoke:
        return config, False
    params, changed = reduced_params_for_smoke(config.params)
    return replace(config, params=params), changed


def maybe_reduce_constrained_spec(spec: Any, smoke: bool) -> tuple[Any, bool]:
    if not smoke:
        return spec, False
    params, changed = reduced_params_for_smoke(spec.params)
    return replace(spec, params=params), changed


def find_by_id(items: list[Any], model_id: str) -> Any:
    for item in items:
        if item.model_id == model_id:
            return item
    available = ", ".join(sorted(str(item.model_id) for item in items))
    raise KeyError(f"Candidate {model_id} not found. Available: {available}")


def constrained_specs_by_id(smoke: bool) -> dict[str, Any]:
    from screening_constrained_0428_experiment import model_specs

    # Use the full spec list so candidate IDs stay stable; only estimator counts
    # are reduced in smoke mode.
    return {spec.model_id: spec for spec in model_specs(False)}


def build_candidates(args: argparse.Namespace) -> list[TraditionalCandidate]:
    candidates: list[TraditionalCandidate] = []

    def add_screening_bundles(model_ids: list[str]) -> None:
        for model_id in model_ids:
            candidates.append(
                TraditionalCandidate(
                    model_id=model_id,
                    source="screening_bundle",
                    task="binary",
                    family="traditional_scm",
                )
            )

    def add_tuning_binary(model_ids: list[str]) -> None:
        binary_lookup = {config.model_id: config for config in binary_configs(False)}
        for model_id in model_ids:
            config, reduced = maybe_reduce_tuning_config(binary_lookup[model_id], args.smoke_test)
            candidates.append(
                TraditionalCandidate(
                    model_id=model_id,
                    source="tuning",
                    task="binary",
                    family=config.model_family,
                    config=config,
                    smoke_reduced=reduced,
                )
            )

    def add_tuning_three(model_ids: list[str]) -> None:
        three_lookup = {config.model_id: config for config in three_configs(False)}
        for model_id in model_ids:
            config, reduced = maybe_reduce_tuning_config(three_lookup[model_id], args.smoke_test)
            candidates.append(
                TraditionalCandidate(
                    model_id=model_id,
                    source="tuning",
                    task="three_collapsed_binary",
                    family=config.model_family,
                    config=config,
                    smoke_reduced=reduced,
                )
            )

    def add_constrained(model_ids: list[str]) -> None:
        lookup = constrained_specs_by_id(args.smoke_test)
        for model_id in model_ids:
            spec, reduced = maybe_reduce_constrained_spec(find_by_id(list(lookup.values()), model_id), args.smoke_test)
            candidates.append(
                TraditionalCandidate(
                    model_id=model_id,
                    source="constrained",
                    task="binary",
                    family=spec.family,
                    config=spec,
                    smoke_reduced=reduced,
                )
            )

    if args.candidate_set == "first_batch":
        add_screening_bundles(SCREENING_BUNDLE_IDS)
        add_tuning_binary(FIRST_BATCH_TUNING_BINARY)
        add_tuning_three(FIRST_BATCH_TUNING_THREE)
    elif args.candidate_set == "fusion_core":
        add_screening_bundles(SCREENING_BUNDLE_IDS)
        add_tuning_binary(FUSION_CORE_TUNING_BINARY)
        add_tuning_three(FUSION_CORE_TUNING_THREE)
        add_constrained(FUSION_CORE_CONSTRAINED)
    elif args.candidate_set == "constrained":
        add_constrained(CONSTRAINED_CANDIDATE_IDS)
    elif args.candidate_set == "all":
        add_screening_bundles(SCREENING_BUNDLE_IDS)
        add_tuning_binary(FUSION_CORE_TUNING_BINARY + ALL_EXTRA_TUNING_BINARY)
        add_tuning_three(FUSION_CORE_TUNING_THREE)
        add_constrained(FUSION_CORE_CONSTRAINED + ALL_EXTRA_CONSTRAINED)

    if not candidates:
        raise ValueError("No traditional candidates selected.")
    seen: set[str] = set()
    duplicate = [candidate.model_id for candidate in candidates if candidate.model_id in seen or seen.add(candidate.model_id)]
    if duplicate:
        raise ValueError(f"Duplicate candidate IDs: {duplicate}")
    return candidates


def collapsed_three_positive_proba(proba: np.ndarray, model_id: str) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{model_id} expected 3-class probabilities, got shape {arr.shape}.")
    return positive_proba(arr[:, 1] + arr[:, 2])


def metadata_from_fitted(candidate: TraditionalCandidate, fitted: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_model": candidate.model_id,
        "traditional_source": candidate.source,
        "traditional_task": candidate.task,
        "traditional_family": candidate.family,
        "smoke_reduced_traditional_params": bool(candidate.smoke_reduced),
        "train_size": int(fitted.get("train_size", fitted.get("fit_size", 0))),
        "fit_size": int(fitted.get("fit_size", fitted.get("train_size", 0))),
        "resampled_train_size": int(fitted.get("train_size", fitted.get("fit_size", 0))),
        "augmented_size": int(fitted.get("augmented_size", 0)),
    }


def fit_traditional_probability(
    candidate: TraditionalCandidate,
    X_train: pd.DataFrame,
    y_train_three: pd.Series,
    X_eval: pd.DataFrame,
    *,
    seed: int,
    smoke: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_binary = task_target(y_train_three, TASK)
    if candidate.source == "screening_bundle":
        bundle, augmented_size = fit_bundle_by_name(
            candidate.model_id,
            X_train,
            y_binary,
            task=TASK,
            seed=seed,
        )
        proba = positive_proba(predict_bundle(bundle, X_eval))
        metadata = {
            "base_model": candidate.model_id,
            "traditional_source": candidate.source,
            "traditional_task": candidate.task,
            "traditional_family": candidate.family,
            "smoke_reduced_traditional_params": bool(candidate.smoke_reduced),
            "train_size": int(bundle.train_size),
            "fit_size": int(bundle.train_size),
            "resampled_train_size": int(bundle.resampled_train_size),
            "augmented_size": int(augmented_size),
        }
        return proba, metadata

    if candidate.source == "tuning":
        if candidate.config is None:
            raise ValueError(f"{candidate.model_id} has no tuning config.")
        if candidate.task == "binary":
            fitted = fit_tuning_model(X_train, y_binary, candidate.config, seed)
            proba = positive_proba(predict_tuning_model(fitted, X_eval, "binary"))
        elif candidate.task == "three_collapsed_binary":
            y_three = y_train_three.astype(int).reset_index(drop=True)
            fitted = fit_tuning_model(X_train, y_three, candidate.config, seed)
            proba = collapsed_three_positive_proba(
                predict_tuning_model(fitted, X_eval, "three"),
                candidate.model_id,
            )
        else:
            raise ValueError(f"Unsupported tuning task: {candidate.task}")
        return proba, metadata_from_fitted(candidate, fitted)

    if candidate.source == "constrained":
        if candidate.config is None:
            raise ValueError(f"{candidate.model_id} has no constrained spec.")
        from screening_constrained_0428_experiment import (
            fit_model as fit_constrained_model,
            predict_model as predict_constrained_model,
        )

        fitted = fit_constrained_model(X_train, y_binary, candidate.config, seed, smoke)
        proba = positive_proba(predict_constrained_model(fitted, X_eval))
        return proba, metadata_from_fitted(candidate, fitted)

    raise ValueError(f"Unknown candidate source: {candidate.source}")


def run_inner_oof(
    X_outer_train_base: pd.DataFrame,
    y_outer_train_three: pd.Series,
    *,
    inner_splits: int,
    outer_fold_index: int,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
    candidates: list[TraditionalCandidate],
) -> InnerOOFMany:
    y_outer_binary = task_target(y_outer_train_three, TASK)
    oof_probs: dict[str, np.ndarray] = {
        TABPFN_VARIANT: np.zeros(len(y_outer_binary), dtype=float)
    }
    for candidate in candidates:
        oof_probs[candidate.model_id] = np.zeros(len(y_outer_binary), dtype=float)

    rows: list[dict[str, Any]] = []
    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=args.random_state + outer_fold_index * 1009,
    )
    for inner_index, (fit_idx, valid_idx) in enumerate(inner_cv.split(X_outer_train_base, y_outer_binary), start=1):
        X_fit_base = X_outer_train_base.iloc[fit_idx].reset_index(drop=True)
        X_valid_base = X_outer_train_base.iloc[valid_idx].reset_index(drop=True)
        y_fit_three = y_outer_train_three.iloc[fit_idx].reset_index(drop=True)
        y_fit_binary = task_target(y_fit_three, TASK)
        y_valid = y_outer_binary.iloc[valid_idx].reset_index(drop=True)
        view = apply_strict_screening_policy(X_fit_base, X_valid_base)
        seed_base = args.random_state + outer_fold_index * 10000 + inner_index * 101

        tab_prob, tab_meta = fit_tabpfn_probability(
            view.X_train,
            y_fit_binary,
            view.X_eval,
            seed=seed_base + 23,
            checkpoint_paths=checkpoint_paths,
            n_estimators=args.n_estimators,
            device=args.device,
            fit_mode=args.fit_mode,
        )
        oof_probs[TABPFN_VARIANT][valid_idx] = tab_prob
        m = binary_metrics(y_valid, tab_prob, 0.5)
        rows.append(
            {
                "outer_fold_index": outer_fold_index,
                "inner_fold_index": inner_index,
                "model_name": TABPFN_VARIANT,
                "candidate_family": "tabpfn_single",
                "train_size": int(len(y_fit_binary)),
                "valid_size": int(len(y_valid)),
                **tab_meta,
                **m,
            }
        )

        for cand_idx, candidate in enumerate(candidates, start=1):
            prob, meta = fit_traditional_probability(
                candidate,
                view.X_train,
                y_fit_three,
                view.X_eval,
                seed=seed_base + 100 + cand_idx * 17,
                smoke=args.smoke_test,
            )
            oof_probs[candidate.model_id][valid_idx] = prob
            m = binary_metrics(y_valid, prob, 0.5)
            rows.append(
                {
                    "outer_fold_index": outer_fold_index,
                    "inner_fold_index": inner_index,
                    "model_name": candidate.model_id,
                    "candidate_family": "traditional_single",
                    "train_size": int(len(y_fit_binary)),
                    "valid_size": int(len(y_valid)),
                    **meta,
                    **m,
                }
            )

    return InnerOOFMany(probs=oof_probs, y=y_outer_binary, rows=rows)


def fit_outer_probabilities(
    X_outer_train_base: pd.DataFrame,
    y_outer_train_three: pd.Series,
    X_outer_test_base: pd.DataFrame,
    *,
    outer_fold_index: int,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
    candidates: list[TraditionalCandidate],
) -> tuple[dict[str, ProbabilitySet], list[str], list[str]]:
    y_outer_binary = task_target(y_outer_train_three, TASK)
    view = apply_strict_screening_policy(X_outer_train_base, X_outer_test_base)
    X_eval_both = pd.concat([view.X_train, view.X_eval], axis=0, ignore_index=True)
    train_len = len(y_outer_binary)
    seed_base = args.random_state + outer_fold_index * 20000

    prob_sets: dict[str, ProbabilitySet] = {}
    tab_both, tab_meta = fit_tabpfn_probability(
        view.X_train,
        y_outer_binary,
        X_eval_both,
        seed=seed_base + 401,
        checkpoint_paths=checkpoint_paths,
        n_estimators=args.n_estimators,
        device=args.device,
        fit_mode=args.fit_mode,
    )
    prob_sets[TABPFN_VARIANT] = ProbabilitySet(
        train=tab_both[:train_len],
        test=tab_both[train_len:],
        metadata=tab_meta,
    )

    for cand_idx, candidate in enumerate(candidates, start=1):
        prob_both, meta = fit_traditional_probability(
            candidate,
            view.X_train,
            y_outer_train_three,
            X_eval_both,
            seed=seed_base + 500 + cand_idx * 29,
            smoke=args.smoke_test,
        )
        prob_sets[candidate.model_id] = ProbabilitySet(
            train=prob_both[:train_len],
            test=prob_both[train_len:],
            metadata=meta,
        )

    return prob_sets, view.included_columns, view.dropped_columns


def model_selection_score(y_oof: pd.Series, prob: np.ndarray) -> float:
    return float(choose_threshold(y_oof, prob)["selection_score"])


def top_traditional_candidates(
    y_oof: pd.Series,
    oof_probs: dict[str, np.ndarray],
    candidates: list[TraditionalCandidate],
    *,
    top_k: int = 2,
) -> list[TraditionalCandidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            model_selection_score(y_oof, oof_probs[candidate.model_id]),
            candidate.model_id,
        ),
        reverse=True,
    )
    return ranked[:top_k]


def inverse_logit(values: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return positive_proba(1.0 / (1.0 + np.exp(-z)))


def logit_blend_prob(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    right_weight = 1.0 - left_weight
    return inverse_logit(left_weight * logit_prob(left) + right_weight * logit_prob(right))


def rank_reference(pos_prob: np.ndarray) -> np.ndarray:
    return np.sort(positive_proba(pos_prob))


def apply_rank_percentile(reference: np.ndarray, pos_prob: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    if ref.size == 0:
        return positive_proba(pos_prob)
    ranks = np.searchsorted(ref, positive_proba(pos_prob), side="right") / float(ref.size)
    return positive_proba(ranks)


def cascade_score(gate_prob: np.ndarray, rescue_prob: np.ndarray, gate_threshold: float, rescue_threshold: float) -> np.ndarray:
    gate = positive_proba(gate_prob)
    rescue = positive_proba(rescue_prob)
    pred_positive = (gate >= gate_threshold) | (rescue >= rescue_threshold)
    high_score = 0.75 + 0.24 * np.maximum(gate, rescue)
    low_score = 0.01 + 0.24 * np.minimum(gate, rescue)
    return positive_proba(np.where(pred_positive, high_score, low_score))


def choose_cascade_thresholds(
    y_oof: pd.Series,
    gate_prob: np.ndarray,
    rescue_prob: np.ndarray,
    *,
    sensitivity_floor: float = PRIMARY_SENSITIVITY_FLOOR,
) -> dict[str, float]:
    gate_grid = np.unique(np.round(np.quantile(positive_proba(gate_prob), np.linspace(0.05, 0.95, 37)), 4))
    rescue_grid = np.unique(np.round(np.quantile(positive_proba(rescue_prob), np.linspace(0.05, 0.95, 37)), 4))
    best: dict[str, float] | None = None
    best_relaxed: dict[str, float] | None = None
    for gate_threshold in gate_grid:
        for rescue_threshold in rescue_grid:
            prob = cascade_score(gate_prob, rescue_prob, float(gate_threshold), float(rescue_threshold))
            metrics = binary_metrics(y_oof, prob, 0.5)
            metrics["selection_score"] = (
                0.30 * metrics["balanced_accuracy"]
                + 0.25 * metrics["specificity"]
                + 0.20 * metrics["macro_f1"]
                + 0.15 * metrics["sensitivity"]
                + 0.10 * metrics["accuracy"]
            )
            metrics["gate_threshold"] = float(gate_threshold)
            metrics["rescue_threshold"] = float(rescue_threshold)
            key = (
                metrics["selection_score"],
                metrics["specificity"],
                metrics["balanced_accuracy"],
                metrics["sensitivity"],
            )
            if best_relaxed is None or key > (
                best_relaxed["selection_score"],
                best_relaxed["specificity"],
                best_relaxed["balanced_accuracy"],
                best_relaxed["sensitivity"],
            ):
                best_relaxed = metrics
            if metrics["sensitivity"] < sensitivity_floor:
                continue
            if best is None or key > (
                best["selection_score"],
                best["specificity"],
                best["balanced_accuracy"],
                best["sensitivity"],
            ):
                best = metrics
    chosen = best if best is not None else best_relaxed
    assert chosen is not None
    chosen["met_sensitivity_floor"] = bool(chosen["sensitivity"] >= sensitivity_floor)
    return chosen


def multi_stack_features(prob_dict: dict[str, np.ndarray], base_names: list[str]) -> np.ndarray:
    probs = [positive_proba(prob_dict[name]) for name in base_names]
    logits = [logit_prob(prob) for prob in probs]
    matrix = np.column_stack(logits)
    prob_matrix = np.column_stack(probs)
    mean_prob = prob_matrix.mean(axis=1)
    min_prob = prob_matrix.min(axis=1)
    max_prob = prob_matrix.max(axis=1)
    spread = prob_matrix.max(axis=1) - prob_matrix.min(axis=1)
    std = prob_matrix.std(axis=1)
    tab_idx = base_names.index(TABPFN_VARIANT) if TABPFN_VARIANT in base_names else 0
    tab_disagreement = np.mean(np.abs(prob_matrix - prob_matrix[:, [tab_idx]]), axis=1)
    return np.column_stack(
        [matrix, logit_prob(mean_prob), logit_prob(min_prob), logit_prob(max_prob), mean_prob, min_prob, max_prob, spread, std, tab_disagreement]
    )


def fit_multi_stacking_model(
    y_oof: pd.Series,
    oof_probs: dict[str, np.ndarray],
    base_names: list[str],
    *,
    seed: int,
) -> tuple[LogisticRegression, float, pd.DataFrame]:
    X_meta = multi_stack_features(oof_probs, base_names)
    y_arr = y_oof.to_numpy(dtype=int)
    class_counts = np.bincount(y_arr, minlength=2)
    n_splits = int(min(5, class_counts.min()))
    candidate_cs = [0.05, 0.1, 1.0, 10.0]
    rows: list[dict[str, Any]] = []
    if n_splits < 2:
        best_c = 1.0
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best_c = 1.0
        best_score = -np.inf
        for c_value in candidate_cs:
            fold_scores = []
            for train_idx, valid_idx in cv.split(X_meta, y_arr):
                model = LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                )
                model.fit(X_meta[train_idx], y_arr[train_idx])
                p_valid = positive_proba(model.predict_proba(X_meta[valid_idx])[:, 1])
                chosen = choose_threshold(pd.Series(y_arr[valid_idx]), p_valid)
                fold_scores.append(float(chosen["selection_score"]))
            mean_score = float(np.mean(fold_scores))
            rows.append(
                {
                    "C": c_value,
                    "selection_score_mean": mean_score,
                    "stack_base_names": ",".join(base_names),
                }
            )
            if mean_score > best_score:
                best_score = mean_score
                best_c = c_value
    final_model = LogisticRegression(
        C=best_c,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )
    final_model.fit(X_meta, y_arr)
    return final_model, best_c, pd.DataFrame(rows)


def calibrated_probability_sets(
    y_oof: pd.Series,
    oof_prob: np.ndarray,
    outer_prob: ProbabilitySet,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    sigmoid_model = fit_sigmoid(y_oof, oof_prob)
    isotonic_model = fit_isotonic(y_oof, oof_prob)
    return {
        "sigmoid": (
            apply_sigmoid(sigmoid_model, oof_prob),
            apply_sigmoid(sigmoid_model, outer_prob.train),
            apply_sigmoid(sigmoid_model, outer_prob.test),
        ),
        "isotonic": (
            apply_isotonic(isotonic_model, oof_prob),
            apply_isotonic(isotonic_model, outer_prob.train),
            apply_isotonic(isotonic_model, outer_prob.test),
        ),
    }


def make_source_probabilities(
    *,
    y_oof: pd.Series,
    oof: InnerOOFMany,
    outer_probs: dict[str, ProbabilitySet],
    candidates: list[TraditionalCandidate],
    blend_values: list[float],
    skip_stacking: bool,
    outer_fold_index: int,
    args: argparse.Namespace,
) -> tuple[list[SourceProbabilities], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources: list[SourceProbabilities] = []
    base_names = [TABPFN_VARIANT] + [candidate.model_id for candidate in candidates]
    tab_rank_reference = rank_reference(oof.probs[TABPFN_VARIANT])

    for name in base_names:
        outer = outer_probs[name]
        is_tabpfn = name == TABPFN_VARIANT
        sources.append(
            SourceProbabilities(
                source_name=name,
                candidate_family="single_model" if not is_tabpfn else "tabpfn_single",
                oof=oof.probs[name],
                train=outer.train,
                test=outer.test,
                metadata={
                    **outer.metadata,
                    "calibration": "none",
                    "base_model_count": 1,
                },
            )
        )

    for candidate in candidates:
        trad_name = candidate.model_id
        trad_outer = outer_probs[trad_name]
        trad_rank_reference = rank_reference(oof.probs[trad_name])
        tab_rank_oof = apply_rank_percentile(tab_rank_reference, oof.probs[TABPFN_VARIANT])
        tab_rank_train = apply_rank_percentile(tab_rank_reference, outer_probs[TABPFN_VARIANT].train)
        tab_rank_test = apply_rank_percentile(tab_rank_reference, outer_probs[TABPFN_VARIANT].test)
        trad_rank_oof = apply_rank_percentile(trad_rank_reference, oof.probs[trad_name])
        trad_rank_train = apply_rank_percentile(trad_rank_reference, trad_outer.train)
        trad_rank_test = apply_rank_percentile(trad_rank_reference, trad_outer.test)
        for tab_weight in blend_values:
            if math.isclose(tab_weight, 0.0) or math.isclose(tab_weight, 1.0):
                continue
            trad_weight = 1.0 - tab_weight
            sources.append(
                SourceProbabilities(
                    source_name=f"blend_raw_tabpfn_{tab_weight:.2f}_{trad_name}_{trad_weight:.2f}",
                    candidate_family="fixed_probability_blend",
                    oof=tab_weight * oof.probs[TABPFN_VARIANT] + trad_weight * oof.probs[trad_name],
                    train=tab_weight * outer_probs[TABPFN_VARIANT].train + trad_weight * trad_outer.train,
                    test=tab_weight * outer_probs[TABPFN_VARIANT].test + trad_weight * trad_outer.test,
                    metadata={
                        "calibration": "none",
                        "base_model": trad_name,
                        "tabpfn_weight": float(tab_weight),
                        "traditional_weight": float(trad_weight),
                        "base_model_count": 2,
                    },
                )
            )
            sources.append(
                SourceProbabilities(
                    source_name=f"blend_logit_tabpfn_{tab_weight:.2f}_{trad_name}_{trad_weight:.2f}",
                    candidate_family="logit_blend",
                    oof=logit_blend_prob(oof.probs[TABPFN_VARIANT], oof.probs[trad_name], tab_weight),
                    train=logit_blend_prob(outer_probs[TABPFN_VARIANT].train, trad_outer.train, tab_weight),
                    test=logit_blend_prob(outer_probs[TABPFN_VARIANT].test, trad_outer.test, tab_weight),
                    metadata={
                        "calibration": "logit_space",
                        "base_model": trad_name,
                        "tabpfn_weight": float(tab_weight),
                        "traditional_weight": float(trad_weight),
                        "base_model_count": 2,
                    },
                )
            )
            sources.append(
                SourceProbabilities(
                    source_name=f"blend_rank_tabpfn_{tab_weight:.2f}_{trad_name}_{trad_weight:.2f}",
                    candidate_family="rank_percentile_fusion",
                    oof=tab_weight * tab_rank_oof + trad_weight * trad_rank_oof,
                    train=tab_weight * tab_rank_train + trad_weight * trad_rank_train,
                    test=tab_weight * tab_rank_test + trad_weight * trad_rank_test,
                    metadata={
                        "calibration": "rank_percentile_oof",
                        "base_model": trad_name,
                        "tabpfn_weight": float(tab_weight),
                        "traditional_weight": float(trad_weight),
                        "base_model_count": 2,
                    },
                )
            )

    calibrated: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in base_names:
        for method, values in calibrated_probability_sets(y_oof, oof.probs[name], outer_probs[name]).items():
            calibrated[(name, method)] = values
            cal_oof, cal_train, cal_test = values
            sources.append(
                SourceProbabilities(
                    source_name=f"{name}_{method}",
                    candidate_family=f"{method}_calibrated_single",
                    oof=cal_oof,
                    train=cal_train,
                    test=cal_test,
                    metadata={
                        **outer_probs[name].metadata,
                        "calibration": method,
                        "base_model": name,
                        "base_model_count": 1,
                    },
                )
            )

    for candidate in candidates:
        trad_name = candidate.model_id
        for method in ("sigmoid", "isotonic"):
            tab_oof, tab_train, tab_test = calibrated[(TABPFN_VARIANT, method)]
            trad_oof, trad_train, trad_test = calibrated[(trad_name, method)]
            for tab_weight in blend_values:
                if math.isclose(tab_weight, 0.0) or math.isclose(tab_weight, 1.0):
                    continue
                trad_weight = 1.0 - tab_weight
                sources.append(
                    SourceProbabilities(
                        source_name=f"blend_{method}_tabpfn_{tab_weight:.2f}_{trad_name}_{trad_weight:.2f}",
                        candidate_family=f"{method}_calibrated_blend",
                        oof=tab_weight * tab_oof + trad_weight * trad_oof,
                        train=tab_weight * tab_train + trad_weight * trad_train,
                        test=tab_weight * tab_test + trad_weight * trad_test,
                        metadata={
                            "calibration": method,
                            "base_model": trad_name,
                            "tabpfn_weight": float(tab_weight),
                            "traditional_weight": float(trad_weight),
                            "base_model_count": 2,
                        },
                    )
                )

    top_trad = top_traditional_candidates(y_oof, oof.probs, candidates, top_k=min(2, len(candidates)))
    selection_rows = [
        {
            "outer_fold_index": outer_fold_index,
            "rank": rank,
            "selected_traditional_model": candidate.model_id,
            "oof_selection_score": model_selection_score(y_oof, oof.probs[candidate.model_id]),
        }
        for rank, candidate in enumerate(top_trad, start=1)
    ]
    if top_trad:
        selected_names = [TABPFN_VARIANT] + [candidate.model_id for candidate in top_trad]
        n_models = len(selected_names)
        sources.append(
            SourceProbabilities(
                source_name="soft_vote_tabpfn_top2_traditional",
                candidate_family="multi_model_soft_voting",
                oof=np.mean([oof.probs[name] for name in selected_names], axis=0),
                train=np.mean([outer_probs[name].train for name in selected_names], axis=0),
                test=np.mean([outer_probs[name].test for name in selected_names], axis=0),
                metadata={
                    "calibration": "none",
                    "base_model": ",".join(selected_names),
                    "base_model_count": int(n_models),
                    "selected_traditional_models": ",".join(candidate.model_id for candidate in top_trad),
                    "tabpfn_weight": 1.0 / n_models,
                    "traditional_weight": (n_models - 1.0) / n_models,
                },
            )
        )

    gate_rows: list[dict[str, Any]] = []
    gate_name = "xgb_base_d3_l20"
    rescue_names = [name for name in [TABPFN_VARIANT, "soft_vote_xgb_cat", "xgb_eng_d2_l10"] if name in outer_probs]
    if gate_name in outer_probs and len(rescue_names) >= 2:
        rescue_oof = np.mean([oof.probs[name] for name in rescue_names], axis=0)
        rescue_train = np.mean([outer_probs[name].train for name in rescue_names], axis=0)
        rescue_test = np.mean([outer_probs[name].test for name in rescue_names], axis=0)
        cascade_info = choose_cascade_thresholds(
            y_oof,
            oof.probs[gate_name],
            rescue_oof,
            sensitivity_floor=PRIMARY_SENSITIVITY_FLOOR,
        )
        gate_threshold = float(cascade_info["gate_threshold"])
        rescue_threshold = float(cascade_info["rescue_threshold"])
        sources.append(
            SourceProbabilities(
                source_name="cascade_specificity_xgb_base_rescue_tabpfn_softvote_xgbeng",
                candidate_family="specificity_rescue_cascade",
                oof=cascade_score(oof.probs[gate_name], rescue_oof, gate_threshold, rescue_threshold),
                train=cascade_score(outer_probs[gate_name].train, rescue_train, gate_threshold, rescue_threshold),
                test=cascade_score(outer_probs[gate_name].test, rescue_test, gate_threshold, rescue_threshold),
                metadata={
                    "calibration": "cascade_oof_thresholds",
                    "gate_model": gate_name,
                    "rescue_models": ",".join(rescue_names),
                    "gate_threshold": gate_threshold,
                    "rescue_threshold": rescue_threshold,
                    "cascade_oof_sensitivity_floor": float(PRIMARY_SENSITIVITY_FLOOR),
                    "cascade_met_sensitivity_floor": bool(cascade_info["met_sensitivity_floor"]),
                    "base_model": gate_name,
                    "base_model_count": int(1 + len(rescue_names)),
                },
            )
        )
        gate_rows.append(
            {
                "outer_fold_index": outer_fold_index,
                "source_name": "cascade_specificity_xgb_base_rescue_tabpfn_softvote_xgbeng",
                "gate_model": gate_name,
                "rescue_models": ",".join(rescue_names),
                "gate_threshold": gate_threshold,
                "rescue_threshold": rescue_threshold,
                "sensitivity_floor": float(PRIMARY_SENSITIVITY_FLOOR),
                **{f"oof_{key}": value for key, value in cascade_info.items()},
            }
        )

    stack_rows = pd.DataFrame()
    if not skip_stacking:
        stack_model, best_c, stack_rows = fit_multi_stacking_model(
            y_oof,
            oof.probs,
            base_names,
            seed=args.random_state + outer_fold_index * 3001,
        )
        stack_oof = positive_proba(
            stack_model.predict_proba(multi_stack_features(oof.probs, base_names))[:, 1]
        )
        train_input = {name: outer_probs[name].train for name in base_names}
        test_input = {name: outer_probs[name].test for name in base_names}
        stack_train = positive_proba(
            stack_model.predict_proba(multi_stack_features(train_input, base_names))[:, 1]
        )
        stack_test = positive_proba(
            stack_model.predict_proba(multi_stack_features(test_input, base_names))[:, 1]
        )
        sources.append(
            SourceProbabilities(
                source_name="stack_logreg_all_base",
                candidate_family="constrained_stacking",
                oof=stack_oof,
                train=stack_train,
                test=stack_test,
                metadata={
                    "stacking_C": float(best_c),
                    "calibration": "stacking_logistic",
                    "base_model": ",".join(base_names),
                    "base_model_count": int(len(base_names)),
                },
            )
        )
        if not stack_rows.empty:
            stack_rows.insert(0, "outer_fold_index", outer_fold_index)
    return sources, stack_rows, pd.DataFrame(selection_rows), pd.DataFrame(gate_rows)


def run_outer_fold(
    prepared_X: pd.DataFrame,
    prepared_y: pd.Series,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    outer_fold_index: int,
    repeat_index: int,
    split_index: int,
    args: argparse.Namespace,
    checkpoint_paths: dict[str, str],
    candidates: list[TraditionalCandidate],
    blend_values: list[float],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    X_train_base = prepared_X.iloc[train_idx].reset_index(drop=True)
    X_test_base = prepared_X.iloc[test_idx].reset_index(drop=True)
    y_train_three = prepared_y.iloc[train_idx].reset_index(drop=True)
    y_test = task_target(prepared_y.iloc[test_idx].reset_index(drop=True), TASK)
    y_train = task_target(y_train_three, TASK)

    oof = run_inner_oof(
        X_train_base,
        y_train_three,
        inner_splits=args.inner_splits,
        outer_fold_index=outer_fold_index,
        args=args,
        checkpoint_paths=checkpoint_paths,
        candidates=candidates,
    )
    outer_probs, included_columns, dropped_columns = fit_outer_probabilities(
        X_train_base,
        y_train_three,
        X_test_base,
        outer_fold_index=outer_fold_index,
        args=args,
        checkpoint_paths=checkpoint_paths,
        candidates=candidates,
    )
    sources, stack_rows, top2_rows, gate_rows = make_source_probabilities(
        y_oof=oof.y,
        oof=oof,
        outer_probs=outer_probs,
        candidates=candidates,
        blend_values=blend_values,
        skip_stacking=args.skip_stacking,
        outer_fold_index=outer_fold_index,
        args=args,
    )
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for source in sources:
        source_rows, source_selection_rows = evaluated_rows_for_source(
            source,
            y_oof=oof.y,
            y_train=y_train,
            y_test=y_test,
            outer_fold_index=outer_fold_index,
            repeat_index=repeat_index,
            split_index=split_index,
        )
        rows.extend(source_rows)
        selection_rows.extend(source_selection_rows)
    add_oof_selected_row(rows, selection_rows, outer_fold_index=outer_fold_index)
    audit = {
        "outer_fold_index": outer_fold_index,
        "repeat_index": repeat_index,
        "split_index": split_index,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_class0": int((y_train == 0).sum()),
        "train_class1": int((y_train == 1).sum()),
        "test_class0": int((y_test == 0).sum()),
        "test_class1": int((y_test == 1).sum()),
        "inner_oof_rows": int(len(oof.rows)),
        "source_count": int(len(sources)),
        "candidate_row_count": int(len(rows)),
        "included_feature_count": int(len(included_columns)),
        "dropped_feature_count": int(len(dropped_columns)),
        "dropped_columns": ",".join(dropped_columns),
    }
    return rows, selection_rows, oof.rows, stack_rows, top2_rows, gate_rows, audit


def write_df(path: Path, rows_or_df: list[dict[str, Any]] | pd.DataFrame) -> None:
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_partial(
    tables_dir: Path,
    metrics_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    inner_rows: list[dict[str, Any]],
    stack_frames: list[pd.DataFrame],
    top2_frames: list[pd.DataFrame],
    gate_frames: list[pd.DataFrame],
) -> None:
    if metrics_rows:
        write_df(tables_dir / "traditional_fusion_metrics_by_outer_fold.partial.csv", metrics_rows)
    if selection_rows:
        write_df(tables_dir / "traditional_fusion_oof_candidates_by_fold.partial.csv", selection_rows)
    if inner_rows:
        write_df(tables_dir / "traditional_fusion_inner_oof_base_metrics.partial.csv", inner_rows)
    if stack_frames:
        write_df(
            tables_dir / "traditional_fusion_stacking_selection.partial.csv",
            pd.concat(stack_frames, ignore_index=True),
        )
    if top2_frames:
        write_df(
            tables_dir / "traditional_fusion_top2_traditional_selection.partial.csv",
            pd.concat(top2_frames, ignore_index=True),
        )
    if gate_frames:
        write_df(
            tables_dir / "traditional_fusion_gate_selection_by_fold.partial.csv",
            pd.concat(gate_frames, ignore_index=True),
        )


def candidate_manifest(candidates: list[TraditionalCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        params = getattr(candidate.config, "params", {}) if candidate.config is not None else {}
        rows.append(
            {
                "model_id": candidate.model_id,
                "source": candidate.source,
                "task": candidate.task,
                "family": candidate.family,
                "smoke_reduced": bool(candidate.smoke_reduced),
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def sensitivity_tradeoff_flags(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    out = ranking.copy()
    out["sensitivity_below_0_90"] = pd.to_numeric(out.get("sensitivity_mean"), errors="coerce") < 0.90
    out["fusion_non_primary_tradeoff"] = (
        out["candidate_family"].astype(str).str.contains("blend|voting|stacking|cascade|rank|logit", regex=True)
        & out["sensitivity_below_0_90"].fillna(True)
    )
    return out


def attach_eligibility_flags(
    ranking: pd.DataFrame,
    fold_coverage: pd.DataFrame,
    *,
    expected_outer: int,
) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    out = ranking.copy()
    if not fold_coverage.empty:
        out = out.merge(
            fold_coverage,
            on=["model_name", "candidate_family"],
            how="left",
        )
    if "completed_outer_folds" not in out.columns:
        out["completed_outer_folds"] = 0
    if "expected_outer_folds" not in out.columns:
        out["expected_outer_folds"] = expected_outer
    out["completed_outer_folds"] = pd.to_numeric(out["completed_outer_folds"], errors="coerce").fillna(0).astype(int)
    out["expected_outer_folds"] = pd.to_numeric(out["expected_outer_folds"], errors="coerce").fillna(expected_outer).astype(int)
    out["full_coverage"] = out["completed_outer_folds"] == out["expected_outer_folds"]
    out["sensitivity_meets_floor"] = pd.to_numeric(out.get("sensitivity_mean"), errors="coerce") >= PRIMARY_SENSITIVITY_FLOOR
    xgb_rows = out[out["model_name"] == XGB_SCM_BASELINE_MODEL]
    xgb_class0 = (
        float(pd.to_numeric(xgb_rows["class0_recall_mean"], errors="coerce").max())
        if not xgb_rows.empty and "class0_recall_mean" in xgb_rows.columns
        else float("nan")
    )
    out["xgb_scm_default_class0_recall"] = xgb_class0
    if math.isfinite(xgb_class0) and "class0_recall_mean" in out.columns:
        out["class0_beats_xgb_scm_default"] = pd.to_numeric(out["class0_recall_mean"], errors="coerce") > xgb_class0
    else:
        out["class0_beats_xgb_scm_default"] = False
    out["primary_eligible"] = (
        out["full_coverage"].fillna(False)
        & out["sensitivity_meets_floor"].fillna(False)
        & out["class0_beats_xgb_scm_default"].fillna(False)
    )
    reasons: list[str] = []
    for _, row in out.iterrows():
        row_reasons = []
        if not bool(row.get("full_coverage", False)):
            row_reasons.append("incomplete_fold_coverage")
        if not bool(row.get("sensitivity_meets_floor", False)):
            row_reasons.append("sensitivity_below_0_90")
        if not bool(row.get("class0_beats_xgb_scm_default", False)):
            row_reasons.append("class0_not_above_xgb_scm_default")
        reasons.append(";".join(row_reasons) if row_reasons else "eligible")
    out["primary_ineligibility_reason"] = reasons
    return out


def base_prediction_audit(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    base_families = {"tabpfn_single", "single_model"}
    cols = [
        "outer_fold_index",
        "repeat_index",
        "split_index",
        "model_name",
        "source_name",
        "candidate_family",
        "threshold_objective",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "class0_recall",
        "sensitivity",
        "specificity",
        "ovr_roc_auc_macro",
        "brier_score",
        "ece",
        "log_loss",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    existing = [col for col in cols if col in metrics_df.columns]
    return metrics_df[metrics_df["candidate_family"].isin(base_families)][existing].copy()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    checkpoint_paths = validate_checkpoints(args.tabpfn_dir, [TABPFN_VARIANT])
    prepared = prepare_data(args.input)
    feature_audit = write_feature_policy_files(tables_dir, list(prepared.X.columns))
    blend_values = blend_weights(args.blend_step)
    y_binary = task_target(prepared.y, TASK)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )
    max_outer = args.max_outer_folds if args.max_outer_folds and args.max_outer_folds > 0 else None

    write_df(tables_dir / "traditional_fusion_candidate_manifest.csv", candidate_manifest(candidates))

    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    stack_frames: list[pd.DataFrame] = []
    top2_frames: list[pd.DataFrame] = []
    gate_frames: list[pd.DataFrame] = []
    prediction_audit_rows: list[dict[str, Any]] = []

    for outer_fold_index, (train_idx, test_idx) in enumerate(outer_cv.split(prepared.X, y_binary), start=1):
        if max_outer is not None and outer_fold_index > max_outer:
            break
        repeat_index = (outer_fold_index - 1) // args.outer_splits + 1
        split_index = (outer_fold_index - 1) % args.outer_splits + 1
        print(
            f"[{format_seconds(time.perf_counter() - start_time)}] "
            f"outer_fold={outer_fold_index} repeat={repeat_index} split={split_index} "
            f"candidate_set={args.candidate_set}",
            flush=True,
        )
        fold_rows, fold_selection_rows, fold_inner_rows, stack_rows, top2_rows, gate_rows, audit = run_outer_fold(
            prepared.X,
            prepared.y,
            train_idx=train_idx,
            test_idx=test_idx,
            outer_fold_index=outer_fold_index,
            repeat_index=repeat_index,
            split_index=split_index,
            args=args,
            checkpoint_paths=checkpoint_paths,
            candidates=candidates,
            blend_values=blend_values,
        )
        metrics_rows.extend(fold_rows)
        selection_rows.extend(fold_selection_rows)
        inner_rows.extend(fold_inner_rows)
        if not stack_rows.empty:
            stack_frames.append(stack_rows)
        if not top2_rows.empty:
            top2_frames.append(top2_rows)
        if not gate_rows.empty:
            gate_frames.append(gate_rows)
        selected = [row for row in fold_selection_rows if row.get("model_name") == "oof_selected_composite"]
        audit["selected_model_name"] = selected[0].get("selected_model_name") if selected else None
        prediction_audit_rows.append(audit)
        save_partial(tables_dir, metrics_rows, selection_rows, inner_rows, stack_frames, top2_frames, gate_frames)

    metrics_df = pd.DataFrame(metrics_rows)
    selection_df = pd.DataFrame(selection_rows)
    inner_df = pd.DataFrame(inner_rows)
    prediction_audit_df = pd.DataFrame(prediction_audit_rows)

    write_df(tables_dir / "traditional_fusion_metrics_by_outer_fold.csv", metrics_df)
    write_df(tables_dir / "traditional_fusion_oof_candidates_by_fold.csv", selection_df)
    if not selection_df.empty:
        selected_df = selection_df[selection_df["model_name"] == "oof_selected_composite"].copy()
        write_df(tables_dir / "traditional_fusion_oof_selection_by_fold.csv", selected_df)
    else:
        write_df(tables_dir / "traditional_fusion_oof_selection_by_fold.csv", pd.DataFrame())
    write_df(tables_dir / "traditional_fusion_inner_oof_base_metrics.csv", inner_df)
    write_df(tables_dir / "traditional_fusion_prediction_audit.csv", prediction_audit_df)
    write_df(tables_dir / "traditional_fusion_base_prediction_audit.csv", base_prediction_audit(metrics_df))
    if stack_frames:
        write_df(
            tables_dir / "traditional_fusion_stacking_selection.csv",
            pd.concat(stack_frames, ignore_index=True),
        )
    else:
        write_df(tables_dir / "traditional_fusion_stacking_selection.csv", pd.DataFrame())
    if top2_frames:
        write_df(
            tables_dir / "traditional_fusion_top2_traditional_selection.csv",
            pd.concat(top2_frames, ignore_index=True),
        )
    else:
        write_df(tables_dir / "traditional_fusion_top2_traditional_selection.csv", pd.DataFrame())
    if gate_frames:
        write_df(
            tables_dir / "traditional_fusion_gate_selection_by_fold.csv",
            pd.concat(gate_frames, ignore_index=True),
        )
    else:
        write_df(tables_dir / "traditional_fusion_gate_selection_by_fold.csv", pd.DataFrame())

    summary = aggregate_metrics(
        metrics_df,
        ["task", "feature_policy", "model_name", "candidate_family", "threshold_objective"],
        seed_base=91000,
    )
    if not summary.empty:
        summary = summary.sort_values(
            ["balanced_accuracy_mean", "accuracy_mean", "macro_f1_mean", "class0_recall_mean"],
            ascending=[False, False, False, False],
        )
    write_df(tables_dir / "traditional_fusion_summary_mean_std.csv", summary)

    completed_outer = int(metrics_df["outer_fold_index"].nunique()) if not metrics_df.empty else 0
    expected_outer = (
        int(args.max_outer_folds)
        if args.max_outer_folds and args.max_outer_folds > 0
        else int(args.outer_splits * args.outer_repeats)
    )
    fold_coverage = pd.DataFrame()
    if not metrics_df.empty:
        fold_coverage = (
            metrics_df.groupby(["model_name", "candidate_family"], dropna=False)["outer_fold_index"]
            .nunique()
            .reset_index(name="completed_outer_folds")
        )
        fold_coverage["expected_outer_folds"] = expected_outer
        fold_coverage["full_coverage"] = fold_coverage["completed_outer_folds"] == expected_outer
    write_df(tables_dir / "traditional_fusion_fold_coverage.csv", fold_coverage)

    gap_df = build_gap_table(metrics_df)
    write_df(tables_dir / "traditional_fusion_overfitting_indicators.csv", gap_df)
    ranking_all = build_composite_ranking(summary, gap_df)
    ranking_all = sensitivity_tradeoff_flags(ranking_all)
    ranking_all = attach_eligibility_flags(ranking_all, fold_coverage, expected_outer=expected_outer)
    if "full_coverage" in ranking_all.columns:
        full_mask = ranking_all["full_coverage"].fillna(False).astype(bool)
    else:
        full_mask = pd.Series(False, index=ranking_all.index)
    full_coverage_ranking = ranking_all[full_mask].copy()
    exploratory_ranking = ranking_all[~full_mask].copy()
    write_df(tables_dir / "traditional_fusion_composite_ranking.csv", full_coverage_ranking)
    write_df(tables_dir / "traditional_fusion_full_coverage_ranking.csv", full_coverage_ranking)
    write_df(tables_dir / "traditional_fusion_exploratory_incomplete_ranking.csv", exploratory_ranking)
    eligibility_cols = [
        "model_name",
        "candidate_family",
        "threshold_objective",
        "completed_outer_folds",
        "expected_outer_folds",
        "full_coverage",
        "sensitivity_mean",
        "sensitivity_meets_floor",
        "class0_recall_mean",
        "xgb_scm_default_class0_recall",
        "class0_beats_xgb_scm_default",
        "primary_eligible",
        "primary_ineligibility_reason",
        "composite_score",
        "composite_rank",
    ]
    write_df(
        tables_dir / "traditional_fusion_eligibility_flags.csv",
        ranking_all[[col for col in eligibility_cols if col in ranking_all.columns]].copy(),
    )

    run_summary = {
        "input_file": str(args.input),
        "output_dir": str(args.output_dir),
        "tabpfn_dir": str(args.tabpfn_dir),
        "checkpoint_paths": checkpoint_paths,
        "task": TASK,
        "feature_policy": FEATURE_POLICY,
        "tabpfn_variant": TABPFN_VARIANT,
        "candidate_set": args.candidate_set,
        "traditional_candidates": [candidate.model_id for candidate in candidates],
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "max_outer_folds": args.max_outer_folds,
        "completed_outer_folds": completed_outer,
        "expected_outer_folds": expected_outer,
        "inner_splits": args.inner_splits,
        "n_estimators": args.n_estimators,
        "blend_step": args.blend_step,
        "blend_values": blend_values,
        "device": args.device,
        "fit_mode": args.fit_mode,
        "skip_stacking": bool(args.skip_stacking),
        "smoke_test": bool(args.smoke_test),
        "random_state": args.random_state,
        "feature_audit": feature_audit,
        "elapsed_seconds": round(time.perf_counter() - start_time, 2),
        "tables": {
            "metrics_by_outer_fold": "tables/traditional_fusion_metrics_by_outer_fold.csv",
            "oof_selection_by_fold": "tables/traditional_fusion_oof_selection_by_fold.csv",
            "oof_candidates_by_fold": "tables/traditional_fusion_oof_candidates_by_fold.csv",
            "summary_mean_std": "tables/traditional_fusion_summary_mean_std.csv",
            "composite_ranking": "tables/traditional_fusion_composite_ranking.csv",
            "full_coverage_ranking": "tables/traditional_fusion_full_coverage_ranking.csv",
            "exploratory_incomplete_ranking": "tables/traditional_fusion_exploratory_incomplete_ranking.csv",
            "overfitting_indicators": "tables/traditional_fusion_overfitting_indicators.csv",
            "prediction_audit": "tables/traditional_fusion_prediction_audit.csv",
            "base_prediction_audit": "tables/traditional_fusion_base_prediction_audit.csv",
            "gate_selection_by_fold": "tables/traditional_fusion_gate_selection_by_fold.csv",
            "eligibility_flags": "tables/traditional_fusion_eligibility_flags.csv",
            "feature_policy_columns": "tables/feature_policy_columns.csv",
            "fold_coverage": "tables/traditional_fusion_fold_coverage.csv",
            "candidate_manifest": "tables/traditional_fusion_candidate_manifest.csv",
        },
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"completed: {args.output_dir}", flush=True)
    if not full_coverage_ranking.empty:
        print(full_coverage_ranking.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
