from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.utils.class_weight import compute_sample_weight
from tabpfn_client import TabPFNClassifier, set_access_token

from env_utils import get_tabpfn_token
from frontier_augmentation import (
    AugmentationResult,
    EXOGENOUS_COLUMNS,
    POST_TREATMENT_COLUMNS,
    SCREENING_COLUMNS,
    TREATMENT_COLUMNS,
    bootstrap_value,
    clip_numeric_value,
    compute_anchor_distance,
    existing_columns,
    infer_discrete_numeric_features,
    make_xgb_classifier,
    make_xgb_regressor,
    summarize_audit,
)
from frontier_scm_v2_experiment import (
    build_hard_case_index_map,
    compute_bootstrap_ci,
    extract_class_recall,
    fit_generic_teacher,
    make_lgbm_classifier,
)
from multiclass_ensemble_experiment import (
    TARGET_LABELS,
    build_preprocessor,
    evaluate_predictions,
    transform_with_preprocessor,
)
from scm_v3_medical_rules import RuleDecision, evaluate_medical_rules


@dataclass
class TeacherModelArtifacts:
    name: str
    teacher_type: str
    preprocessor: Any
    model: Any
    feature_names: list[str]
    extra: dict[str, Any]


@dataclass
class NodeModelBundle:
    target: str
    parents: list[str]
    preprocessor: Any
    feature_names: list[str]
    model: Any
    residuals: np.ndarray
    parent_train_matrix: np.ndarray
    target_train_values: np.ndarray


def _fit_tabpfn_teacher(X_train_raw: pd.DataFrame, y_train: pd.Series, project_dir: Path, seed: int) -> TeacherModelArtifacts:
    token = get_tabpfn_token(project_dir)
    if not token:
        raise RuntimeError("未找到 TabPFN access token。")

    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, _ = transform_with_preprocessor(preprocessor, X_train_raw, None)
    train_mean = X_train_df.mean(axis=0)
    train_std = X_train_df.std(axis=0, ddof=0).replace(0, 1.0)
    X_train_dense = ((X_train_df - train_mean) / train_std).to_numpy(dtype=float)

    set_access_token(token)
    model = TabPFNClassifier(random_state=seed, balance_probabilities=True)
    model.fit(X_train_dense, y_train.to_numpy())
    return TeacherModelArtifacts(
        name="tabpfn_teacher",
        teacher_type="tabpfn",
        preprocessor=preprocessor,
        model=model,
        feature_names=list(X_train_df.columns),
        extra={"train_mean": train_mean, "train_std": train_std},
    )


def _wrap_generic_teacher(teacher: Any, teacher_type: str) -> TeacherModelArtifacts:
    return TeacherModelArtifacts(
        name=teacher.name,
        teacher_type=teacher_type,
        preprocessor=teacher.preprocessor,
        model=teacher.model,
        feature_names=teacher.feature_names,
        extra={},
    )


def build_teacher_ensemble_v3(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    teacher_filter_mode: str,
    project_dir: Path | None = None,
    use_remote_teacher: bool = False,
) -> list[TeacherModelArtifacts]:
    if teacher_filter_mode == "single":
        return [
            _wrap_generic_teacher(
                fit_generic_teacher(X_train_raw, y_train, make_xgb_classifier, seed, "xgb_teacher"),
                "tree",
            )
        ]

    if teacher_filter_mode == "hetero_consensus":
        teachers = [
            _wrap_generic_teacher(
                fit_generic_teacher(X_train_raw, y_train, make_xgb_classifier, seed, "xgb_teacher"),
                "tree",
            ),
            _wrap_generic_teacher(
                fit_generic_teacher(X_train_raw, y_train, make_lgbm_classifier, seed + 17, "lgb_teacher"),
                "tree",
            ),
        ]
        if use_remote_teacher and project_dir is not None:
            try:
                teachers.append(_fit_tabpfn_teacher(X_train_raw, y_train, project_dir, seed + 29))
            except Exception:
                pass
        return teachers

    raise ValueError(f"未知 teacher_filter_mode: {teacher_filter_mode}")


def _score_single_teacher(teacher: TeacherModelArtifacts, candidate_df: pd.DataFrame) -> np.ndarray:
    transformed = teacher.preprocessor.transform(candidate_df)
    X_df = pd.DataFrame(transformed, columns=teacher.feature_names, index=candidate_df.index)
    if teacher.teacher_type == "tabpfn":
        train_mean = teacher.extra["train_mean"]
        train_std = teacher.extra["train_std"]
        X_dense = ((X_df - train_mean) / train_std).to_numpy(dtype=float)
        return np.asarray(teacher.model.predict_proba(X_dense))[0]
    return np.asarray(teacher.model.predict_proba(X_df))[0]


def score_candidate_with_teachers_v3(
    teachers: list[TeacherModelArtifacts],
    candidate_df: pd.DataFrame,
    target_class: int,
    teacher_filter_mode: str,
) -> dict[str, Any]:
    named_scores: dict[str, np.ndarray] = {}
    for teacher in teachers:
        named_scores[teacher.name] = _score_single_teacher(teacher, candidate_df)

    stacked = np.vstack(list(named_scores.values()))
    mean_proba = stacked.mean(axis=0)
    ordered = np.sort(mean_proba)[::-1]
    margin = float(ordered[0] - ordered[1]) if mean_proba.size >= 2 else float(ordered[0])
    target_probs = stacked[:, int(target_class)]
    disagreement = float(target_probs.std(ddof=0)) if target_probs.size > 1 else 0.0
    teacher_pass_count = int((target_probs >= 0.55).sum())
    if teacher_filter_mode == "hetero_consensus":
        passed = teacher_pass_count >= max(2, min(2, len(teachers))) or (
            float(target_probs.mean()) >= 0.62 and disagreement <= 0.12
        )
    else:
        passed = float(target_probs.mean()) >= (0.57 if int(target_class) == 0 else 0.52)

    return {
        "named_scores": named_scores,
        "mean_proba": mean_proba,
        "pred": int(np.argmax(mean_proba)),
        "target_proba": float(mean_proba[int(target_class)]),
        "margin": margin,
        "disagreement": disagreement,
        "teacher_pass_count": teacher_pass_count,
        "passed": bool(passed),
    }


def _fit_numeric_node(
    train_df: pd.DataFrame,
    target_col: str,
    parent_cols: list[str],
    random_state: int,
) -> NodeModelBundle | None:
    if target_col not in train_df.columns or not parent_cols:
        return None
    y = pd.to_numeric(train_df[target_col], errors="coerce")
    valid_mask = y.notna()
    if valid_mask.sum() < 12:
        return None
    X = train_df.loc[valid_mask, parent_cols].copy()
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_df, _ = transform_with_preprocessor(preprocessor, X, None)
    model = make_xgb_regressor(random_state)
    model.fit(X_df, y.loc[valid_mask], verbose=False)
    pred = model.predict(X_df)
    residuals = y.loc[valid_mask].to_numpy(dtype=float) - pred
    return NodeModelBundle(
        target=target_col,
        parents=parent_cols,
        preprocessor=preprocessor,
        feature_names=list(X_df.columns),
        model=model,
        residuals=residuals,
        parent_train_matrix=X_df.to_numpy(dtype=float),
        target_train_values=y.loc[valid_mask].to_numpy(dtype=float),
    )


def _sample_residual_value(
    rng: np.random.Generator,
    bundle: NodeModelBundle,
    row: pd.Series,
    sampler_strength: float,
    support: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    X_raw = pd.DataFrame([row[bundle.parents].to_dict()])
    transformed = bundle.preprocessor.transform(X_raw)
    X_df = pd.DataFrame(transformed, columns=bundle.feature_names, index=X_raw.index)
    pred = float(bundle.model.predict(X_df)[0])
    if bundle.residuals.size > 0:
        sampled_residual = float(bundle.residuals[int(rng.integers(0, bundle.residuals.size))])
        pred += sampled_residual * sampler_strength
    return clip_numeric_value(pred, pd.to_numeric(support, errors="coerce")), {
        "local_neighbor_count": np.nan,
        "kde_bandwidth": np.nan,
        "sampling_fallback": "residual",
    }


def _sample_kde_value(
    rng: np.random.Generator,
    bundle: NodeModelBundle,
    row: pd.Series,
    sampler_strength: float,
    support: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    X_raw = pd.DataFrame([row[bundle.parents].to_dict()])
    transformed = bundle.preprocessor.transform(X_raw)
    X_df = pd.DataFrame(transformed, columns=bundle.feature_names, index=X_raw.index)
    query = X_df.to_numpy(dtype=float)[0]
    train_matrix = bundle.parent_train_matrix
    if train_matrix.shape[0] < 8:
        return _sample_residual_value(rng, bundle, row, sampler_strength, support)

    distances = np.linalg.norm(train_matrix - query, axis=1)
    k = max(8, min(24, train_matrix.shape[0] // 2))
    neighbor_idx = np.argsort(distances)[:k]
    local_targets = bundle.target_train_values[neighbor_idx]
    local_targets = local_targets[np.isfinite(local_targets)]
    if local_targets.size < 6:
        return _sample_residual_value(rng, bundle, row, sampler_strength, support)

    local_std = float(np.std(local_targets, ddof=0))
    bandwidth = max(local_std * max(0.15, sampler_strength * 0.35), 1e-3)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(local_targets.reshape(-1, 1))
    sampled = float(kde.sample(1, random_state=int(rng.integers(0, 1_000_000)))[0, 0])
    sampled = clip_numeric_value(sampled, pd.to_numeric(support, errors="coerce"))
    return sampled, {
        "local_neighbor_count": int(local_targets.size),
        "kde_bandwidth": float(bandwidth),
        "sampling_fallback": "none",
    }


def fit_counterfactual_arr_bundle(X_train_raw: pd.DataFrame) -> dict[str, Any]:
    treatment_cols = existing_columns(X_train_raw, TREATMENT_COLUMNS)
    parent_cols = existing_columns(X_train_raw, EXOGENOUS_COLUMNS + treatment_cols)
    target_cols = [col for col in ["肾素", "醛固酮", "ARR比值"] if col in X_train_raw.columns]
    bundles = {col: _fit_numeric_node(X_train_raw, col, parent_cols, 42) for col in target_cols}
    return {"parents": parent_cols, "targets": bundles}


def generate_counterfactual_rows(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    max_rows: int | None = None,
) -> AugmentationResult:
    rng = np.random.default_rng(random_state)
    class_zero = X_train_raw.loc[y_train == 0].copy()
    if class_zero.empty:
        return AugmentationResult(
            X_aug=pd.DataFrame(columns=X_train_raw.columns),
            y_aug=pd.Series(dtype=int, name=y_train.name),
            audit=pd.DataFrame([{"row_type": "summary", "feature": "no_counterfactual_rows", "value": 0}]),
            metadata=pd.DataFrame(),
        )

    bundles = fit_counterfactual_arr_bundle(X_train_raw)
    treatment_cols = existing_columns(X_train_raw, TREATMENT_COLUMNS)
    limit = max_rows or min(len(class_zero) * 2, 80)
    rows: list[pd.Series] = []
    metadata_rows: list[dict[str, Any]] = []
    for idx in range(min(limit, len(class_zero))):
        base_row = class_zero.iloc[int(rng.integers(0, len(class_zero)))].copy()
        candidate = base_row.copy()
        changed_cols: list[str] = []
        for col in treatment_cols:
            if rng.random() < 0.6:
                candidate[col] = 0.0
                changed_cols.append(col)

        for col, bundle in bundles["targets"].items():
            if bundle is None:
                continue
            value, sampler_meta = _sample_residual_value(rng, bundle, candidate, 0.35, X_train_raw[col])
            candidate[col] = value
            metadata_rows.append(
                {
                    "row_index": idx,
                    "target": col,
                    "sampler": "counterfactual_residual",
                    "local_neighbor_count": sampler_meta["local_neighbor_count"],
                    "kde_bandwidth": sampler_meta["kde_bandwidth"],
                    "sampling_fallback": sampler_meta["sampling_fallback"],
                }
            )

        rows.append(candidate)
        metadata_rows.append(
            {
                "row_index": idx,
                "target_class": 0,
                "method": "counterfactual_do",
                "changed_treatment_columns": "|".join(changed_cols),
                "changed_treatment_count": len(changed_cols),
            }
        )

    augmented_df = pd.DataFrame(rows).reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata_rows)
    y_aug = pd.Series(np.zeros(len(augmented_df), dtype=int), name=y_train.name)
    audit = summarize_audit(X_train_raw, augmented_df, y_aug, metadata_df)
    return AugmentationResult(X_aug=augmented_df, y_aug=y_aug, audit=audit, metadata=metadata_df)


class SCMMixV3Augmentor:
    def __init__(
        self,
        *,
        random_state: int,
        seed_strategy: str,
        target_classes: str,
        treat_mix_prob: float,
        sampler_strength: float,
        node_sampler: str = "conditional_kde",
        rule_filter_mode: str = "medical_rules",
        teacher_filter_mode: str = "single",
        counterfactual_mode: str = "off",
        project_dir: Path | None = None,
        use_remote_teacher: bool = False,
    ) -> None:
        self.random_state = random_state
        self.seed_strategy = seed_strategy
        self.target_classes = target_classes
        self.treat_mix_prob = treat_mix_prob
        self.sampler_strength = sampler_strength
        self.node_sampler = node_sampler
        self.rule_filter_mode = rule_filter_mode
        self.teacher_filter_mode = teacher_filter_mode
        self.counterfactual_mode = counterfactual_mode
        self.project_dir = project_dir
        self.use_remote_teacher = use_remote_teacher

    def _target_class_ids(self) -> set[int]:
        if self.target_classes == "class0_only":
            return {0}
        if self.target_classes == "class0_and_class2":
            return {0, 2}
        raise ValueError(f"未知 target_classes: {self.target_classes}")

    def _target_counts(self, y_train: pd.Series) -> dict[int, int]:
        counts = y_train.value_counts().to_dict()
        if not counts:
            return {}
        allowed = self._target_class_ids()
        majority_count = int(max(counts.values()))
        target_counts: dict[int, int] = {}
        for cls, count in counts.items():
            if int(cls) not in allowed:
                continue
            max_ratio = 0.90 if int(cls) == 0 else 0.92
            desired = min(int(np.ceil(majority_count * max_ratio)), int(count * 6))
            if desired > count:
                target_counts[int(cls)] = desired
        return target_counts

    def _select_base_row(
        self,
        class_df: pd.DataFrame,
        class_id: int,
        hard_case_map: dict[int, list[int]] | None,
        rng: np.random.Generator,
    ) -> pd.Series:
        if self.seed_strategy != "hard_case_seed" or not hard_case_map or class_id not in hard_case_map:
            return class_df.iloc[int(rng.integers(0, len(class_df)))].copy()
        ranked_indices = [idx for idx in hard_case_map[class_id] if idx in class_df.index]
        if not ranked_indices:
            return class_df.iloc[int(rng.integers(0, len(class_df)))].copy()
        top_k = max(3, min(len(ranked_indices), max(5, len(ranked_indices) // 2)))
        candidate_indices = ranked_indices[:top_k]
        weights = np.linspace(top_k, 1, top_k, dtype=float)
        weights /= weights.sum()
        chosen_index = int(rng.choice(candidate_indices, p=weights))
        return class_df.loc[chosen_index].copy()

    def _generate_node_value(
        self,
        rng: np.random.Generator,
        bundle: NodeModelBundle | None,
        row: pd.Series,
        support: pd.Series,
    ) -> tuple[Any, dict[str, Any]]:
        if bundle is None:
            return bootstrap_value(rng, support), {
                "local_neighbor_count": np.nan,
                "kde_bandwidth": np.nan,
                "sampling_fallback": "bootstrap",
            }
        if self.node_sampler == "conditional_kde":
            return _sample_kde_value(rng, bundle, row, self.sampler_strength, support)
        if self.node_sampler == "residual":
            return _sample_residual_value(rng, bundle, row, self.sampler_strength, support)
        raise ValueError(f"未知 node_sampler: {self.node_sampler}")

    def generate(
        self,
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
        hard_case_map: dict[int, list[int]] | None = None,
    ) -> AugmentationResult:
        rng = np.random.default_rng(self.random_state)
        teachers = build_teacher_ensemble_v3(
            X_train_raw,
            y_train,
            self.random_state,
            self.teacher_filter_mode,
            self.project_dir,
            self.use_remote_teacher,
        )
        discrete_features = set(infer_discrete_numeric_features(X_train_raw))
        target_counts = self._target_counts(y_train)

        exogenous_columns = existing_columns(X_train_raw, EXOGENOUS_COLUMNS)
        treatment_columns = existing_columns(X_train_raw, TREATMENT_COLUMNS)
        screening_columns = existing_columns(X_train_raw, SCREENING_COLUMNS)
        post_columns = existing_columns(X_train_raw, POST_TREATMENT_COLUMNS)
        screening_models = {
            col: _fit_numeric_node(X_train_raw, col, exogenous_columns + treatment_columns, self.random_state)
            for col in screening_columns
        }
        post_models = {
            col: _fit_numeric_node(
                X_train_raw,
                col,
                exogenous_columns + treatment_columns + screening_columns,
                self.random_state,
            )
            for col in post_columns
        }
        numeric_support_cache = {
            col: pd.to_numeric(X_train_raw[col], errors="coerce")
            for col in set(treatment_columns + screening_columns + post_columns)
            if col in X_train_raw.columns
        }

        augmented_rows: list[pd.Series] = []
        metadata_rows: list[dict[str, Any]] = []

        for cls, desired_total in target_counts.items():
            class_mask = y_train == cls
            class_df = X_train_raw.loc[class_mask].copy()
            current_count = int(class_mask.sum())
            need = max(0, desired_total - current_count)
            if need == 0 or class_df.empty:
                continue

            attempts = 0
            max_attempts = max(need * 40, 80)
            accepted_for_class = 0

            while accepted_for_class < need and attempts < max_attempts:
                attempts += 1
                base_row = self._select_base_row(class_df, int(cls), hard_case_map, rng)
                donor_row = X_train_raw.iloc[int(rng.integers(0, len(X_train_raw)))].copy()
                candidate = base_row.copy()
                intervention_columns: list[str] = []

                for col in exogenous_columns:
                    candidate[col] = base_row[col]

                for col in treatment_columns:
                    if self.counterfactual_mode == "treatment_do" and rng.random() < self.treat_mix_prob * 0.5:
                        candidate[col] = 0.0 if col != "联合用药_总数" else max(0.0, float(base_row.get(col, 0)) - 1.0)
                        intervention_columns.append(col)
                    elif rng.random() < self.treat_mix_prob:
                        candidate[col] = donor_row[col]
                    else:
                        candidate[col] = base_row[col]
                    numeric_value = pd.to_numeric(pd.Series([candidate[col]]), errors="coerce").iloc[0]
                    if pd.notna(numeric_value):
                        if col in discrete_features:
                            numeric_value = float(np.round(numeric_value))
                        candidate[col] = clip_numeric_value(float(numeric_value), numeric_support_cache.get(col, pd.Series(dtype=float)))

                sampler_meta: dict[str, Any] = {}
                for col in screening_columns:
                    generated, meta = self._generate_node_value(rng, screening_models.get(col), candidate, X_train_raw[col])
                    if col in discrete_features and pd.notna(generated):
                        generated = float(np.round(generated))
                    candidate[col] = generated
                    sampler_meta[f"{col}__neighbor_count"] = meta["local_neighbor_count"]
                    sampler_meta[f"{col}__bandwidth"] = meta["kde_bandwidth"]
                    sampler_meta[f"{col}__fallback"] = meta["sampling_fallback"]

                for col in post_columns:
                    generated, meta = self._generate_node_value(rng, post_models.get(col), candidate, X_train_raw[col])
                    if col in discrete_features and pd.notna(generated):
                        generated = float(np.round(generated))
                    candidate[col] = generated
                    sampler_meta[f"{col}__neighbor_count"] = meta["local_neighbor_count"]
                    sampler_meta[f"{col}__bandwidth"] = meta["kde_bandwidth"]
                    sampler_meta[f"{col}__fallback"] = meta["sampling_fallback"]

                rule_decision: RuleDecision | None = None
                if self.rule_filter_mode == "medical_rules":
                    rule_decision = evaluate_medical_rules(candidate, X_train_raw)
                    if not rule_decision.is_valid:
                        metadata_rows.append(
                            {
                                "method": "scm_v3_rejected",
                                "target_class": int(cls),
                                "rule_pass": False,
                                "rule_fail_reason": "|".join(rule_decision.reasons),
                                "rule_ids": "|".join(rule_decision.failed_rule_ids),
                                "teacher_filter_mode": self.teacher_filter_mode,
                                "node_sampler": self.node_sampler,
                                "counterfactual_mode": self.counterfactual_mode,
                            }
                        )
                        continue

                candidate_df = pd.DataFrame([candidate])
                teacher_score = score_candidate_with_teachers_v3(
                    teachers,
                    candidate_df,
                    int(cls),
                    self.teacher_filter_mode,
                )
                target_proba = teacher_score["target_proba"]
                pred = teacher_score["pred"]
                margin = teacher_score["margin"]
                disagreement = teacher_score["disagreement"]
                distance_to_seed = compute_anchor_distance(base_row, candidate, exogenous_columns + treatment_columns)
                changed_treatment_count = sum(int(str(candidate[col]) != str(base_row[col])) for col in treatment_columns)
                threshold = 0.58 if int(cls) == 0 else 0.52

                if pred != int(cls) or target_proba < threshold or not teacher_score["passed"]:
                    continue

                augmented_rows.append(candidate)
                metadata_rows.append(
                    {
                        "method": "scm_v3",
                        "target_class": int(cls),
                        "node_sampler": self.node_sampler,
                        "rule_filter_mode": self.rule_filter_mode,
                        "teacher_filter_mode": self.teacher_filter_mode,
                        "counterfactual_mode": self.counterfactual_mode,
                        "seed_strategy": self.seed_strategy,
                        "target_classes": self.target_classes,
                        "treat_mix_prob": self.treat_mix_prob,
                        "sampler_strength": self.sampler_strength,
                        "teacher_target_proba": target_proba,
                        "teacher_pred_class": pred,
                        "teacher_margin": margin,
                        "teacher_disagreement": disagreement,
                        "teacher_pass_count": teacher_score["teacher_pass_count"],
                        "changed_treatment_count": changed_treatment_count,
                        "distance_to_seed": distance_to_seed,
                        "rule_pass": True if rule_decision is None else rule_decision.is_valid,
                        "rule_fail_reason": "",
                        "rule_ids": "",
                        "intervention_columns": "|".join(intervention_columns),
                        **sampler_meta,
                    }
                )
                accepted_for_class += 1

        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df = metadata_df[metadata_df["method"] == "scm_v3"].reset_index(drop=True)
            y_aug = metadata_df["target_class"].astype(int).reset_index(drop=True)
            augmented_df = augmented_df.iloc[: len(metadata_df)].reset_index(drop=True)
        else:
            augmented_df = pd.DataFrame(columns=X_train_raw.columns)
            y_aug = pd.Series(dtype=int, name=y_train.name)
            metadata_df = pd.DataFrame(columns=["method", "target_class", "teacher_target_proba"])

        audit = summarize_audit(X_train_raw, augmented_df, y_aug, metadata_df)
        return AugmentationResult(X_aug=augmented_df, y_aug=y_aug, audit=audit, metadata=metadata_df)


def summarize_model_metrics(metrics_by_seed: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, model_name in enumerate(model_names):
        model_df = metrics_by_seed[metrics_by_seed["model_name"] == model_name].copy()
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
            "augmented_size",
        ]:
            values = model_df[metric].to_numpy(dtype=float)
            ci_low, ci_high = compute_bootstrap_ci(values, seed=6060 + idx)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def collect_metric_row(
    *,
    seed: int,
    model_name: str,
    metrics: dict[str, Any],
    augmented_size: int,
    train_size: int | None = None,
    resampled_train_size: int | None = None,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "model_name": model_name,
        "train_size": train_size if train_size is not None else np.nan,
        "resampled_train_size": resampled_train_size if resampled_train_size is not None else np.nan,
        "augmented_size": augmented_size,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "ovr_roc_auc_macro": metrics["ovr_roc_auc_macro"],
        "class0_recall": extract_class_recall(metrics, "非确诊"),
        "class1_recall": extract_class_recall(metrics, "确诊"),
        "class2_recall": extract_class_recall(metrics, "灰色区域"),
    }


def evaluate_model_from_raw(
    *,
    seed: int,
    model_name: str,
    y_test: pd.Series,
    proba: np.ndarray,
    augmented_size: int,
    train_size: int | None = None,
    resampled_train_size: int | None = None,
) -> dict[str, Any]:
    metrics = evaluate_predictions(y_test, proba, TARGET_LABELS, [0, 1, 2])
    return collect_metric_row(
        seed=seed,
        model_name=model_name,
        metrics=metrics,
        augmented_size=augmented_size,
        train_size=train_size,
        resampled_train_size=resampled_train_size,
    )
