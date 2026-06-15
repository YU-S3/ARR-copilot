from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tabpfn_client import TabPFNClassifier, set_access_token
from xgboost import XGBClassifier

from causal_xgboost_variants_experiment import XGB_BEST_PARAMS
from env_utils import get_tabpfn_token
from frontier_augmentation import (
    EXOGENOUS_COLUMNS,
    POST_TREATMENT_COLUMNS,
    SCREENING_COLUMNS,
    TREATMENT_COLUMNS,
    AugmentationResult,
    SCMMixAugmentor,
    bootstrap_value,
    clip_numeric_value,
    compute_anchor_distance,
    existing_columns,
    fit_teacher_model,
    infer_discrete_numeric_features,
    make_xgb_regressor,
    summarize_audit,
    transform_raw_with_teacher,
)
from multiclass_ensemble_experiment import (
    DEFAULT_INPUT_FILE,
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


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="SCM-v2 rerun experiment.")
    parser.add_argument("--input", type=Path, default=project_dir / DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "frontier_scm_v2_outputs")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--smoke-test", action="store_true", help="Run one seed and a tiny config set.")
    parser.add_argument("--phase1-only", action="store_true", help="Stop after Phase 1 baseline/config validation.")
    parser.add_argument("--skip-tabpfn", action="store_true", help="Skip TabPFN phase even when a token is present.")
    return parser.parse_args()


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
    resampled_train_size: int
    used_adasyn: bool


@dataclass
class TeacherModelArtifacts:
    name: str
    preprocessor: Any
    model: Any
    feature_names: list[str]


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressTracker:
    def __init__(self, output_dir: Path, total_steps: int) -> None:
        self.output_dir = output_dir
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0
        self.start_time = time.perf_counter()
        self.progress_file = output_dir / "progress.json"
        self.log_file = output_dir / "progress.log"
        self.current_stage = "init"
        self.current_message = "初始化"
        self.current_context: dict[str, Any] = {}
        self._write_progress()

    def _snapshot(self) -> dict[str, Any]:
        elapsed_seconds = time.perf_counter() - self.start_time
        return {
            "stage": self.current_stage,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "progress_percent": round(self.completed_steps / self.total_steps * 100, 2),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "elapsed_human": format_seconds(elapsed_seconds),
            "current_message": self.current_message,
            "context": self.current_context,
        }

    def _write_progress(self) -> None:
        snapshot = self._snapshot()
        self.progress_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(
        self,
        message: str,
        *,
        stage: str | None = None,
        advance: int = 0,
        context: dict[str, Any] | None = None,
    ) -> None:
        if stage is not None:
            self.current_stage = stage
        self.current_message = message
        if context is not None:
            self.current_context = context
        if advance:
            self.completed_steps = min(self.total_steps, self.completed_steps + advance)

        snapshot = self._snapshot()
        line = (
            f"[{snapshot['elapsed_human']}] "
            f"{snapshot['completed_steps']}/{snapshot['total_steps']} "
            f"({snapshot['progress_percent']:.2f}%) | "
            f"{snapshot['stage']} | {snapshot['current_message']}"
        )
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._write_progress()


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


def make_lgbm_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=seed,
        n_estimators=260,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=4,
        min_child_samples=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.1,
        n_jobs=-1,
        verbose=-1,
    )


def prepare_split_data(input_path: Path, seed: int) -> SplitData:
    input_path = Path(input_path)
    resolved_input = input_path / DEFAULT_INPUT_FILE if input_path.is_dir() else input_path
    prepared = prepare_data(resolved_input)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=seed,
        stratify=prepared.y,
    )
    return SplitData(
        X_train_raw=x_train_raw.reset_index(drop=True),
        X_test_raw=x_test_raw.reset_index(drop=True),
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
    return TrainResult(
        proba=np.asarray(model.predict_proba(X_test_df)),
        train_size=int(len(X_train_raw)),
        resampled_train_size=resampled_train_size,
        used_adasyn=use_adasyn,
    )


def run_tabpfn_with_raw(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    project_dir: Path,
    seed: int,
    max_retries: int = 4,
    retry_wait_seconds: float = 8.0,
    status_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    token = get_tabpfn_token(project_dir)
    if not token:
        raise RuntimeError("未找到 TabPFN access token。请先在 .env 中填写 TABPFN_API_TOKEN。")

    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, X_test_df = transform_with_preprocessor(preprocessor, X_train_raw, X_test_raw)

    train_mean = X_train_df.mean(axis=0)
    train_std = X_train_df.std(axis=0, ddof=0).replace(0, 1.0)
    X_train_dense = ((X_train_df - train_mean) / train_std).to_numpy(dtype=float)
    X_test_dense = ((X_test_df - train_mean) / train_std).to_numpy(dtype=float)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if status_callback is not None:
                status_callback(f"TabPFN 调用尝试 {attempt}/{max_retries}，seed={seed}")
            set_access_token(token)
            model = TabPFNClassifier(random_state=seed, balance_probabilities=True)
            model.fit(X_train_dense, y_train.to_numpy())
            return np.asarray(model.predict_proba(X_test_dense))
        except Exception as exc:  # pragma: no cover - 依赖远程服务
            last_error = exc
            if attempt >= max_retries:
                break
            wait_seconds = retry_wait_seconds * attempt
            if status_callback is not None:
                status_callback(
                    f"TabPFN 第 {attempt} 次失败：{type(exc).__name__}: {exc}；{wait_seconds:.0f}s 后重试"
                )
            time.sleep(wait_seconds)

    assert last_error is not None
    raise RuntimeError(
        f"TabPFN 在 {max_retries} 次尝试后仍失败，最后错误：{type(last_error).__name__}: {last_error}"
    ) from last_error


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


def fit_generic_teacher(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    model_factory: Any,
    seed: int,
    teacher_name: str,
) -> TeacherModelArtifacts:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, _ = transform_with_preprocessor(preprocessor, X_train_raw, None)
    model = model_factory(seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_df, y_train, sample_weight=sample_weight)
    return TeacherModelArtifacts(
        name=teacher_name,
        preprocessor=preprocessor,
        model=model,
        feature_names=list(X_train_df.columns),
    )


def score_candidate_with_teachers(
    teachers: list[TeacherModelArtifacts],
    candidate_df: pd.DataFrame,
    target_class: int,
) -> dict[str, Any]:
    proba_list: list[np.ndarray] = []
    for teacher in teachers:
        transformed = teacher.preprocessor.transform(candidate_df)
        X_df = pd.DataFrame(transformed, columns=teacher.feature_names, index=candidate_df.index)
        proba = np.asarray(teacher.model.predict_proba(X_df))[0]
        proba_list.append(proba)

    stacked = np.vstack(proba_list)
    mean_proba = stacked.mean(axis=0)
    ordered = np.sort(mean_proba)[::-1]
    margin = float(ordered[0] - ordered[1]) if mean_proba.size >= 2 else float(ordered[0])
    target_probs = stacked[:, int(target_class)]
    disagreement = float(target_probs.std(ddof=0)) if target_probs.size > 1 else 0.0
    return {
        "mean_proba": mean_proba,
        "pred": int(np.argmax(mean_proba)),
        "target_proba": float(mean_proba[int(target_class)]),
        "margin": margin,
        "disagreement": disagreement,
    }


def build_teacher_ensemble(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    teacher_mode: str,
) -> list[TeacherModelArtifacts]:
    if teacher_mode == "single":
        teacher = fit_teacher_model(X_train_raw, y_train)
        return [
            TeacherModelArtifacts(
                name="xgb_single",
                preprocessor=teacher.preprocessor,
                model=teacher.model,
                feature_names=teacher.feature_names,
            )
        ]
    if teacher_mode == "oof":
        teachers: list[TeacherModelArtifacts] = []
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        for fold_id, (train_idx, _) in enumerate(skf.split(X_train_raw, y_train)):
            X_fold = X_train_raw.iloc[train_idx].reset_index(drop=True)
            y_fold = y_train.iloc[train_idx].reset_index(drop=True)
            teacher = fit_teacher_model(X_fold, y_fold)
            teachers.append(
                TeacherModelArtifacts(
                    name=f"xgb_oof_{fold_id}",
                    preprocessor=teacher.preprocessor,
                    model=teacher.model,
                    feature_names=teacher.feature_names,
                )
            )
        return teachers
    if teacher_mode == "dual":
        return [
            fit_generic_teacher(X_train_raw, y_train, make_xgb_classifier, seed, "xgb_teacher"),
            fit_generic_teacher(X_train_raw, y_train, make_lgbm_classifier, seed + 11, "lgb_teacher"),
        ]
    raise ValueError(f"未知 teacher_mode: {teacher_mode}")


def build_hard_case_index_map(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> dict[int, list[int]]:
    records: list[dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(X_train_raw, y_train)):
        X_fold_train = X_train_raw.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
        X_fold_valid = X_train_raw.iloc[valid_idx].reset_index(drop=True)
        y_fold_valid = y_train.iloc[valid_idx].reset_index(drop=True)
        result = fit_xgb_from_raw(
            X_fold_train,
            y_fold_train,
            X_fold_valid,
            infer_discrete_numeric_features(X_fold_train),
            seed=seed + fold_id,
            use_adasyn=False,
        )
        pred = np.argmax(result.proba, axis=1)
        sorted_proba = np.sort(result.proba, axis=1)
        margins = sorted_proba[:, -1] - sorted_proba[:, -2]
        for local_idx, global_idx in enumerate(valid_idx):
            records.append(
                {
                    "row_index": int(global_idx),
                    "class_id": int(y_train.iloc[global_idx]),
                    "is_error": int(pred[local_idx] != y_train.iloc[global_idx]),
                    "margin": float(margins[local_idx]),
                    "hard_score": float(int(pred[local_idx] != y_train.iloc[global_idx]) * 2 + (1.0 - margins[local_idx])),
                }
            )
    score_df = pd.DataFrame(records).sort_values(
        ["class_id", "hard_score", "margin"],
        ascending=[True, False, True],
    )
    return {
        int(class_id): group["row_index"].astype(int).tolist()
        for class_id, group in score_df.groupby("class_id")
    }


class SCMMixV2Augmentor:
    def __init__(
        self,
        random_state: int,
        seed_strategy: str,
        target_classes: str,
        treat_mix_prob: float,
        residual_scale: float,
        teacher_mode: str = "single",
    ) -> None:
        self.random_state = random_state
        self.seed_strategy = seed_strategy
        self.target_classes = target_classes
        self.treat_mix_prob = treat_mix_prob
        self.residual_scale = residual_scale
        self.teacher_mode = teacher_mode

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
            max_ratio = 0.85 if int(cls) == 0 else 0.95
            desired = min(int(np.ceil(majority_count * max_ratio)), int(count * 5))
            if desired > count:
                target_counts[int(cls)] = desired
        return target_counts

    def _fit_numeric_node(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        parent_cols: list[str],
    ) -> dict[str, Any] | None:
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
        model = make_xgb_regressor(self.random_state)
        model.fit(X_df, y.loc[valid_mask], verbose=False)
        pred = model.predict(X_df)
        residuals = y.loc[valid_mask].to_numpy(dtype=float) - pred
        return {
            "target": target_col,
            "parents": parent_cols,
            "preprocessor": preprocessor,
            "feature_names": list(X_df.columns),
            "model": model,
            "residuals": residuals,
        }

    def _generate_node_value(
        self,
        rng: np.random.Generator,
        model_bundle: dict[str, Any] | None,
        row: pd.Series,
        support: pd.Series,
        support_numeric: pd.Series | None = None,
    ) -> Any:
        if model_bundle is None:
            return bootstrap_value(rng, support)
        X_raw = pd.DataFrame([row[model_bundle["parents"]].to_dict()])
        transformed = model_bundle["preprocessor"].transform(X_raw)
        X_df = pd.DataFrame(transformed, columns=model_bundle["feature_names"], index=X_raw.index)
        pred = float(model_bundle["model"].predict(X_df)[0])
        residuals = model_bundle["residuals"]
        if residuals.size > 0:
            sampled_residual = float(residuals[int(rng.integers(0, residuals.size))])
            pred += sampled_residual * self.residual_scale
        numeric_support = support_numeric if support_numeric is not None else pd.to_numeric(support, errors="coerce")
        return clip_numeric_value(pred, numeric_support)

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

    def generate(
        self,
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
        hard_case_map: dict[int, list[int]] | None = None,
    ) -> AugmentationResult:
        rng = np.random.default_rng(self.random_state)
        teachers = build_teacher_ensemble(X_train_raw, y_train, self.random_state, self.teacher_mode)
        discrete_features = set(infer_discrete_numeric_features(X_train_raw))
        target_counts = self._target_counts(y_train)

        exogenous_columns = existing_columns(X_train_raw, EXOGENOUS_COLUMNS)
        treatment_columns = existing_columns(X_train_raw, TREATMENT_COLUMNS)
        screening_columns = existing_columns(X_train_raw, SCREENING_COLUMNS)
        post_columns = existing_columns(X_train_raw, POST_TREATMENT_COLUMNS)

        screening_models = {
            col: self._fit_numeric_node(X_train_raw, col, exogenous_columns + treatment_columns)
            for col in screening_columns
        }
        post_models = {
            col: self._fit_numeric_node(
                X_train_raw,
                col,
                exogenous_columns + treatment_columns + screening_columns,
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
            class_numeric_support = {
                col: pd.to_numeric(class_df[col], errors="coerce")
                for col in treatment_columns
                if col in class_df.columns
            }
            current_count = int(class_mask.sum())
            need = max(0, desired_total - current_count)
            if need == 0 or class_df.empty:
                continue

            attempts = 0
            max_attempts = max(need * 35, 70)
            accepted_for_class = 0

            while accepted_for_class < need and attempts < max_attempts:
                attempts += 1
                base_row = self._select_base_row(class_df, int(cls), hard_case_map, rng)
                donor_row = X_train_raw.iloc[int(rng.integers(0, len(X_train_raw)))].copy()
                candidate = base_row.copy()

                for col in exogenous_columns:
                    candidate[col] = base_row[col]

                for col in treatment_columns:
                    if rng.random() < self.treat_mix_prob:
                        candidate[col] = donor_row[col]
                    else:
                        candidate[col] = base_row[col]
                    support = class_numeric_support.get(col, pd.Series(dtype=float))
                    numeric_value = pd.to_numeric(candidate[col], errors="coerce")
                    if col in discrete_features:
                        numeric_value = float(np.round(numeric_value))
                    if pd.notna(numeric_value):
                        candidate[col] = clip_numeric_value(float(numeric_value), support)

                for col in screening_columns:
                    generated = self._generate_node_value(
                        rng,
                        screening_models.get(col),
                        candidate,
                        X_train_raw[col],
                        numeric_support_cache.get(col),
                    )
                    if col in discrete_features and pd.notna(generated):
                        generated = float(np.round(generated))
                    candidate[col] = generated

                for col in post_columns:
                    generated = self._generate_node_value(
                        rng,
                        post_models.get(col),
                        candidate,
                        X_train_raw[col],
                        numeric_support_cache.get(col),
                    )
                    if col in discrete_features and pd.notna(generated):
                        generated = float(np.round(generated))
                    candidate[col] = generated

                candidate_df = pd.DataFrame([candidate])
                teacher_score = score_candidate_with_teachers(teachers, candidate_df, int(cls))
                target_proba = teacher_score["target_proba"]
                pred = teacher_score["pred"]
                margin = teacher_score["margin"]
                disagreement = teacher_score["disagreement"]
                changed_treatment_count = sum(
                    int(str(candidate[col]) != str(base_row[col])) for col in treatment_columns
                )
                distance_to_seed = compute_anchor_distance(base_row, candidate, exogenous_columns + treatment_columns)
                score = (
                    target_proba
                    + 0.03 * changed_treatment_count
                    + 0.03 * margin
                    - 0.08 * disagreement
                    - 0.02 * distance_to_seed
                    + (0.05 if int(cls) == 0 else 0.0)
                )
                threshold = 0.55 if int(cls) == 0 else 0.50

                if pred != int(cls) or target_proba < threshold:
                    continue

                augmented_rows.append(candidate)
                metadata_rows.append(
                    {
                        "method": "scm_v2",
                        "target_class": int(cls),
                        "teacher_mode": self.teacher_mode,
                        "seed_strategy": self.seed_strategy,
                        "target_classes": self.target_classes,
                        "treat_mix_prob": self.treat_mix_prob,
                        "residual_scale": self.residual_scale,
                        "teacher_target_proba": target_proba,
                        "teacher_pred_class": pred,
                        "teacher_margin": margin,
                        "teacher_disagreement": disagreement,
                        "changed_treatment_count": changed_treatment_count,
                        "distance_to_seed": distance_to_seed,
                        "utility_score": score,
                    }
                )
                accepted_for_class += 1

        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
            metadata_df = pd.DataFrame(metadata_rows).sort_values(
                ["target_class", "utility_score"],
                ascending=[True, False],
            )
            y_aug = metadata_df["target_class"].astype(int).reset_index(drop=True)
            augmented_df = augmented_df.loc[metadata_df.index].reset_index(drop=True)
            metadata_df = metadata_df.reset_index(drop=True)
        else:
            augmented_df = pd.DataFrame(columns=X_train_raw.columns)
            y_aug = pd.Series(dtype=int, name=y_train.name)
            metadata_df = pd.DataFrame(columns=["method", "target_class", "teacher_target_proba"])

        audit = summarize_audit(X_train_raw, augmented_df, y_aug, metadata_df)
        return AugmentationResult(X_aug=augmented_df, y_aug=y_aug, audit=audit, metadata=metadata_df)


def build_phase1_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for seed_strategy in ["random_seed", "hard_case_seed"]:
        for target_classes in ["class0_only", "class0_and_class2"]:
            for treat_mix_prob in [0.2, 0.4]:
                for residual_scale in [0.5, 0.8]:
                    configs.append(
                        {
                            "seed_strategy": seed_strategy,
                            "target_classes": target_classes,
                            "treat_mix_prob": treat_mix_prob,
                            "residual_scale": residual_scale,
                            "teacher_mode": "single",
                        }
                    )
    return configs


def config_to_name(config: dict[str, Any]) -> str:
    seed_tag = "hard" if config["seed_strategy"] == "hard_case_seed" else "random"
    class_tag = "c0" if config["target_classes"] == "class0_only" else "c02"
    tm_tag = f"tm{int(round(config['treat_mix_prob'] * 100)):02d}"
    rs_tag = f"rs{int(round(config['residual_scale'] * 100)):02d}"
    teacher_tag = f"teacher_{config['teacher_mode']}"
    return f"scm_v2_{seed_tag}_{class_tag}_{tm_tag}_{rs_tag}_{teacher_tag}"


def aggregate_metrics(metrics_df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, model_name in enumerate(model_names):
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
            "augmented_size",
        ]:
            values = model_df[metric].to_numpy(dtype=float)
            ci_low, ci_high = compute_bootstrap_ci(values, seed=4040 + idx)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        row["train_size_mean"] = (
            float(model_df["train_size"].mean()) if "train_size" in model_df.columns else float("nan")
        )
        row["resampled_train_size_mean"] = (
            float(model_df["resampled_train_size"].mean())
            if "resampled_train_size" in model_df.columns
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_boxplot(metrics_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(13, 6))
    order = (
        metrics_df.groupby("model_name")[metric]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    sns.boxplot(data=metrics_df, x="model_name", y=metric, order=order)
    sns.stripplot(data=metrics_df, x="model_name", y=metric, order=order, color="black", size=3, alpha=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.title(title)
    plt.xlabel("方案")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_phase1_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    phase1_df = summary_df[summary_df["model_name"].str.startswith("scm_v2_")].copy()
    if phase1_df.empty:
        return
    display_rows: list[dict[str, Any]] = []
    for _, row in phase1_df.iterrows():
        parts = row["model_name"].split("_")
        display_rows.append(
            {
                "seed_strategy": parts[2],
                "target_classes": parts[3],
                "mix_scale": f"{parts[4]}_{parts[5]}",
                "balanced_accuracy_mean": row["balanced_accuracy_mean"],
            }
        )
    display_df = pd.DataFrame(display_rows)
    display_df["row_label"] = display_df["seed_strategy"] + " | " + display_df["target_classes"]
    pivot = display_df.pivot(index="row_label", columns="mix_scale", values="balanced_accuracy_mean")
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("SCM-v2 最小矩阵 Balanced Accuracy 均值热图")
    plt.xlabel("参数组合")
    plt.ylabel("策略")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_partial_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    phase1_configs = build_phase1_configs()
    if args.smoke_test:
        phase1_configs = [
            {
                "seed_strategy": "hard_case_seed",
                "target_classes": "class0_only",
                "treat_mix_prob": 0.4,
                "residual_scale": 0.5,
                "teacher_mode": "single",
            }
        ]
    teacher_modes = ["single"] if args.smoke_test else ["single", "oof", "dual"]
    tabpfn_token_available = bool(get_tabpfn_token(project_dir)) and not args.skip_tabpfn and not args.phase1_only
    total_steps = (
        len(seeds) * (2 + len(phase1_configs))
        + (0 if args.phase1_only else len(seeds) * len(teacher_modes))
        + (len(seeds) * 2 if tabpfn_token_available else 0)
    )
    tracker = ProgressTracker(output_dir=output_dir, total_steps=total_steps)
    tracker.log(
        "实验启动，准备进入 Phase 1",
        stage="startup",
        context={
            "phase1_config_count": len(phase1_configs),
            "teacher_mode_count": len(teacher_modes),
            "tabpfn_token_available": tabpfn_token_available,
            "input_file": str(args.input),
            "smoke_test": args.smoke_test,
            "phase1_only": args.phase1_only,
        },
    )

    phase1_rows: list[dict[str, Any]] = []
    phase1_audit_frames: list[pd.DataFrame] = []
    phase1_meta_frames: list[pd.DataFrame] = []

    for seed_idx, seed in enumerate(seeds, start=1):
        tracker.log(
            f"Phase 1：开始处理随机种子 {seed}（{seed_idx}/{len(seeds)}）",
            stage="phase1_seed_start",
            context={"seed": seed, "seed_index": seed_idx, "phase": "phase1"},
        )
        split_data = prepare_split_data(args.input, seed)
        tracker.log(
            f"Phase 1：构建 hard case 索引，seed={seed}",
            stage="phase1_hard_case",
            context={"seed": seed, "phase": "phase1"},
        )
        hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)

        reference_result = fit_xgb_from_raw(
            split_data.X_train_raw,
            split_data.y_train,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=True,
        )
        reference_metrics = evaluate_predictions(
            split_data.y_test,
            reference_result.proba,
            TARGET_LABELS,
            [0, 1, 2],
        )
        phase1_rows.append(
            {
                "phase": "phase1_baseline",
                "seed": seed,
                "model_name": "xgb_reference_adasyn",
                "train_size": reference_result.train_size,
                "resampled_train_size": reference_result.resampled_train_size,
                "augmented_size": 0,
                "accuracy": reference_metrics["accuracy"],
                "balanced_accuracy": reference_metrics["balanced_accuracy"],
                "macro_f1": reference_metrics["macro_f1"],
                "weighted_f1": reference_metrics["weighted_f1"],
                "ovr_roc_auc_macro": reference_metrics["ovr_roc_auc_macro"],
                "class0_recall": extract_class_recall(reference_metrics, "非确诊"),
                "class1_recall": extract_class_recall(reference_metrics, "确诊"),
                "class2_recall": extract_class_recall(reference_metrics, "灰色区域"),
            }
        )
        write_partial_table(phase1_rows, tables_dir / "phase1_metrics_by_seed.partial.csv")
        tracker.log(
            f"Phase 1：完成 xgb_reference_adasyn，seed={seed}",
            stage="phase1_baseline",
            advance=1,
            context={"seed": seed, "model_name": "xgb_reference_adasyn", "phase": "phase1"},
        )

        baseline_aug = SCMMixAugmentor(random_state=seed).generate(split_data.X_train_raw, split_data.y_train)
        baseline_train_X = pd.concat([split_data.X_train_raw, baseline_aug.X_aug], axis=0, ignore_index=True)
        baseline_train_y = pd.concat([split_data.y_train, baseline_aug.y_aug], axis=0, ignore_index=True)
        baseline_result = fit_xgb_from_raw(
            baseline_train_X,
            baseline_train_y,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed,
            use_adasyn=True,
        )
        baseline_metrics = evaluate_predictions(
            split_data.y_test,
            baseline_result.proba,
            TARGET_LABELS,
            [0, 1, 2],
        )
        phase1_rows.append(
            {
                "phase": "phase1_baseline",
                "seed": seed,
                "model_name": "xgb_scm_plus_adasyn_baseline",
                "train_size": baseline_result.train_size,
                "resampled_train_size": baseline_result.resampled_train_size,
                "augmented_size": int(len(baseline_aug.X_aug)),
                "accuracy": baseline_metrics["accuracy"],
                "balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "macro_f1": baseline_metrics["macro_f1"],
                "weighted_f1": baseline_metrics["weighted_f1"],
                "ovr_roc_auc_macro": baseline_metrics["ovr_roc_auc_macro"],
                "class0_recall": extract_class_recall(baseline_metrics, "非确诊"),
                "class1_recall": extract_class_recall(baseline_metrics, "确诊"),
                "class2_recall": extract_class_recall(baseline_metrics, "灰色区域"),
            }
        )
        phase1_audit_frames.append(baseline_aug.audit.assign(seed=seed, model_name="xgb_scm_plus_adasyn_baseline"))
        phase1_meta_frames.append(baseline_aug.metadata.assign(seed=seed, model_name="xgb_scm_plus_adasyn_baseline"))
        write_partial_table(phase1_rows, tables_dir / "phase1_metrics_by_seed.partial.csv")
        tracker.log(
            f"Phase 1：完成 xgb_scm_plus_adasyn_baseline，seed={seed}，增广样本={len(baseline_aug.X_aug)}",
            stage="phase1_baseline",
            advance=1,
            context={
                "seed": seed,
                "model_name": "xgb_scm_plus_adasyn_baseline",
                "augmented_size": int(len(baseline_aug.X_aug)),
                "phase": "phase1",
            },
        )

        for config_idx, config in enumerate(phase1_configs, start=1):
            model_name = config_to_name(config)
            tracker.log(
                (
                    f"Phase 1：运行配置 {config_idx}/{len(phase1_configs)} | "
                    f"seed={seed} | {model_name}"
                ),
                stage="phase1_grid_running",
                context={
                    "seed": seed,
                    "seed_index": seed_idx,
                    "config_index": config_idx,
                    "config_total": len(phase1_configs),
                    "model_name": model_name,
                    "phase": "phase1",
                },
            )
            augmentor = SCMMixV2Augmentor(
                random_state=seed,
                seed_strategy=config["seed_strategy"],
                target_classes=config["target_classes"],
                treat_mix_prob=config["treat_mix_prob"],
                residual_scale=config["residual_scale"],
                teacher_mode=config["teacher_mode"],
            )
            aug_result = augmentor.generate(
                split_data.X_train_raw,
                split_data.y_train,
                hard_case_map=hard_case_map,
            )
            train_X = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
            train_y = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)
            train_result = fit_xgb_from_raw(
                train_X,
                train_y,
                split_data.X_test_raw,
                split_data.discrete_numeric_features,
                seed,
                use_adasyn=True,
            )
            metrics = evaluate_predictions(split_data.y_test, train_result.proba, TARGET_LABELS, [0, 1, 2])
            phase1_rows.append(
                {
                    "phase": "phase1_grid",
                    "seed": seed,
                    "model_name": model_name,
                    "train_size": train_result.train_size,
                    "resampled_train_size": train_result.resampled_train_size,
                    "augmented_size": int(len(aug_result.X_aug)),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "ovr_roc_auc_macro": metrics["ovr_roc_auc_macro"],
                    "class0_recall": extract_class_recall(metrics, "非确诊"),
                    "class1_recall": extract_class_recall(metrics, "确诊"),
                    "class2_recall": extract_class_recall(metrics, "灰色区域"),
                    **config,
                }
            )
            phase1_audit_frames.append(aug_result.audit.assign(seed=seed, model_name=model_name))
            phase1_meta_frames.append(aug_result.metadata.assign(seed=seed, model_name=model_name))
            write_partial_table(phase1_rows, tables_dir / "phase1_metrics_by_seed.partial.csv")
            tracker.log(
                (
                    f"Phase 1：完成配置 {config_idx}/{len(phase1_configs)} | "
                    f"seed={seed} | {model_name} | 增广样本={len(aug_result.X_aug)}"
                ),
                stage="phase1_grid_done",
                advance=1,
                context={
                    "seed": seed,
                    "config_index": config_idx,
                    "config_total": len(phase1_configs),
                    "model_name": model_name,
                    "augmented_size": int(len(aug_result.X_aug)),
                    "phase": "phase1",
                },
            )

    phase1_df = pd.DataFrame(phase1_rows)
    phase1_df.to_csv(tables_dir / "phase1_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    pd.concat(phase1_audit_frames, ignore_index=True).to_csv(
        tables_dir / "phase1_augmentation_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(phase1_meta_frames, ignore_index=True).to_csv(
        tables_dir / "phase1_augmentation_metadata.csv", index=False, encoding="utf-8-sig"
    )

    phase1_model_names = sorted(phase1_df["model_name"].unique().tolist())
    phase1_summary = aggregate_metrics(phase1_df, phase1_model_names).sort_values(
        ["balanced_accuracy_mean", "macro_f1_mean", "class0_recall_mean", "accuracy_mean"],
        ascending=False,
    )
    phase1_summary.to_csv(tables_dir / "phase1_metrics_mean_std.csv", index=False, encoding="utf-8-sig")

    phase1_config_summary = phase1_summary[phase1_summary["model_name"].str.startswith("scm_v2_")].copy()
    best_phase1_name = phase1_config_summary.iloc[0]["model_name"]
    best_phase1_row = phase1_df[phase1_df["model_name"] == best_phase1_name].iloc[0]
    best_phase1_config = {
        "seed_strategy": best_phase1_row["seed_strategy"],
        "target_classes": best_phase1_row["target_classes"],
        "treat_mix_prob": float(best_phase1_row["treat_mix_prob"]),
        "residual_scale": float(best_phase1_row["residual_scale"]),
    }

    plot_metric_boxplot(
        phase1_df[phase1_df["model_name"].isin(["xgb_reference_adasyn", "xgb_scm_plus_adasyn_baseline", best_phase1_name])],
        "balanced_accuracy",
        figures_dir / "phase1_balanced_accuracy_boxplot.png",
        "Phase 1 关键方案 Balanced Accuracy",
    )
    plot_phase1_heatmap(phase1_summary, figures_dir / "phase1_scm_v2_heatmap.png")
    tracker.log(
        f"Phase 1 完成，当前最优配置：{best_phase1_name}",
        stage="phase1_complete",
        context={"best_phase1_name": best_phase1_name, "phase": "phase1"},
    )

    if args.phase1_only:
        summary = {
            "seeds": seeds,
            "input_file": str(args.input),
            "smoke_test": args.smoke_test,
            "phase1_only": True,
            "phase1_best_config_name": best_phase1_name,
            "phase1_best_config": best_phase1_config,
            "phase2_best_teacher_name": None,
            "phase2_best_teacher_mode": None,
            "tabpfn_token_available": tabpfn_token_available,
            "tabpfn_phase_completed": False,
            "tabpfn_error_count": 0,
        }
        (output_dir / "experiment_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tracker.log("Phase 1-only smoke 完成", stage="complete", context=summary)
        print("SCM-v2 Phase 1-only smoke 完成")
        print("Phase 1 最优配置:", best_phase1_name)
        print(f"结果目录: {output_dir}")
        return

    phase2_rows: list[dict[str, Any]] = []
    for seed_idx, seed in enumerate(seeds, start=1):
        tracker.log(
            f"Phase 2：开始处理随机种子 {seed}（{seed_idx}/{len(seeds)}）",
            stage="phase2_seed_start",
            context={"seed": seed, "seed_index": seed_idx, "phase": "phase2"},
        )
        split_data = prepare_split_data(args.input, seed)
        tracker.log(
            f"Phase 2：构建 hard case 索引，seed={seed}",
            stage="phase2_hard_case",
            context={"seed": seed, "phase": "phase2"},
        )
        hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)
        for teacher_idx, teacher_mode in enumerate(teacher_modes, start=1):
            config = {
                **best_phase1_config,
                "teacher_mode": teacher_mode,
            }
            model_name = f"{best_phase1_name}_teachercmp_{teacher_mode}"
            tracker.log(
                (
                    f"Phase 2：运行教师模式 {teacher_idx}/{len(teacher_modes)} | "
                    f"seed={seed} | {teacher_mode}"
                ),
                stage="phase2_teacher_running",
                context={
                    "seed": seed,
                    "seed_index": seed_idx,
                    "teacher_mode": teacher_mode,
                    "teacher_index": teacher_idx,
                    "teacher_total": len(teacher_modes),
                    "phase": "phase2",
                },
            )
            augmentor = SCMMixV2Augmentor(
                random_state=seed,
                seed_strategy=config["seed_strategy"],
                target_classes=config["target_classes"],
                treat_mix_prob=config["treat_mix_prob"],
                residual_scale=config["residual_scale"],
                teacher_mode=config["teacher_mode"],
            )
            aug_result = augmentor.generate(
                split_data.X_train_raw,
                split_data.y_train,
                hard_case_map=hard_case_map,
            )
            train_X = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
            train_y = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)
            train_result = fit_xgb_from_raw(
                train_X,
                train_y,
                split_data.X_test_raw,
                split_data.discrete_numeric_features,
                seed,
                use_adasyn=True,
            )
            metrics = evaluate_predictions(split_data.y_test, train_result.proba, TARGET_LABELS, [0, 1, 2])
            phase2_rows.append(
                {
                    "phase": "phase2_teacher",
                    "seed": seed,
                    "model_name": model_name,
                    "teacher_mode": teacher_mode,
                    "train_size": train_result.train_size,
                    "resampled_train_size": train_result.resampled_train_size,
                    "augmented_size": int(len(aug_result.X_aug)),
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
            write_partial_table(phase2_rows, tables_dir / "phase2_teacher_metrics_by_seed.partial.csv")
            tracker.log(
                (
                    f"Phase 2：完成教师模式 {teacher_idx}/{len(teacher_modes)} | "
                    f"seed={seed} | {teacher_mode} | 增广样本={len(aug_result.X_aug)}"
                ),
                stage="phase2_teacher_done",
                advance=1,
                context={
                    "seed": seed,
                    "teacher_mode": teacher_mode,
                    "teacher_index": teacher_idx,
                    "teacher_total": len(teacher_modes),
                    "model_name": model_name,
                    "augmented_size": int(len(aug_result.X_aug)),
                    "phase": "phase2",
                },
            )

    phase2_df = pd.DataFrame(phase2_rows)
    phase2_df.to_csv(tables_dir / "phase2_teacher_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    phase2_summary = aggregate_metrics(phase2_df, sorted(phase2_df["model_name"].unique().tolist())).sort_values(
        ["balanced_accuracy_mean", "macro_f1_mean", "class0_recall_mean", "accuracy_mean"],
        ascending=False,
    )
    phase2_summary.to_csv(tables_dir / "phase2_teacher_metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    plot_metric_boxplot(
        phase2_df,
        "balanced_accuracy",
        figures_dir / "phase2_teacher_balanced_accuracy_boxplot.png",
        "Phase 2 教师去偏比较",
    )

    best_phase2_name = phase2_summary.iloc[0]["model_name"]
    best_teacher_mode = phase2_df[phase2_df["model_name"] == best_phase2_name].iloc[0]["teacher_mode"]
    tracker.log(
        f"Phase 2 完成，当前最优教师模式：{best_teacher_mode}",
        stage="phase2_complete",
        context={"best_phase2_name": best_phase2_name, "best_teacher_mode": best_teacher_mode, "phase": "phase2"},
    )

    tabpfn_rows: list[dict[str, Any]] = []
    tabpfn_error_rows: list[dict[str, Any]] = []
    if tabpfn_token_available:
        tracker.log("Phase 3：检测到 TabPFN token，开始第三阶段实验", stage="phase3_start", context={"phase": "phase3"})
        for seed_idx, seed in enumerate(seeds, start=1):
            tracker.log(
                f"Phase 3：开始处理随机种子 {seed}（{seed_idx}/{len(seeds)}）",
                stage="phase3_seed_start",
                context={"seed": seed, "seed_index": seed_idx, "phase": "phase3"},
            )
            split_data = prepare_split_data(args.input, seed)
            tracker.log(
                f"Phase 3：构建 hard case 索引，seed={seed}",
                stage="phase3_hard_case",
                context={"seed": seed, "phase": "phase3"},
            )
            hard_case_map = build_hard_case_index_map(split_data.X_train_raw, split_data.y_train, seed)
            augmentor = SCMMixV2Augmentor(
                random_state=seed,
                seed_strategy=best_phase1_config["seed_strategy"],
                target_classes=best_phase1_config["target_classes"],
                treat_mix_prob=best_phase1_config["treat_mix_prob"],
                residual_scale=best_phase1_config["residual_scale"],
                teacher_mode=best_teacher_mode,
            )
            aug_result = augmentor.generate(
                split_data.X_train_raw,
                split_data.y_train,
                hard_case_map=hard_case_map,
            )
            scm_train_X = pd.concat([split_data.X_train_raw, aug_result.X_aug], axis=0, ignore_index=True)
            scm_train_y = pd.concat([split_data.y_train, aug_result.y_aug], axis=0, ignore_index=True)

            try:
                ref_proba = run_tabpfn_with_raw(
                    split_data.X_train_raw,
                    split_data.y_train,
                    split_data.X_test_raw,
                    project_dir,
                    seed,
                    status_callback=lambda msg, seed=seed: tracker.log(
                        msg,
                        stage="phase3_tabpfn_ref_retry",
                        context={"seed": seed, "model_name": "tabpfn_reference_v2", "phase": "phase3"},
                    ),
                )
                ref_metrics = evaluate_predictions(split_data.y_test, ref_proba, TARGET_LABELS, [0, 1, 2])
                tabpfn_rows.append(
                    {
                        "seed": seed,
                        "model_name": "tabpfn_reference_v2",
                        "augmented_size": 0,
                        "accuracy": ref_metrics["accuracy"],
                        "balanced_accuracy": ref_metrics["balanced_accuracy"],
                        "macro_f1": ref_metrics["macro_f1"],
                        "weighted_f1": ref_metrics["weighted_f1"],
                        "ovr_roc_auc_macro": ref_metrics["ovr_roc_auc_macro"],
                        "class0_recall": extract_class_recall(ref_metrics, "非确诊"),
                        "class1_recall": extract_class_recall(ref_metrics, "确诊"),
                        "class2_recall": extract_class_recall(ref_metrics, "灰色区域"),
                    }
                )
                write_partial_table(tabpfn_rows, tables_dir / "phase3_tabpfn_metrics_by_seed.partial.csv")
                tracker.log(
                    f"Phase 3：完成 tabpfn_reference_v2，seed={seed}",
                    stage="phase3_tabpfn_ref_done",
                    advance=1,
                    context={"seed": seed, "model_name": "tabpfn_reference_v2", "phase": "phase3"},
                )
            except Exception as exc:
                tabpfn_error_rows.append(
                    {
                        "seed": seed,
                        "model_name": "tabpfn_reference_v2",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                write_partial_table(tabpfn_error_rows, tables_dir / "phase3_tabpfn_errors.partial.csv")
                tracker.log(
                    f"Phase 3：tabpfn_reference_v2 失败，seed={seed}，错误={type(exc).__name__}",
                    stage="phase3_tabpfn_ref_failed",
                    advance=1,
                    context={
                        "seed": seed,
                        "model_name": "tabpfn_reference_v2",
                        "error_type": type(exc).__name__,
                        "phase": "phase3",
                    },
                )
                continue

            try:
                scm_proba = run_tabpfn_with_raw(
                    scm_train_X,
                    scm_train_y,
                    split_data.X_test_raw,
                    project_dir,
                    seed,
                    status_callback=lambda msg, seed=seed: tracker.log(
                        msg,
                        stage="phase3_tabpfn_scm_retry",
                        context={"seed": seed, "model_name": "tabpfn_scm_v2_best", "phase": "phase3"},
                    ),
                )
                scm_metrics = evaluate_predictions(split_data.y_test, scm_proba, TARGET_LABELS, [0, 1, 2])
                tabpfn_rows.append(
                    {
                        "seed": seed,
                        "model_name": "tabpfn_scm_v2_best",
                        "augmented_size": int(len(aug_result.X_aug)),
                        "accuracy": scm_metrics["accuracy"],
                        "balanced_accuracy": scm_metrics["balanced_accuracy"],
                        "macro_f1": scm_metrics["macro_f1"],
                        "weighted_f1": scm_metrics["weighted_f1"],
                        "ovr_roc_auc_macro": scm_metrics["ovr_roc_auc_macro"],
                        "class0_recall": extract_class_recall(scm_metrics, "非确诊"),
                        "class1_recall": extract_class_recall(scm_metrics, "确诊"),
                        "class2_recall": extract_class_recall(scm_metrics, "灰色区域"),
                    }
                )
                write_partial_table(tabpfn_rows, tables_dir / "phase3_tabpfn_metrics_by_seed.partial.csv")
                tracker.log(
                    f"Phase 3：完成 tabpfn_scm_v2_best，seed={seed}，增广样本={len(aug_result.X_aug)}",
                    stage="phase3_tabpfn_scm_done",
                    advance=1,
                    context={
                        "seed": seed,
                        "model_name": "tabpfn_scm_v2_best",
                        "augmented_size": int(len(aug_result.X_aug)),
                        "phase": "phase3",
                    },
                )
            except Exception as exc:
                tabpfn_error_rows.append(
                    {
                        "seed": seed,
                        "model_name": "tabpfn_scm_v2_best",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                write_partial_table(tabpfn_error_rows, tables_dir / "phase3_tabpfn_errors.partial.csv")
                tracker.log(
                    f"Phase 3：tabpfn_scm_v2_best 失败，seed={seed}，错误={type(exc).__name__}",
                    stage="phase3_tabpfn_scm_failed",
                    advance=1,
                    context={
                        "seed": seed,
                        "model_name": "tabpfn_scm_v2_best",
                        "error_type": type(exc).__name__,
                        "phase": "phase3",
                    },
                )
    else:
        tracker.log(
            "Phase 3：未检测到 TabPFN token，跳过第三阶段",
            stage="phase3_skipped",
            context={"phase": "phase3", "tabpfn_token_available": False},
        )

    if tabpfn_rows:
        phase3_df = pd.DataFrame(tabpfn_rows)
        phase3_df.to_csv(tables_dir / "phase3_tabpfn_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
        phase3_summary = aggregate_metrics(phase3_df, sorted(phase3_df["model_name"].unique().tolist())).sort_values(
            ["balanced_accuracy_mean", "macro_f1_mean", "class0_recall_mean", "accuracy_mean"],
            ascending=False,
        )
        phase3_summary.to_csv(tables_dir / "phase3_tabpfn_metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    else:
        phase3_summary = pd.DataFrame()
    if tabpfn_error_rows:
        pd.DataFrame(tabpfn_error_rows).to_csv(
            tables_dir / "phase3_tabpfn_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "seeds": seeds,
        "input_file": str(args.input),
        "smoke_test": args.smoke_test,
        "phase1_best_config_name": best_phase1_name,
        "phase1_best_config": best_phase1_config,
        "phase2_best_teacher_name": best_phase2_name,
        "phase2_best_teacher_mode": best_teacher_mode,
        "tabpfn_token_available": tabpfn_token_available,
        "tabpfn_phase_completed": bool(tabpfn_rows),
        "tabpfn_error_count": len(tabpfn_error_rows),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tracker.log(
        "全部阶段完成，实验摘要已写入 experiment_summary.json",
        stage="complete",
        context=summary,
    )

    print("SCM-v2 与教师去偏实验完成")
    print("Phase 1 最优配置:", best_phase1_name)
    print("Phase 2 最优教师:", best_phase2_name)
    print("TabPFN token 可用:", tabpfn_token_available)
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
