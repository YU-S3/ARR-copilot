from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from causal_xgboost_variants_experiment import XGB_BEST_PARAMS
from multiclass_ensemble_experiment import build_preprocessor, transform_with_preprocessor


RANDOM_STATE = 42

ANCHOR_COLUMNS = [
    "年龄",
    "性别",
    "确诊实验类型",
    "是否有肾上腺结节",
    "是否有增生",
    "结节最大直径",
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
]

REPAIR_COLUMNS = [
    "ARR比值",
    "醛固酮",
    "肾素",
    "试验前醛固酮",
    "试验前肾素",
    "试验后醛固酮",
    "试验后肾素",
    "收缩压",
    "舒展压",
    "钾",
    "钠",
    "氯",
    "肌酐",
    "血红蛋白",
    "白细胞",
    "血小板",
    "谷草",
    "谷丙",
]

EXOGENOUS_COLUMNS = ["年龄", "性别", "是否有肾上腺结节", "是否有增生", "结节最大直径"]
TREATMENT_COLUMNS = [
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
]
SCREENING_COLUMNS = ["醛固酮", "肾素", "ARR比值", "收缩压", "舒展压", "钾", "钠", "氯"]
POST_TREATMENT_COLUMNS = [
    "试验前醛固酮",
    "试验前肾素",
    "试验后醛固酮",
    "试验后肾素",
]


@dataclass
class TeacherArtifacts:
    preprocessor: Any
    model: XGBClassifier
    feature_names: list[str]


@dataclass
class AugmentationResult:
    X_aug: pd.DataFrame
    y_aug: pd.Series
    audit: pd.DataFrame
    metadata: pd.DataFrame


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def infer_discrete_numeric_features(df: pd.DataFrame) -> list[str]:
    discrete_cols: list[str] = []
    numeric_df = df.select_dtypes(include=["number", "bool"])
    for col in numeric_df.columns:
        values = numeric_df[col].dropna()
        if values.empty:
            continue
        unique_count = values.nunique()
        if unique_count <= 12 or np.allclose(values, np.round(values)):
            discrete_cols.append(col)
    return discrete_cols


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


def make_xgb_regressor(seed: int = RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_estimators=140,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=0.1,
        n_jobs=-1,
    )


def fit_teacher_model(X_train_raw: pd.DataFrame, y_train: pd.Series) -> TeacherArtifacts:
    numeric_features = X_train_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_raw.columns if c not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, _ = transform_with_preprocessor(preprocessor, X_train_raw, None)
    model = make_xgb_classifier()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_df, y_train, sample_weight=sample_weight, verbose=False)
    return TeacherArtifacts(
        preprocessor=preprocessor,
        model=model,
        feature_names=list(X_train_df.columns),
    )


def transform_raw_with_teacher(teacher: TeacherArtifacts, X_raw: pd.DataFrame) -> pd.DataFrame:
    transformed = teacher.preprocessor.transform(X_raw)
    return pd.DataFrame(transformed, columns=teacher.feature_names, index=X_raw.index)


def clip_numeric_value(value: float, support: pd.Series) -> float:
    values = support.dropna()
    if values.empty:
        return float(value)
    lower = float(values.quantile(0.01))
    upper = float(values.quantile(0.99))
    return float(np.clip(value, lower, upper))


def bootstrap_value(rng: np.random.Generator, support: pd.Series) -> Any:
    values = support.dropna()
    if values.empty:
        return np.nan
    return values.iloc[int(rng.integers(0, len(values)))]


def compute_anchor_distance(seed_row: pd.Series, candidate_row: pd.Series, anchor_columns: list[str]) -> float:
    distances: list[float] = []
    for col in anchor_columns:
        if col not in seed_row.index or col not in candidate_row.index:
            continue
        seed_val = seed_row[col]
        cand_val = candidate_row[col]
        if pd.isna(seed_val) and pd.isna(cand_val):
            distances.append(0.0)
        elif pd.api.types.is_numeric_dtype(pd.Series([seed_val, cand_val])):
            if pd.isna(seed_val) or pd.isna(cand_val):
                distances.append(1.0)
            else:
                distances.append(float(abs(seed_val - cand_val)))
        else:
            distances.append(0.0 if str(seed_val) == str(cand_val) else 1.0)
    if not distances:
        return 0.0
    return float(np.mean(distances))


def summarize_audit(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    y_aug: pd.Series,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if augmented_df.empty:
        return pd.DataFrame(
            [
                {
                    "row_type": "summary",
                    "feature": "no_augmented_rows",
                    "value": 0,
                }
            ]
        )

    counts = y_aug.value_counts().sort_index().to_dict()
    for cls, count in counts.items():
        rows.append({"row_type": "class_count", "feature": f"class_{cls}", "value": int(count)})

    if "distance_to_seed" in metadata.columns:
        rows.append(
            {
                "row_type": "summary",
                "feature": "mean_distance_to_seed",
                "value": float(metadata["distance_to_seed"].mean()),
            }
        )
    if "teacher_target_proba" in metadata.columns:
        rows.append(
            {
                "row_type": "summary",
                "feature": "mean_teacher_target_proba",
                "value": float(metadata["teacher_target_proba"].mean()),
            }
        )

    numeric_columns = original_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    for col in numeric_columns:
        if col not in augmented_df.columns:
            continue
        orig = pd.to_numeric(original_df[col], errors="coerce")
        aug = pd.to_numeric(augmented_df[col], errors="coerce")
        orig_valid = orig.dropna()
        aug_valid = aug.dropna()
        if orig_valid.empty or aug_valid.empty:
            continue
        lower = float(orig_valid.min())
        upper = float(orig_valid.max())
        out_of_support = ((aug_valid < lower) | (aug_valid > upper)).mean()
        mean_shift = float(aug_valid.mean() - orig_valid.mean())
        std_shift = float(aug_valid.std(ddof=0) - orig_valid.std(ddof=0))
        rows.append(
            {
                "row_type": "feature_stats",
                "feature": col,
                "mean_shift": mean_shift,
                "std_shift": std_shift,
                "out_of_support_rate": float(out_of_support),
            }
        )
    return pd.DataFrame(rows)


class TapInspiredInpaintingAugmentor:
    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state

    def _target_counts(self, y_train: pd.Series) -> dict[int, int]:
        counts = y_train.value_counts().to_dict()
        if not counts:
            return {}
        majority_count = int(max(counts.values()))
        target_counts: dict[int, int] = {}
        for cls, count in counts.items():
            max_ratio = 0.85 if int(cls) == 0 else 0.95
            desired = min(int(np.ceil(majority_count * max_ratio)), int(count * 5))
            if desired > count:
                target_counts[int(cls)] = desired
        return target_counts

    def _build_anchor_matrix(self, class_df: pd.DataFrame, anchor_columns: list[str]) -> np.ndarray:
        if not anchor_columns:
            return np.zeros((len(class_df), 1), dtype=float)
        anchor_df = class_df[anchor_columns].copy()
        for col in anchor_df.columns:
            if pd.api.types.is_numeric_dtype(anchor_df[col]):
                fill_value = anchor_df[col].median()
                anchor_df[col] = anchor_df[col].fillna(fill_value)
            else:
                anchor_df[col] = pd.Categorical(anchor_df[col].fillna("Missing")).codes
        return anchor_df.to_numpy(dtype=float)

    def generate(self, X_train_raw: pd.DataFrame, y_train: pd.Series) -> AugmentationResult:
        rng = np.random.default_rng(self.random_state)
        teacher = fit_teacher_model(X_train_raw, y_train)
        discrete_features = set(infer_discrete_numeric_features(X_train_raw))
        anchor_columns = existing_columns(X_train_raw, ANCHOR_COLUMNS)
        repair_columns = existing_columns(X_train_raw, REPAIR_COLUMNS)
        target_counts = self._target_counts(y_train)
        augmented_rows: list[pd.Series] = []
        metadata_rows: list[dict[str, Any]] = []

        for cls, desired_total in target_counts.items():
            class_mask = y_train == cls
            class_df = X_train_raw.loc[class_mask].copy()
            current_count = int(class_mask.sum())
            need = max(0, desired_total - current_count)
            if need == 0 or class_df.empty:
                continue

            anchor_matrix = self._build_anchor_matrix(class_df, anchor_columns)
            neighbor_k = int(min(max(3, len(class_df) - 1), 5))
            attempts = 0
            max_attempts = max(need * 30, 60)
            accepted_for_class = 0

            while accepted_for_class < need and attempts < max_attempts:
                attempts += 1
                seed_idx = int(rng.integers(0, len(class_df)))
                seed_row = class_df.iloc[seed_idx].copy()
                candidate = seed_row.copy()
                if len(class_df) > 1:
                    distances = pairwise_distances(
                        anchor_matrix[[seed_idx]],
                        anchor_matrix,
                        metric="euclidean",
                    ).ravel()
                    order = np.argsort(distances)
                    neighbor_positions = [pos for pos in order if pos != seed_idx][:neighbor_k]
                    neighbors = class_df.iloc[neighbor_positions] if neighbor_positions else class_df
                else:
                    neighbors = class_df

                for col in repair_columns:
                    support = neighbors[col] if col in neighbors.columns else class_df[col]
                    valid = pd.to_numeric(support, errors="coerce").dropna()
                    if valid.empty:
                        continue
                    mean = float(valid.mean())
                    std = float(valid.std(ddof=0))
                    if not np.isfinite(std) or std == 0:
                        sampled = mean
                    else:
                        sampled = float(rng.normal(mean, max(std * 0.35, 1e-6)))
                    sampled = clip_numeric_value(sampled, pd.to_numeric(class_df[col], errors="coerce"))
                    if col in discrete_features:
                        sampled = float(np.round(sampled))
                    candidate[col] = sampled

                candidate_df = pd.DataFrame([candidate])
                candidate_transformed = transform_raw_with_teacher(teacher, candidate_df)
                proba = teacher.model.predict_proba(candidate_transformed)[0]
                pred = int(np.argmax(proba))
                target_proba = float(proba[int(cls)])
                distance_to_seed = compute_anchor_distance(seed_row, candidate, anchor_columns)
                score = target_proba - 0.05 * distance_to_seed + (0.05 if int(cls) == 0 else 0.0)

                if pred != int(cls) or target_proba < 0.40:
                    continue

                augmented_rows.append(candidate)
                metadata_rows.append(
                    {
                        "method": "tap_proxy",
                        "target_class": int(cls),
                        "teacher_target_proba": target_proba,
                        "teacher_pred_class": pred,
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


class SCMMixAugmentor:
    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state

    def _target_counts(self, y_train: pd.Series) -> dict[int, int]:
        counts = y_train.value_counts().to_dict()
        if not counts:
            return {}
        majority_count = int(max(counts.values()))
        target_counts: dict[int, int] = {}
        for cls, count in counts.items():
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
    ) -> Any:
        if model_bundle is None:
            return bootstrap_value(rng, support)
        X_raw = pd.DataFrame([row[model_bundle["parents"]].to_dict()])
        transformed = model_bundle["preprocessor"].transform(X_raw)
        X_df = pd.DataFrame(transformed, columns=model_bundle["feature_names"], index=X_raw.index)
        pred = float(model_bundle["model"].predict(X_df)[0])
        residuals = model_bundle["residuals"]
        if residuals.size > 0:
            pred += float(residuals[int(rng.integers(0, residuals.size))])
        return clip_numeric_value(pred, pd.to_numeric(support, errors="coerce"))

    def generate(self, X_train_raw: pd.DataFrame, y_train: pd.Series) -> AugmentationResult:
        rng = np.random.default_rng(self.random_state)
        teacher = fit_teacher_model(X_train_raw, y_train)
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
            max_attempts = max(need * 30, 60)
            accepted_for_class = 0

            while accepted_for_class < need and attempts < max_attempts:
                attempts += 1
                base_row = class_df.iloc[int(rng.integers(0, len(class_df)))].copy()
                donor_row = X_train_raw.iloc[int(rng.integers(0, len(X_train_raw)))].copy()
                candidate = base_row.copy()

                for col in exogenous_columns:
                    candidate[col] = base_row[col]

                for col in treatment_columns:
                    if rng.random() < 0.45:
                        candidate[col] = donor_row[col]
                    else:
                        candidate[col] = base_row[col]
                    support = pd.to_numeric(class_df[col], errors="coerce")
                    if col in discrete_features:
                        candidate[col] = float(np.round(pd.to_numeric(pd.Series([candidate[col]]), errors="coerce").iloc[0]))
                    if pd.api.types.is_numeric_dtype(pd.Series([candidate[col]])):
                        candidate[col] = clip_numeric_value(
                            pd.to_numeric(pd.Series([candidate[col]]), errors="coerce").iloc[0],
                            support,
                        )

                for col in screening_columns:
                    generated = self._generate_node_value(
                        rng,
                        screening_models.get(col),
                        candidate,
                        X_train_raw[col],
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
                    )
                    if col in discrete_features and pd.notna(generated):
                        generated = float(np.round(generated))
                    candidate[col] = generated

                candidate_df = pd.DataFrame([candidate])
                candidate_transformed = transform_raw_with_teacher(teacher, candidate_df)
                proba = teacher.model.predict_proba(candidate_transformed)[0]
                pred = int(np.argmax(proba))
                target_proba = float(proba[int(cls)])
                changed_treatment_count = sum(
                    int(str(candidate[col]) != str(base_row[col])) for col in treatment_columns
                )
                score = target_proba + 0.03 * changed_treatment_count + (0.05 if int(cls) == 0 else 0.0)

                if pred != int(cls) or target_proba < 0.55:
                    continue

                augmented_rows.append(candidate)
                metadata_rows.append(
                    {
                        "method": "scm_proxy",
                        "target_class": int(cls),
                        "teacher_target_proba": target_proba,
                        "teacher_pred_class": pred,
                        "changed_treatment_count": changed_treatment_count,
                        "distance_to_seed": compute_anchor_distance(base_row, candidate, exogenous_columns + treatment_columns),
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
