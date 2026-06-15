from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "确诊（0为排除；1为确诊；2为灰色区域）"
ZERO_FILL_COLUMNS = [
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
]
EXPERIMENT_TYPE_COLUMN = "确诊实验类型"
MEDICAL_PRIORITY_FEATURES = [
    "ARR比值",
    "立位醛固酮",
    "肾素",
    "钾",
    "钠",
    "收缩压",
    "舒展压",
    "RASS_等效分数",
    "利尿剂_等效分数",
    "Beta_等效分数",
    "是否有肾上腺结节",
    "是否有增生",
    "结节 最大直径",
    "ARR比值>192为阳性，推荐进行确诊试验",
    "确诊实验类型",
]
PROMPT_INSTRUCTIONS = """你是一个用于原发性醛固酮增多症（PA）筛查与确诊辅助判断的表格诊断模型。
请根据历史带标签病例和当前患者的结构化检查结果，判断当前患者是否为 PA。

标签定义：
- 0：非 PA
- 1：PA

请重点关注：
- ARR 比值与醛固酮/肾素关系
- 电解质异常，尤其是血钾
- 血压水平
- 药物暴露对 ARR 和肾素的干扰
- 影像学提示的肾上腺结节或增生

请严格输出 JSON，不要输出任何额外文本。
JSON 格式如下：
{
  "label": 0,
  "probability": 0.50,
  "reasoning": "简要说明依据"
}
"""


@dataclass
class BinaryDataset:
    frame: pd.DataFrame
    label_column: str
    protocol: str
    gray_zone_frame: pd.DataFrame | None = None


@dataclass
class PromptExample:
    sample_id: str
    prompt: str
    response: str
    label: int


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    icl_train_df: pd.DataFrame
    icl_val_df: pd.DataFrame
    icl_test_df: pd.DataFrame


def clean_string_cell(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "nan", "None", "?", "？"}:
        return np.nan
    return text


def normalize_target_value(value: Any) -> float | None:
    value = clean_string_cell(value)
    if pd.isna(value):
        return None
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return None
    if numeric in {0, 1, 2}:
        return float(numeric)
    return None


def normalize_experiment_type(value: Any) -> Any:
    value = clean_string_cell(value)
    if pd.isna(value):
        return np.nan
    mapping = {
        "冷盐水": "冷盐水实验",
        "冷盐水实验": "冷盐水实验",
        "卡托普利": "卡托普利实验",
        "卡托普利实验": "卡托普利实验",
    }
    return mapping.get(value, value)


def maybe_convert_object_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(clean_string_cell)
    if cleaned.dropna().empty:
        return cleaned
    converted = pd.to_numeric(cleaned, errors="coerce")
    ratio = converted.notna().sum() / max(cleaned.notna().sum(), 1)
    if ratio >= 0.85:
        return converted
    return cleaned


def load_clean_frame(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    df = pd.read_excel(input_path).copy()
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"未找到目标列: {TARGET_COLUMN}")

    missing_zero_fill_cols = [col for col in ZERO_FILL_COLUMNS if col not in df.columns]
    if missing_zero_fill_cols:
        raise KeyError(f"以下需要补 0 的列不存在: {missing_zero_fill_cols}")

    df[ZERO_FILL_COLUMNS] = df[ZERO_FILL_COLUMNS].fillna(0)
    if EXPERIMENT_TYPE_COLUMN in df.columns:
        df[EXPERIMENT_TYPE_COLUMN] = df[EXPERIMENT_TYPE_COLUMN].map(normalize_experiment_type)

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if df[col].dtype == "object":
            df[col] = maybe_convert_object_to_numeric(df[col])

    df["_target_clean"] = df[TARGET_COLUMN].apply(normalize_target_value)
    df["_sample_id"] = [f"sample_{i:04d}" for i in range(len(df))]
    return df


def build_binary_dataset(frame: pd.DataFrame, protocol: str) -> BinaryDataset:
    protocol = protocol.upper()
    valid = frame[frame["_target_clean"].isin([0.0, 1.0, 2.0])].copy()

    if protocol == "A":
        binary = valid[valid["_target_clean"].isin([0.0, 1.0])].copy()
        binary["binary_label"] = binary["_target_clean"].astype(int)
        return BinaryDataset(frame=binary, label_column="binary_label", protocol=protocol)

    if protocol == "B":
        binary = valid.copy()
        binary["binary_label"] = (binary["_target_clean"] == 1.0).astype(int)
        return BinaryDataset(frame=binary, label_column="binary_label", protocol=protocol)

    if protocol == "C":
        binary = valid[valid["_target_clean"].isin([0.0, 1.0])].copy()
        gray = valid[valid["_target_clean"] == 2.0].copy()
        binary["binary_label"] = binary["_target_clean"].astype(int)
        return BinaryDataset(
            frame=binary,
            label_column="binary_label",
            protocol=protocol,
            gray_zone_frame=gray,
        )

    raise ValueError("protocol 仅支持 A / B / C。")


def split_with_icl_pools(
    df: pd.DataFrame,
    label_column: str,
    random_state: int,
) -> SplitBundle:
    if df[label_column].nunique() < 2:
        raise ValueError("二分类标签不足，无法切分。")
    if len(df) < 30:
        raise ValueError("样本量过少，无法稳定切分 train/val/test/ICL。")

    train_df, rest_df = train_test_split(
        df,
        test_size=0.60,
        random_state=random_state,
        stratify=df[label_column],
    )
    val_df, rest_df = train_test_split(
        rest_df,
        test_size=5 / 6,
        random_state=random_state,
        stratify=rest_df[label_column],
    )
    test_df, rest_df = train_test_split(
        rest_df,
        test_size=0.50,
        random_state=random_state,
        stratify=rest_df[label_column],
    )
    # 将 30% 余量切为三个不重叠 ICL 池。
    icl_train_df, holdout_df = train_test_split(
        rest_df,
        test_size=2 / 3,
        random_state=random_state,
        stratify=rest_df[label_column],
    )
    icl_val_df, icl_test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=random_state,
        stratify=holdout_df[label_column],
    )
    return SplitBundle(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        icl_train_df=icl_train_df.reset_index(drop=True),
        icl_val_df=icl_val_df.reset_index(drop=True),
        icl_test_df=icl_test_df.reset_index(drop=True),
    )


def infer_feature_types(frame: pd.DataFrame, exclude: Iterable[str]) -> tuple[list[str], list[str]]:
    exclude_set = set(exclude)
    feature_df = frame[[col for col in frame.columns if col not in exclude_set]].copy()
    numeric_features = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in feature_df.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_feature_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def rank_features(
    train_df: pd.DataFrame,
    label_column: str,
    top_p: int,
) -> list[str]:
    excluded = {TARGET_COLUMN, "_target_clean", "_sample_id", label_column}
    candidate_columns = [col for col in train_df.columns if col not in excluded]
    numeric_features, categorical_features = infer_feature_types(train_df, excluded)
    scores: dict[str, float] = {feature: 0.0 for feature in candidate_columns}

    for rank, feature in enumerate(MEDICAL_PRIORITY_FEATURES[::-1], start=1):
        if feature in scores:
            scores[feature] += 1000.0 - rank

    if candidate_columns:
        preprocessor = build_feature_preprocessor(numeric_features, categorical_features)
        transformed = preprocessor.fit_transform(train_df[candidate_columns])
        transformed = np.asarray(transformed, dtype=float)
        mi_scores = mutual_info_classif(
            transformed,
            train_df[label_column].astype(int),
            discrete_features=False,
            random_state=42,
        )
        feature_names = preprocessor.get_feature_names_out()
        raw_feature_scores: dict[str, float] = {feature: 0.0 for feature in candidate_columns}
        for feature_name, score in zip(feature_names, mi_scores):
            raw_name = feature_name.split("__", 1)[1]
            raw_name = raw_name.split("_missingindicator_", 1)[0]
            raw_name = raw_name.split("_", 1)[0] if raw_name not in raw_feature_scores else raw_name
            if raw_name in raw_feature_scores:
                raw_feature_scores[raw_name] = max(raw_feature_scores[raw_name], float(score))
        for feature, score in raw_feature_scores.items():
            scores[feature] += score

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    selected = [feature for feature, _ in ranked[:top_p]]
    if len(selected) < min(top_p, len(candidate_columns)):
        seen = set(selected)
        for column in candidate_columns:
            if column not in seen:
                selected.append(column)
            if len(selected) >= top_p:
                break
    return selected


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4g}"
    return str(value)


def rows_to_markdown_table(frame: pd.DataFrame, features: list[str], label_column: str) -> str:
    columns = features + [label_column]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(format_value(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *rows])


def build_prompt(
    demo_df: pd.DataFrame,
    query_row: pd.Series,
    features: list[str],
    label_column: str,
) -> str:
    demo_table = rows_to_markdown_table(demo_df, features, label_column)
    query_df = pd.DataFrame([{**{col: query_row[col] for col in features}, label_column: "?"}])
    query_table = rows_to_markdown_table(query_df, features, label_column)
    return (
        f"{PROMPT_INSTRUCTIONS}\n\n"
        f"历史样本：\n\n{demo_table}\n\n"
        f"当前患者：\n\n{query_table}\n\n"
        "Let's think step by step."
    )


def response_json(label: int, row: pd.Series) -> str:
    arr_value = format_value(row.get("ARR比值", np.nan))
    ald_value = format_value(row.get("立位醛固酮", np.nan))
    renin_value = format_value(row.get("肾素", np.nan))
    potassium = format_value(row.get("钾", np.nan))
    reasoning = (
        f"ARR比值为{arr_value}，立位醛固酮为{ald_value}，肾素为{renin_value}，"
        f"血钾为{potassium}，综合这些指标判断为{'PA' if label == 1 else '非PA'}。"
    )
    payload = {
        "label": int(label),
        "probability": 0.9 if int(label) == 1 else 0.1,
        "reasoning": reasoning,
    }
    return json.dumps(payload, ensure_ascii=False)


def stratified_demo_sample(
    pool_df: pd.DataFrame,
    label_column: str,
    k_shot: int,
    random_state: int,
) -> pd.DataFrame:
    if len(pool_df) <= k_shot:
        return pool_df.copy()

    rng = random.Random(random_state)
    class_groups = {
        label: group.sample(frac=1.0, random_state=random_state)
        for label, group in pool_df.groupby(label_column)
    }
    selected_rows: list[pd.DataFrame] = []
    while sum(len(group) for group in selected_rows) < k_shot:
        progressed = False
        for label in sorted(class_groups):
            group = class_groups[label]
            if group.empty:
                continue
            idx = rng.randrange(len(group))
            selected_rows.append(group.iloc[[idx]])
            class_groups[label] = group.drop(group.index[idx])
            progressed = True
            if sum(len(g) for g in selected_rows) >= k_shot:
                break
        if not progressed:
            break
    sampled = pd.concat(selected_rows, ignore_index=True)
    return sampled.head(k_shot)


def build_prompt_examples(
    frame: pd.DataFrame,
    icl_pool: pd.DataFrame,
    features: list[str],
    label_column: str,
    k_shot: int,
    seed: int,
) -> list[PromptExample]:
    examples: list[PromptExample] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        demo_df = stratified_demo_sample(
            icl_pool,
            label_column=label_column,
            k_shot=k_shot,
            random_state=seed + idx,
        )
        prompt = build_prompt(demo_df, row, features, label_column)
        response = response_json(int(row[label_column]), row)
        examples.append(
            PromptExample(
                sample_id=str(row["_sample_id"]),
                prompt=prompt,
                response=response,
                label=int(row[label_column]),
            )
        )
    return examples


def parse_json_response(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("未在模型输出中找到 JSON 对象。")
    return json.loads(match.group(0))


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
    }
    if y_prob is not None and len(set(y_true_array.tolist())) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, np.asarray(y_prob, dtype=float)))
        except ValueError:
            pass
    return metrics


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
