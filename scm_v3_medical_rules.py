from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CORE_NON_NEGATIVE_COLUMNS = [
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
]

PHYSIOLOGY_COLUMNS = [
    "ARR比值",
    "醛固酮",
    "肾素",
    "收缩压",
    "舒展压",
    "钾",
    "钠",
    "氯",
    "肌酐",
]

TREATMENT_COLUMNS = [
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
]


@dataclass
class RuleDecision:
    is_valid: bool
    reasons: list[str]
    failed_rule_ids: list[str]


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _support_bounds(
    support: pd.Series,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    margin_ratio: float = 0.08,
) -> tuple[float, float] | None:
    numeric = pd.to_numeric(support, errors="coerce").dropna()
    if numeric.empty:
        return None
    lower = float(numeric.quantile(lower_q))
    upper = float(numeric.quantile(upper_q))
    width = max(upper - lower, 1e-6)
    margin = width * margin_ratio
    return lower - margin, upper + margin


def evaluate_medical_rules(
    candidate_row: pd.Series,
    reference_df: pd.DataFrame,
    *,
    allow_missing_fraction: float = 0.35,
) -> RuleDecision:
    reasons: list[str] = []
    failed_rule_ids: list[str] = []

    physiology_cols = [col for col in PHYSIOLOGY_COLUMNS if col in candidate_row.index]
    if physiology_cols:
        missing_fraction = float(candidate_row[physiology_cols].isna().mean())
        if missing_fraction > allow_missing_fraction:
            failed_rule_ids.append("rule_missing_fraction")
            reasons.append(f"关键生理变量缺失比例过高: {missing_fraction:.2f}")

    for col in CORE_NON_NEGATIVE_COLUMNS:
        if col not in candidate_row.index:
            continue
        numeric = _to_float(candidate_row[col])
        if numeric is None:
            continue
        if numeric < 0:
            failed_rule_ids.append(f"rule_non_negative::{col}")
            reasons.append(f"{col} 小于 0")

    for col in physiology_cols + [c for c in TREATMENT_COLUMNS if c in candidate_row.index]:
        if col not in reference_df.columns:
            continue
        numeric = _to_float(candidate_row[col])
        if numeric is None:
            continue
        bounds = _support_bounds(reference_df[col])
        if bounds is None:
            continue
        lower, upper = bounds
        if numeric < lower or numeric > upper:
            failed_rule_ids.append(f"rule_support_bounds::{col}")
            reasons.append(f"{col} 超出支持范围 [{lower:.3f}, {upper:.3f}]")

    sbp = _to_float(candidate_row.get("收缩压"))
    dbp = _to_float(candidate_row.get("舒展压"))
    if sbp is not None and dbp is not None and sbp < dbp:
        failed_rule_ids.append("rule_bp_order")
        reasons.append("收缩压小于舒展压")

    arr = _to_float(candidate_row.get("ARR比值"))
    renin = _to_float(candidate_row.get("肾素"))
    aldosterone = _to_float(candidate_row.get("醛固酮"))
    if arr is not None and renin is not None and aldosterone is not None:
        if renin > 1e-6:
            implied_arr = aldosterone / renin
            denom = max(abs(implied_arr), abs(arr), 1.0)
            relative_gap = abs(arr - implied_arr) / denom
            if relative_gap > 0.85:
                failed_rule_ids.append("rule_arr_consistency")
                reasons.append(f"ARR 与 醛固酮/肾素 比值明显不一致: gap={relative_gap:.2f}")

    drug_total = _to_float(candidate_row.get("联合用药_总数"))
    if drug_total is not None:
        component_scores: list[float] = []
        for col in TREATMENT_COLUMNS:
            if col == "联合用药_总数" or col not in candidate_row.index:
                continue
            numeric = _to_float(candidate_row[col])
            if numeric is None:
                continue
            component_scores.append(numeric)
        if component_scores:
            non_zero_components = sum(float(value > 1e-8) for value in component_scores)
            if drug_total + 1e-8 < non_zero_components - 1:
                failed_rule_ids.append("rule_treatment_count")
                reasons.append("联合用药总数与各药物等效分数数量明显冲突")

    return RuleDecision(
        is_valid=not failed_rule_ids,
        reasons=reasons,
        failed_rule_ids=failed_rule_ids,
    )
