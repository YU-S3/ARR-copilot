from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from tablegpt2_pa.common import (
    TARGET_COLUMN,
    build_binary_dataset,
    compute_binary_metrics,
    ensure_directory,
    infer_feature_types,
    load_clean_frame,
    rank_features,
    split_with_icl_pools,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="运行 PA 二分类对照实验。")
    parser.add_argument("--input", type=Path, default=project_dir / "数据表格测试.xlsx")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "tablegpt2_pa_outputs" / "comparison_baselines",
    )
    parser.add_argument("--protocol", choices=["A", "B", "C"], default="A")
    parser.add_argument("--top-p", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_baseline_preprocessor(frame: pd.DataFrame, label_column: str) -> tuple[ColumnTransformer, list[str]]:
    excluded = {TARGET_COLUMN, "_target_clean", "_sample_id", label_column}
    numeric_features, categorical_features = infer_feature_types(frame, excluded)
    candidate_columns = [col for col in frame.columns if col not in excluded]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return preprocessor, candidate_columns


def train_and_score(
    name: str,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    selected_features: list[str],
) -> dict[str, Any]:
    X_train = train_df[selected_features].copy()
    y_train = train_df[label_column].astype(int).copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df[label_column].astype(int).copy()

    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in selected_features if col not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_binary_metrics(y_test, y_pred, y_prob)
    metrics["model_name"] = name
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    frame = load_clean_frame(args.input)
    dataset = build_binary_dataset(frame, args.protocol)
    splits = split_with_icl_pools(dataset.frame, dataset.label_column, args.random_state)
    selected_features = rank_features(splits.train_df, dataset.label_column, top_p=args.top_p)

    rf_model = RandomForestClassifier(
        n_estimators=400,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
    )

    results = [
        train_and_score(
            "RandomForest",
            rf_model,
            splits.train_df,
            splits.test_df,
            dataset.label_column,
            selected_features,
        ),
        train_and_score(
            "XGBoost",
            xgb_model,
            splits.train_df,
            splits.test_df,
            dataset.label_column,
            selected_features,
        ),
    ]
    summary = {
        "input_file": str(args.input),
        "protocol": args.protocol,
        "target_column": TARGET_COLUMN,
        "selected_features": selected_features,
        "results": results,
    }
    pd.DataFrame(results).to_csv(output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

