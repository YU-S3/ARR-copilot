from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
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


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    default_input = project_dir / "data_0428.xlsx"
    default_output = project_dir / "outputs"

    parser = argparse.ArgumentParser(
        description="使用 Excel 数据训练三分类随机森林模型，并导出评估结果与预测结果。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"输入 Excel 路径，默认: {default_input}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"输出目录，默认: {default_output}",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="工作表名称或索引，默认读取第一个工作表。",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集占比，默认 0.2。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，默认 42。",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="随机森林树的数量，默认 500。",
    )
    return parser.parse_args()


def normalize_target_value(value: object) -> float | None:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text in {"0", "1", "2"}:
            return float(text)
        try:
            numeric = float(text)
        except ValueError:
            return None
        if numeric in {0.0, 1.0, 2.0}:
            return numeric
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric in {0.0, 1.0, 2.0}:
        return numeric
    return None


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int,
    random_state: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            # Preserve missingness information with indicator columns while allowing RF to train.
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ]
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input, sheet_name=args.sheet_name)
    df = df.copy()

    missing_zero_fill_cols = [col for col in ZERO_FILL_COLUMNS if col not in df.columns]
    if missing_zero_fill_cols:
        raise KeyError(f"以下需要补 0 的列不存在: {missing_zero_fill_cols}")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"未找到目标列: {TARGET_COLUMN}")

    excluded_columns = [
        col
        for col in df.columns
        if col in {"住院号", "Unnamed: 0"} or str(col).startswith("Unnamed:")
    ]
    if excluded_columns:
        df = df.drop(columns=excluded_columns)

    df[ZERO_FILL_COLUMNS] = df[ZERO_FILL_COLUMNS].fillna(0)
    df["_target_clean"] = df[TARGET_COLUMN].apply(normalize_target_value)

    feature_df = df.drop(columns=[TARGET_COLUMN, "_target_clean"])
    train_mask = df["_target_clean"].isin([0.0, 1.0, 2.0])

    X_train_all = feature_df.loc[train_mask].copy()
    y_train_all = df.loc[train_mask, "_target_clean"].astype(int).copy()

    if y_train_all.nunique() < 3:
        raise ValueError("可用于训练的 0/1/2 标签不足，无法进行三分类训练。")
    if len(X_train_all) < 10:
        raise ValueError("可用于训练的样本量过少。")

    numeric_features = X_train_all.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X_train_all.columns if col not in numeric_features]
    if categorical_features:
        for col in categorical_features:
            feature_df[col] = feature_df[col].map(lambda value: value if pd.isna(value) else str(value))
        X_train_all = feature_df.loc[train_mask].copy()

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_all,
        y_train_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_train_all,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = pipeline.named_steps["model"].classes_.tolist()
    class_index = {int(cls): idx for idx, cls in enumerate(classes)}

    metrics = {
        "input_file": str(args.input),
        "sheet_name": str(args.sheet_name),
        "total_rows": int(len(df)),
        "trainable_rows": int(train_mask.sum()),
        "excluded_rows_missing_or_invalid_target": int((~train_mask).sum()),
        "target_distribution_trainable": {
            str(k): int(v) for k, v in y_train_all.value_counts().sort_index().items()
        },
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "classes": classes,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "ovr_roc_auc_macro": float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, labels=classes, zero_division=0, output_dict=True
        ),
    }

    all_pred = pipeline.predict(feature_df)
    all_proba = pipeline.predict_proba(feature_df)

    predictions_df = df.drop(columns=["_target_clean"]).copy()
    predictions_df["训练是否纳入"] = train_mask.map({True: "是", False: "否"})
    predictions_df["模型预测_类别"] = all_pred.astype(int)
    predictions_df["模型预测_类别名称"] = predictions_df["模型预测_类别"].map(
        {0: "非确诊", 1: "确诊", 2: "灰色区域"}
    )
    for cls in classes:
        cls_int = int(cls)
        cls_name = {0: "非确诊", 1: "确诊", 2: "灰色区域"}[cls_int]
        predictions_df[f"模型预测_概率_{cls_int}_{cls_name}"] = all_proba[:, class_index[cls_int]]

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    model_path = args.output_dir / "random_forest_model.joblib"
    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "predictions.xlsx"
    importance_path = args.output_dir / "feature_importance.csv"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_df.to_excel(predictions_path, index=False)
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print("训练完成")
    print(f"输入文件: {args.input}")
    print(f"总样本数: {len(df)}")
    print(f"用于训练的 0/1/2 样本数: {int(train_mask.sum())}")
    print(f"排除的缺失/非法标签样本数: {int((~train_mask).sum())}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"OVR ROC AUC Macro: {metrics['ovr_roc_auc_macro']:.4f}")
    print(f"模型文件: {model_path}")
    print(f"指标文件: {metrics_path}")
    print(f"预测结果: {predictions_path}")
    print(f"特征重要性: {importance_path}")


if __name__ == "__main__":
    main()
