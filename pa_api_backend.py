from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from multiclass_ensemble_experiment import DEFAULT_INPUT_FILE, ZERO_FILL_COLUMNS, normalize_experiment_type, prepare_data
from screening_0428_experiment import apply_feature_policy, task_target


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / DEFAULT_INPUT_FILE
ARTIFACT_DIR = PROJECT_DIR / "pa_backend_artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_bin_d3_l20_bal_isotonic.joblib"

MODEL_VERSION = "xgb_bin_d3_l20_bal_isotonic"
MODEL_KIND = "binary screening XGBoost + isotonic calibration + fixed threshold modes"
DEFAULT_SEED = 42
CALIBRATION_SIZE = 0.25
SCREENING_POLICY = "screening_no_post"

BALANCED_MODE_KEY = "balanced_screening"
HIGH_SENSITIVITY_MODE_KEY = "high_sensitivity_hint"
BALANCED_THRESHOLD = 0.775
HIGH_SENSITIVITY_THRESHOLD = 0.558

MODE_METADATA = {
    BALANCED_MODE_KEY: {
        "key": BALANCED_MODE_KEY,
        "title": "默认平衡筛查模式",
        "role": "primary",
        "modelVersion": MODEL_VERSION,
        "thresholdPolicy": "maximize_balanced_accuracy",
        "threshold": BALANCED_THRESHOLD,
        "sensitivityEstimate": 0.822,
        "specificityEstimate": 0.701,
        "description": "主推结果展示，兼顾少漏筛和少误报。",
    },
    HIGH_SENSITIVITY_MODE_KEY: {
        "key": HIGH_SENSITIVITY_MODE_KEY,
        "title": "高敏感度提示模式",
        "role": "auxiliary",
        "modelVersion": "specificity_rescue_threshold_v1",
        "sourceModelVersion": MODEL_VERSION,
        "thresholdPolicy": "sensitivity_0_90_spec_max",
        "threshold": HIGH_SENSITIVITY_THRESHOLD,
        "sensitivityEstimate": 0.934,
        "specificityEstimate": 0.540,
        "description": "用于尽量减少漏筛，假阳性可能增加，不作为默认结论。",
    },
}

CLASS_LABELS = {0: "筛查阴性", 1: "需进一步评估", 2: "需进一步评估"}

FIELD_KEY_TO_COLUMN = {
    "sex": "性别",
    "age": "年龄",
    "wbc": "白细胞",
    "hemoglobin": "血红蛋白",
    "platelet": "血小板",
    "alt": "谷丙",
    "ast": "谷草",
    "potassium": "钾",
    "sodium": "钠",
    "chloride": "氯",
    "creatinine": "肌酐",
    "rassEquivalentScore": "RASS_等效分数",
    "diureticEquivalentScore": "利尿剂_等效分数",
    "dhpCcbEquivalentScore": "二氢吡啶类_等效分数",
    "betaEquivalentScore": "Beta_等效分数",
    "alphaEquivalentScore": "Alpha_等效分数",
    "nonDhpCcbEquivalentScore": "非二氢吡啶类_等效分数",
    "combinedMedicationCount": "联合用药_总数",
    "standingAldosterone": "醛固酮",
    "renin": "肾素",
    "arrRatio": "ARR比值",
    "systolicBp": "收缩压",
    "diastolicBp": "舒展压",
    "adrenalNodule": "是否有肾上腺结节",
    "hyperplasia": "是否有增生",
    "noduleMaxDiameter": "结节最大直径",
    "arrPositive192": "ARR比值>192为阳性，推荐进行确诊试验",
    "confirmatoryTestType": "确诊实验类型",
    "postAldosterone0": "试验前醛固酮",
    "postRenin0": "试验前肾素",
}

XGB_BIN_D3_L20_BAL_PARAMS = {
    "n_estimators": 220,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 0.5,
    "reg_lambda": 20.0,
}

MODEL_BUNDLE: dict[str, Any] | None = None


def is_filled(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def to_number(value: Any) -> float | None:
    if not is_filled(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def make_xgb_classifier(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **XGB_BIN_D3_L20_BAL_PARAMS,
    )


def transform_with_fitted(preprocessor: Any, X_raw: pd.DataFrame) -> pd.DataFrame:
    transformed = preprocessor.transform(X_raw.reset_index(drop=True))
    return pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out(), index=X_raw.index)


def fit_preprocessor_train_only(X_train_raw: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    from screening_0428_experiment import fit_preprocessor_train_only as fit_preprocessor

    return fit_preprocessor(X_train_raw)


def train_model_bundle(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    prepared = prepare_data(DATA_PATH)
    X_raw = prepared.X.reset_index(drop=True)
    y_three = prepared.y.reset_index(drop=True)
    y_binary = task_target(y_three, "binary").reset_index(drop=True)

    fit_idx, cal_idx = train_test_split(
        np.arange(len(y_binary)),
        test_size=CALIBRATION_SIZE,
        random_state=seed + 2028,
        stratify=y_binary,
    )
    X_fit_raw = X_raw.iloc[fit_idx].reset_index(drop=True)
    X_cal_raw = X_raw.iloc[cal_idx].reset_index(drop=True)
    y_fit = y_binary.iloc[fit_idx].reset_index(drop=True)
    y_cal = y_binary.iloc[cal_idx].reset_index(drop=True)

    view = apply_feature_policy(X_fit_raw, X_cal_raw, SCREENING_POLICY)
    preprocessor, X_fit_df = fit_preprocessor_train_only(view.X_train)
    model = make_xgb_classifier(seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    model.fit(X_fit_df, y_fit, sample_weight=sample_weight, verbose=False)

    X_cal_df = transform_with_fitted(preprocessor, view.X_eval)
    raw_cal_prob = np.asarray(model.predict_proba(X_cal_df))[:, 1]
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(np.clip(raw_cal_prob, 1e-6, 1.0 - 1e-6), y_cal)

    bundle = {
        "model": model,
        "calibrator": isotonic,
        "preprocessor": preprocessor,
        "feature_names": list(X_fit_df.columns),
        "raw_columns": list(X_raw.columns),
        "included_columns": list(view.included_columns),
        "dropped_columns": list(view.dropped_columns),
        "zero_fill_columns": list(ZERO_FILL_COLUMNS),
        "model_version": MODEL_VERSION,
        "model_kind": MODEL_KIND,
        "seed": seed,
        "training_rows": int(len(X_raw)),
        "fit_rows": int(len(X_fit_raw)),
        "calibration_rows": int(len(X_cal_raw)),
        "target_distribution_three": {str(k): int(v) for k, v in y_three.value_counts().sort_index().to_dict().items()},
        "target_distribution_binary": {str(k): int(v) for k, v in y_binary.value_counts().sort_index().to_dict().items()},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "screening_policy": SCREENING_POLICY,
        "screening_modes": MODE_METADATA,
    }
    return bundle


def load_or_train_model_bundle(force_retrain: bool = False) -> dict[str, Any]:
    global MODEL_BUNDLE
    if MODEL_BUNDLE is not None and not force_retrain:
        return MODEL_BUNDLE

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force_retrain:
        MODEL_BUNDLE = joblib.load(MODEL_PATH)
        return MODEL_BUNDLE

    print("Training xgb_bin_d3_l20_bal_isotonic runtime screening model...", flush=True)
    MODEL_BUNDLE = train_model_bundle()
    joblib.dump(MODEL_BUNDLE, MODEL_PATH)
    print(f"Saved model artifact: {MODEL_PATH}", flush=True)
    return MODEL_BUNDLE


def make_single_feature_frame(features: dict[str, Any], bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    row = {column: np.nan for column in bundle["raw_columns"]}
    missing_fields: list[str] = []
    included_columns = set(bundle["included_columns"])

    for column in bundle["zero_fill_columns"]:
        if column in row:
            row[column] = 0

    for field_key, column in FIELD_KEY_TO_COLUMN.items():
        value = features.get(field_key)
        if not is_filled(value):
            if column in included_columns:
                missing_fields.append(field_key)
            continue
        if column == "确诊实验类型":
            row[column] = normalize_experiment_type(value)
        else:
            numeric_value = to_number(value)
            row[column] = numeric_value if numeric_value is not None else np.nan

    arr_col = FIELD_KEY_TO_COLUMN["arrRatio"]
    ald_col = FIELD_KEY_TO_COLUMN["standingAldosterone"]
    renin_col = FIELD_KEY_TO_COLUMN["renin"]
    arr_positive_col = FIELD_KEY_TO_COLUMN["arrPositive192"]
    if pd.isna(row[arr_col]) and not pd.isna(row[ald_col]) and not pd.isna(row[renin_col]) and float(row[renin_col]) > 0:
        row[arr_col] = float(row[ald_col]) / float(row[renin_col])
        if "arrRatio" in missing_fields:
            missing_fields.remove("arrRatio")
    if pd.isna(row[arr_positive_col]) and not pd.isna(row[arr_col]):
        row[arr_positive_col] = 1 if float(row[arr_col]) > 192 else 0
        if "arrPositive192" in missing_fields:
            missing_fields.remove("arrPositive192")

    return pd.DataFrame([row], columns=bundle["raw_columns"]), missing_fields


def predict_probability(payload: dict[str, Any], bundle: dict[str, Any]) -> tuple[float, list[str]]:
    features = payload.get("features") or {}
    if not isinstance(features, dict):
        raise ValueError("features must be an object")

    X_raw, missing_fields = make_single_feature_frame(features, bundle)
    X_screening = X_raw[bundle["included_columns"]].copy()
    X_df = transform_with_fitted(bundle["preprocessor"], X_screening)
    raw_prob = np.asarray(bundle["model"].predict_proba(X_df))[:, 1]
    calibrated = bundle["calibrator"].transform(np.clip(raw_prob, 1e-6, 1.0 - 1e-6))
    probability = float(np.clip(calibrated[0], 1e-6, 1.0 - 1e-6))
    return probability, missing_fields


def build_mode_result(mode_key: str, probability: float) -> dict[str, Any]:
    metadata = dict(MODE_METADATA[mode_key])
    predicted_positive = probability >= float(metadata["threshold"])
    return {
        **metadata,
        "probability": round_float(probability),
        "negativeProbability": round_float(1.0 - probability),
        "predictedPositive": bool(predicted_positive),
        "label": "需进一步评估" if predicted_positive else "筛查阴性",
    }


def build_advice(
    balanced: dict[str, Any],
    high_sensitivity: dict[str, Any],
    completeness_score: float,
    bundle: dict[str, Any],
) -> list[str]:
    advice: list[str] = []
    if completeness_score < 0.35:
        advice.append("当前录入字段较少，模型已按训练管线进行缺失值处理；建议补充初筛、用药、影像和试验前信息后再参考。")
    if balanced["predictedPositive"]:
        advice.append("默认平衡筛查模式提示“需进一步评估”，建议结合规范采血条件、ARR、用药状态和专科流程判断是否进入确诊评估。")
    elif high_sensitivity["predictedPositive"]:
        advice.append("默认平衡筛查模式未达阳性阈值，但高敏感度提示模式已触发；如临床怀疑较强，可考虑复查 ARR 或补充关键指标以减少漏筛。")
    else:
        advice.append("两个固定筛查模式均未触发阳性提示；若临床表现或采血条件仍可疑，仍建议复核关键指标。")
    advice.append("高敏感度提示模式仅用于减少漏筛，可能增加假阳性，不作为默认结论。")
    advice.append(f"当前后端主模型为 {bundle['model_version']}，正式筛查特征策略为 {bundle['screening_policy']}。")
    return advice


def predict_with_model(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_or_train_model_bundle()
    positive_probability, missing_fields = predict_probability(payload, bundle)
    balanced = build_mode_result(BALANCED_MODE_KEY, positive_probability)
    high_sensitivity = build_mode_result(HIGH_SENSITIVITY_MODE_KEY, positive_probability)
    model_field_count = sum(1 for column in FIELD_KEY_TO_COLUMN.values() if column in set(bundle["included_columns"]))

    predicted_positive = bool(balanced["predictedPositive"])
    predicted_class = 2 if predicted_positive else 0
    probabilities = {
        "class0": round_float(1.0 - positive_probability),
        "class1": 0.0,
        "class2": round_float(positive_probability),
    }
    confidence = positive_probability if predicted_positive else 1.0 - positive_probability
    completeness_score = round_float((model_field_count - len(missing_fields)) / max(model_field_count, 1))

    return {
        "mode": "api",
        "task": "binary_screening",
        "primaryModeKey": BALANCED_MODE_KEY,
        "predictedClass": predicted_class,
        "predictedLabel": balanced["label"],
        "probabilities": probabilities,
        "confidence": round_float(confidence),
        "completenessScore": completeness_score,
        "missingFields": missing_fields,
        "advice": build_advice(balanced, high_sensitivity, completeness_score, bundle),
        "modelVersion": bundle["model_version"],
        "modelKind": bundle["model_kind"],
        "screeningModes": {
            BALANCED_MODE_KEY: balanced,
            HIGH_SENSITIVITY_MODE_KEY: high_sensitivity,
        },
    }


class PARequestHandler(BaseHTTPRequestHandler):
    server_version = "PAApiBackend/3.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            bundle = load_or_train_model_bundle()
            self._send_json(
                200,
                {
                    "ok": True,
                    "mode": "api",
                    "modelVersion": bundle["model_version"],
                    "modelKind": bundle["model_kind"],
                    "trainedAt": bundle["trained_at"],
                    "artifactPath": str(MODEL_PATH),
                    "trainingRows": bundle["training_rows"],
                    "fitRows": bundle["fit_rows"],
                    "calibrationRows": bundle["calibration_rows"],
                    "targetDistribution": bundle["target_distribution_binary"],
                    "screeningPolicy": bundle["screening_policy"],
                    "screeningModes": bundle["screening_modes"],
                },
            )
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/pa-diagnosis/predict":
            self._send_json(404, {"error": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            result = predict_with_model(payload)
        except ValueError as exc:
            self._send_json(400, {"error": "bad_request", "message": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": "server_error", "message": str(exc)})
            return

        self._send_json(200, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PA screening API backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--retrain", action="store_true", help="Retrain and overwrite the cached model artifact.")
    args = parser.parse_args()
    load_or_train_model_bundle(force_retrain=args.retrain)
    server = ThreadingHTTPServer((args.host, args.port), PARequestHandler)
    print(f"PA API backend listening on http://{args.host}:{args.port}", flush=True)
    print(f"Primary model: {MODEL_VERSION}", flush=True)
    print("Modes: balanced_screening, high_sensitivity_hint", flush=True)
    print("POST /pa-diagnosis/predict", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping PA API backend.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
