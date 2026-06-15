from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

from multiclass_ensemble_experiment import (
    TARGET_LABELS,
    build_preprocessor,
    evaluate_predictions,
    plot_confusion_heatmap,
    plot_multiclass_roc,
    prepare_data,
    transform_with_preprocessor,
)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42
TREATMENT_COLUMNS = [
    "RASS_等效分数",
    "利尿剂_等效分数",
    "二氢吡啶类_等效分数",
    "Beta_等效分数",
    "Alpha_等效分数",
    "非二氢吡啶类_等效分数",
    "联合用药_总数",
]
BASE_COVARIATE_COLUMNS = [
    "性别",
    "年龄",
    "白细胞",
    "血红蛋白",
    "血小板",
    "谷丙",
    "谷草",
    "钾",
    "钠",
    "氯",
    "肌酐",
    "收缩压",
    "舒展压",
    "是否有肾上腺结节",
    "是否有增生",
    "结节最大直径",
    "立位醛固酮",
    "肾素",
    "确诊实验类型",
]
POST_TREATMENT_COLUMNS = [
    "确诊后_卧位醛固酮醛固酮",
    "肾素.1",
    "确诊后_卧位醛固酮醛固酮.1",
    "肾素.2",
    "确诊后_卧位醛固酮醛固酮.2",
    "肾素.3",
]
ARR_COLUMN = "ARR比值"
DX_COLUMN = "确诊（0为排除；1为确诊；2为灰色区域）"


@dataclass
class DeconfounderData:
    full_df: pd.DataFrame
    X_base: pd.DataFrame
    A: pd.DataFrame
    y_arr: pd.Series
    y_dx: pd.Series


@dataclass
class RouteArtifacts:
    route_name: str
    latent_train: np.ndarray | None
    latent_test: np.ndarray | None
    factual_train: np.ndarray
    factual_test: np.ndarray
    counterfactual_train: np.ndarray | None
    counterfactual_test: np.ndarray | None
    arr_metrics: dict[str, float]
    extra: dict[str, Any]


class TreatmentVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 3, hidden_dim: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> tuple[Path, Path]:
    project_dir = Path(__file__).resolve().parent
    input_path = project_dir / "数据表格测试.xlsx"
    output_dir = project_dir / "medical_deconfounder_outputs"
    return input_path, output_dir


def prepare_data_for_deconfounder(file_path: Path) -> DeconfounderData:
    prepared = prepare_data(file_path)
    df = prepared.cleaned_df.copy()
    df[ARR_COLUMN] = pd.to_numeric(df[ARR_COLUMN], errors="coerce")
    df = df[df[ARR_COLUMN].notna()].copy()
    df[ARR_COLUMN] = df[ARR_COLUMN].clip(lower=0)

    available_x = [col for col in BASE_COVARIATE_COLUMNS if col in df.columns]
    missing_x = [col for col in BASE_COVARIATE_COLUMNS if col not in df.columns]
    available_treatments = [col for col in TREATMENT_COLUMNS if col in df.columns]
    if len(available_treatments) != len(TREATMENT_COLUMNS):
        missing = [col for col in TREATMENT_COLUMNS if col not in df.columns]
        raise KeyError(f"以下治疗变量缺失，无法实施去交杂实验: {missing}")

    X_base = df[available_x].copy()
    A = df[available_treatments].astype(float).copy()
    y_arr = df[ARR_COLUMN].astype(float).copy()
    y_dx = df["_target_clean"].astype(int).copy()

    if missing_x:
        print(f"警告：以下基础协变量在当前表中不存在，已自动跳过: {missing_x}")

    return DeconfounderData(
        full_df=df,
        X_base=X_base,
        A=A,
        y_arr=y_arr,
        y_dx=y_dx,
    )


def make_train_test_split(data: DeconfounderData) -> dict[str, Any]:
    indices = np.arange(len(data.full_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data.y_dx,
    )
    train_idx = pd.Index(data.full_df.index[train_idx])
    test_idx = pd.Index(data.full_df.index[test_idx])
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def build_x_blocks(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X_train.columns if col not in numeric_features]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_df, X_test_df = transform_with_preprocessor(preprocessor, X_train, X_test)
    return X_train_df, X_test_df


def concat_feature_blocks(
    X_block: pd.DataFrame,
    A_block: pd.DataFrame | None = None,
    latent_block: np.ndarray | None = None,
    arr_feature: np.ndarray | None = None,
    latent_prefix: str = "z",
) -> pd.DataFrame:
    frames = [X_block.copy()]
    if A_block is not None:
        A_df = A_block.copy()
        A_df.columns = [f"treat__{col}" for col in A_df.columns]
        frames.append(A_df)
    if latent_block is not None:
        latent_df = pd.DataFrame(
            latent_block,
            index=X_block.index,
            columns=[f"{latent_prefix}_{i}" for i in range(latent_block.shape[1])],
        )
        frames.append(latent_df)
    if arr_feature is not None:
        arr_df = pd.DataFrame(
            {"arr_feature": np.asarray(arr_feature).reshape(-1)},
            index=X_block.index,
        )
        frames.append(arr_df)
    return pd.concat(frames, axis=1)


def make_arr_regressor(seed: int = RANDOM_STATE) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=320,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.05,
        random_state=seed,
        n_jobs=1,
    )


def make_dx_classifier(seed: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=320,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.05,
        random_state=seed,
        eval_metric="mlogloss",
        n_jobs=1,
    )


def fit_simple_deconfounder(
    A_train: pd.DataFrame,
    A_test: pd.DataFrame,
    latent_dim: int = 3,
) -> dict[str, Any]:
    scaler = StandardScaler()
    A_train_scaled = scaler.fit_transform(A_train)
    A_test_scaled = scaler.transform(A_test)
    n_components = min(latent_dim, A_train.shape[1] - 1)
    model = FactorAnalysis(n_components=n_components, random_state=RANDOM_STATE)
    z_train = model.fit_transform(A_train_scaled)
    z_test = model.transform(A_test_scaled)
    recon_train_scaled = z_train @ model.components_ + model.mean_
    recon_test_scaled = z_test @ model.components_ + model.mean_
    recon_train = scaler.inverse_transform(recon_train_scaled)
    recon_test = scaler.inverse_transform(recon_test_scaled)
    reconstruction_rmse = float(
        math.sqrt(mean_squared_error(A_test.to_numpy().reshape(-1), recon_test.reshape(-1)))
    )
    return {
        "model": model,
        "scaler": scaler,
        "z_train": z_train,
        "z_test": z_test,
        "reconstruction_rmse": reconstruction_rmse,
        "recon_train": recon_train,
        "recon_test": recon_test,
    }


def fit_deep_deconfounder(
    A_train: pd.DataFrame,
    A_test: pd.DataFrame,
    output_dir: Path,
    latent_dim: int = 3,
) -> dict[str, Any]:
    set_random_seed(RANDOM_STATE)
    scaler = StandardScaler()
    A_train_scaled = scaler.fit_transform(A_train)
    A_test_scaled = scaler.transform(A_test)
    A_train_tensor = torch.tensor(A_train_scaled, dtype=torch.float32)
    val_size = max(1, int(0.2 * len(A_train_tensor)))
    perm = torch.randperm(len(A_train_tensor))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    train_tensor = A_train_tensor[train_idx]
    val_tensor = A_train_tensor[val_idx]

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=64, shuffle=False)

    model = TreatmentVAE(input_dim=A_train.shape[1], latent_dim=min(latent_dim, A_train.shape[1] - 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience = 35
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, 241):
        model.train()
        batch_losses: list[float] = []
        for (batch_x,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch_x)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.08 * kl_loss
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (batch_x,) in val_loader:
                recon, mu, logvar = model(batch_x)
                recon_loss = F.mse_loss(recon, batch_x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                val_losses.append(float((recon_loss + 0.08 * kl_loss).detach().cpu()))

        train_loss = float(np.mean(batch_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("深度去交杂器训练失败，未保存到有效模型。")

    model.load_state_dict(best_state)
    model.eval()

    train_full_tensor = torch.tensor(A_train_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(A_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        mu_train, _ = model.encode(train_full_tensor)
        mu_test, _ = model.encode(test_tensor)
        recon_test = model.decode(mu_test).cpu().numpy()

    z_train = mu_train.cpu().numpy()
    z_test = mu_test.cpu().numpy()
    recon_test_raw = scaler.inverse_transform(recon_test)
    reconstruction_rmse = float(
        math.sqrt(mean_squared_error(A_test.to_numpy().reshape(-1), recon_test_raw.reshape(-1)))
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "deep_deconfounder_training_history.csv", index=False, encoding="utf-8-sig")
    plot_training_history(history_df, output_dir / "figures" / "deep_deconfounder_training_history.png")

    return {
        "model": model,
        "scaler": scaler,
        "z_train": z_train,
        "z_test": z_test,
        "reconstruction_rmse": reconstruction_rmse,
        "history_df": history_df,
    }


def plot_training_history(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.title("深度去交杂器训练过程")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def fit_arr_model_and_predict(
    route_name: str,
    X_train_block: pd.DataFrame,
    X_test_block: pd.DataFrame,
    y_arr_train: pd.Series,
    y_arr_test: pd.Series,
    A_train: pd.DataFrame | None = None,
    A_test: pd.DataFrame | None = None,
    latent_train: np.ndarray | None = None,
    latent_test: np.ndarray | None = None,
) -> RouteArtifacts:
    train_features = concat_feature_blocks(X_train_block, A_train, latent_train, latent_prefix=route_name)
    test_features = concat_feature_blocks(X_test_block, A_test, latent_test, latent_prefix=route_name)

    y_train_log = np.log1p(np.clip(y_arr_train.to_numpy(), a_min=0, a_max=None))
    reg = make_arr_regressor(RANDOM_STATE)
    reg.fit(train_features, y_train_log)
    factual_train = np.expm1(reg.predict(train_features)).clip(min=0)
    factual_test = np.expm1(reg.predict(test_features)).clip(min=0)

    arr_metrics = evaluate_arr_regression(y_arr_test.to_numpy(), factual_test)

    counterfactual_train: np.ndarray | None = None
    counterfactual_test: np.ndarray | None = None
    if A_train is not None and A_test is not None:
        A_train_zero = pd.DataFrame(0.0, index=A_train.index, columns=A_train.columns)
        A_test_zero = pd.DataFrame(0.0, index=A_test.index, columns=A_test.columns)
        cf_train_features = concat_feature_blocks(
            X_train_block, A_train_zero, latent_train, latent_prefix=route_name
        )
        cf_test_features = concat_feature_blocks(
            X_test_block, A_test_zero, latent_test, latent_prefix=route_name
        )
        counterfactual_train = np.expm1(reg.predict(cf_train_features)).clip(min=0)
        counterfactual_test = np.expm1(reg.predict(cf_test_features)).clip(min=0)

    return RouteArtifacts(
        route_name=route_name,
        latent_train=latent_train,
        latent_test=latent_test,
        factual_train=factual_train,
        factual_test=factual_test,
        counterfactual_train=counterfactual_train,
        counterfactual_test=counterfactual_test,
        arr_metrics=arr_metrics,
        extra={"model": reg},
    )


def evaluate_arr_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_diagnosis_model(
    model_name: str,
    X_train_block: pd.DataFrame,
    X_test_block: pd.DataFrame,
    arr_train_feature: np.ndarray,
    arr_test_feature: np.ndarray,
    y_dx_train: pd.Series,
    y_dx_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray]:
    train_features = concat_feature_blocks(
        X_train_block,
        A_block=None,
        latent_block=None,
        arr_feature=arr_train_feature,
        latent_prefix=model_name,
    )
    test_features = concat_feature_blocks(
        X_test_block,
        A_block=None,
        latent_block=None,
        arr_feature=arr_test_feature,
        latent_prefix=model_name,
    )
    clf = make_dx_classifier(RANDOM_STATE)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_dx_train)
    clf.fit(train_features, y_dx_train, sample_weight=sample_weight)
    proba = clf.predict_proba(test_features)
    metrics = evaluate_predictions(y_dx_test, proba, TARGET_LABELS, [0, 1, 2])
    metrics["route_name"] = model_name
    return metrics, proba


def plot_treatment_correlation(A_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(A_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("治疗矩阵相关性热力图")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_treatment_distribution(A_df: pd.DataFrame, output_path: Path) -> None:
    long_df = A_df.melt(var_name="drug", value_name="score")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=long_df, x="drug", y="score")
    plt.xticks(rotation=35, ha="right")
    plt.title("各药物等效分数分布")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_latent_projection(
    latent: np.ndarray,
    labels: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    if latent.shape[1] >= 2:
        coords = latent[:, :2]
    else:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(latent)
    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": labels.map(TARGET_LABELS),
        },
        index=labels.index,
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="label", s=60)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_counterfactual_scatter(
    obs_arr: np.ndarray,
    cf_simple: np.ndarray,
    cf_deep: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(obs_arr, cf_simple, alpha=0.7, label="Simple")
    plt.scatter(obs_arr, cf_deep, alpha=0.7, label="Deep")
    max_val = max(float(np.max(obs_arr)), float(np.max(cf_simple)), float(np.max(cf_deep)))
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="gray")
    plt.xlabel("观测 ARR")
    plt.ylabel("反事实停药 ARR")
    plt.title("观测 ARR 与反事实停药 ARR 散点图")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_delta_boxplot(
    patient_df: pd.DataFrame,
    output_path: Path,
) -> None:
    long_df = patient_df[
        ["true_dx_label", "delta_arr_simple", "delta_arr_deep"]
    ].melt(id_vars="true_dx_label", var_name="route", value_name="delta_arr")
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=long_df, x="true_dx_label", y="delta_arr", hue="route")
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("真实诊断类别")
    plt.ylabel("反事实 ARR - 观测 ARR")
    plt.title("不同诊断类别下的反事实 ARR 变化")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_diagnosis_roc_comparison(
    y_true: pd.Series,
    probas: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for model_name, proba in probas.items():
        auc_score = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin[:, 1], proba[:, 1])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("基于观测/反事实 ARR 的确诊 ROC 对比")
    plt.xlabel("假阳性率")
    plt.ylabel("真正率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_arr_metrics(arr_metrics_df: pd.DataFrame, output_path: Path) -> None:
    melted = arr_metrics_df.melt(id_vars="route_name", var_name="metric", value_name="score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="route_name")
    plt.title("ARR 回归模型指标对比")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_dx_metrics(dx_metrics_df: pd.DataFrame, output_path: Path) -> None:
    melted = dx_metrics_df[
        ["route_name", "accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"]
    ].melt(id_vars="route_name", var_name="metric", value_name="score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="route_name")
    plt.ylim(0, 1.05)
    plt.title("基于观测/反事实 ARR 的诊断指标对比")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    set_random_seed(RANDOM_STATE)
    input_path, output_dir = parse_args()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    artifacts_dir = output_dir / "artifacts"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data_for_deconfounder(input_path)
    split = make_train_test_split(data)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    X_train_raw = data.X_base.loc[train_idx]
    X_test_raw = data.X_base.loc[test_idx]
    A_train = data.A.loc[train_idx]
    A_test = data.A.loc[test_idx]
    y_arr_train = data.y_arr.loc[train_idx]
    y_arr_test = data.y_arr.loc[test_idx]
    y_dx_train = data.y_dx.loc[train_idx]
    y_dx_test = data.y_dx.loc[test_idx]

    X_train_block, X_test_block = build_x_blocks(X_train_raw, X_test_raw)

    variable_partition = {
        "treatment_columns": TREATMENT_COLUMNS,
        "base_covariate_columns": [col for col in BASE_COVARIATE_COLUMNS if col in data.full_df.columns],
        "post_treatment_columns": [col for col in POST_TREATMENT_COLUMNS if col in data.full_df.columns],
        "arr_outcome_column": ARR_COLUMN,
        "diagnosis_outcome_column": DX_COLUMN,
        "n_samples_with_arr": int(len(data.full_df)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
    }
    (artifacts_dir / "variable_partition.json").write_text(
        json.dumps(variable_partition, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_treatment_correlation(data.A, figures_dir / "treatment_correlation_heatmap.png")
    plot_treatment_distribution(data.A, figures_dir / "treatment_distribution_boxplot.png")

    # Model A: X -> ARR_obs
    route_A = fit_arr_model_and_predict(
        "model_a_x_only",
        X_train_block,
        X_test_block,
        y_arr_train,
        y_arr_test,
    )

    # Model B: X + A -> ARR_obs
    route_B = fit_arr_model_and_predict(
        "model_b_x_a",
        X_train_block,
        X_test_block,
        y_arr_train,
        y_arr_test,
        A_train=A_train,
        A_test=A_test,
    )

    # Simple deconfounder
    simple_dc = fit_simple_deconfounder(A_train, A_test, latent_dim=3)
    route_simple = fit_arr_model_and_predict(
        "model_c_simple",
        X_train_block,
        X_test_block,
        y_arr_train,
        y_arr_test,
        A_train=A_train,
        A_test=A_test,
        latent_train=simple_dc["z_train"],
        latent_test=simple_dc["z_test"],
    )

    # Deep deconfounder
    deep_dc = fit_deep_deconfounder(A_train, A_test, output_dir, latent_dim=3)
    route_deep = fit_arr_model_and_predict(
        "model_c_deep",
        X_train_block,
        X_test_block,
        y_arr_train,
        y_arr_test,
        A_train=A_train,
        A_test=A_test,
        latent_train=deep_dc["z_train"],
        latent_test=deep_dc["z_test"],
    )

    latent_full_simple = np.vstack([simple_dc["z_train"], simple_dc["z_test"]])
    latent_full_deep = np.vstack([deep_dc["z_train"], deep_dc["z_test"]])
    full_dx = pd.concat([y_dx_train, y_dx_test], axis=0)
    plot_latent_projection(
        latent_full_simple,
        full_dx,
        figures_dir / "simple_latent_projection.png",
        "简单版去交杂器潜变量投影",
    )
    plot_latent_projection(
        latent_full_deep,
        full_dx,
        figures_dir / "deep_latent_projection.png",
        "增强版去交杂器潜变量投影",
    )

    arr_metrics_rows = [
        {"route_name": route_A.route_name, **route_A.arr_metrics},
        {"route_name": route_B.route_name, **route_B.arr_metrics},
        {"route_name": route_simple.route_name, **route_simple.arr_metrics},
        {"route_name": route_deep.route_name, **route_deep.arr_metrics},
    ]
    arr_metrics_df = pd.DataFrame(arr_metrics_rows).sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
    arr_metrics_df.to_csv(tables_dir / "arr_regression_metrics.csv", index=False, encoding="utf-8-sig")
    plot_arr_metrics(arr_metrics_df, figures_dir / "arr_regression_metrics.png")

    reconstruction_df = pd.DataFrame(
        [
            {"route_name": "simple_deconfounder", "reconstruction_rmse": simple_dc["reconstruction_rmse"]},
            {"route_name": "deep_deconfounder", "reconstruction_rmse": deep_dc["reconstruction_rmse"]},
        ]
    )
    reconstruction_df.to_csv(tables_dir / "deconfounder_reconstruction_metrics.csv", index=False, encoding="utf-8-sig")

    # Diagnosis models: observed ARR vs counterfactual ARR
    dx_metrics_obs, proba_obs = fit_diagnosis_model(
        "diagnosis_obs_arr",
        X_train_block,
        X_test_block,
        y_arr_train.to_numpy(),
        y_arr_test.to_numpy(),
        y_dx_train,
        y_dx_test,
    )
    dx_metrics_simple, proba_simple = fit_diagnosis_model(
        "diagnosis_cf_simple",
        X_train_block,
        X_test_block,
        route_simple.counterfactual_train,
        route_simple.counterfactual_test,
        y_dx_train,
        y_dx_test,
    )
    dx_metrics_deep, proba_deep = fit_diagnosis_model(
        "diagnosis_cf_deep",
        X_train_block,
        X_test_block,
        route_deep.counterfactual_train,
        route_deep.counterfactual_test,
        y_dx_train,
        y_dx_test,
    )

    diagnosis_metrics_df = pd.DataFrame([dx_metrics_obs, dx_metrics_simple, dx_metrics_deep]).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False
    )
    diagnosis_metrics_df.to_csv(tables_dir / "diagnosis_metrics.csv", index=False, encoding="utf-8-sig")
    plot_dx_metrics(diagnosis_metrics_df, figures_dir / "diagnosis_metrics.png")
    plot_diagnosis_roc_comparison(
        y_dx_test,
        {
            "Observed ARR": proba_obs,
            "CF Simple": proba_simple,
            "CF Deep": proba_deep,
        },
        figures_dir / "diagnosis_roc_comparison.png",
    )

    best_dx_row = diagnosis_metrics_df.iloc[0]
    best_name = str(best_dx_row["route_name"])
    best_proba_map = {
        "diagnosis_obs_arr": proba_obs,
        "diagnosis_cf_simple": proba_simple,
        "diagnosis_cf_deep": proba_deep,
    }
    best_pred = np.argmax(best_proba_map[best_name], axis=1)
    plot_confusion_heatmap(y_dx_test, best_pred, figures_dir / "best_diagnosis_confusion_heatmap.png")
    plot_multiclass_roc(y_dx_test, best_proba_map[best_name], figures_dir / "best_diagnosis_multiclass_roc.png")

    patient_df = data.full_df.loc[test_idx].copy()
    patient_df["true_dx"] = y_dx_test
    patient_df["true_dx_label"] = y_dx_test.map(TARGET_LABELS)
    patient_df["arr_observed"] = y_arr_test
    patient_df["arr_pred_model_a"] = route_A.factual_test
    patient_df["arr_pred_model_b"] = route_B.factual_test
    patient_df["arr_pred_simple_factual"] = route_simple.factual_test
    patient_df["arr_pred_deep_factual"] = route_deep.factual_test
    patient_df["arr_cf_stop_simple"] = route_simple.counterfactual_test
    patient_df["arr_cf_stop_deep"] = route_deep.counterfactual_test
    patient_df["delta_arr_simple"] = patient_df["arr_cf_stop_simple"] - patient_df["arr_observed"]
    patient_df["delta_arr_deep"] = patient_df["arr_cf_stop_deep"] - patient_df["arr_observed"]
    patient_df["dx_pred_obs"] = np.argmax(proba_obs, axis=1)
    patient_df["dx_pred_simple"] = np.argmax(proba_simple, axis=1)
    patient_df["dx_pred_deep"] = np.argmax(proba_deep, axis=1)
    patient_df.to_excel(output_dir / "patient_counterfactual_arr.xlsx", index=True)

    plot_counterfactual_scatter(
        patient_df["arr_observed"].to_numpy(),
        patient_df["arr_cf_stop_simple"].to_numpy(),
        patient_df["arr_cf_stop_deep"].to_numpy(),
        figures_dir / "counterfactual_arr_scatter.png",
    )
    plot_delta_boxplot(patient_df, figures_dir / "counterfactual_delta_boxplot.png")

    latent_summary = {
        "simple_deconfounder": {
            "latent_dim": int(simple_dc["z_train"].shape[1]),
            "reconstruction_rmse": float(simple_dc["reconstruction_rmse"]),
        },
        "deep_deconfounder": {
            "latent_dim": int(deep_dc["z_train"].shape[1]),
            "reconstruction_rmse": float(deep_dc["reconstruction_rmse"]),
            "best_val_loss": float(deep_dc["history_df"]["val_loss"].min()),
        },
    }
    (artifacts_dir / "deconfounder_latent_summary.json").write_text(
        json.dumps(latent_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "n_samples": int(len(data.full_df)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "arr_outcome_summary": {
            "best_route": arr_metrics_df.iloc[0]["route_name"],
            "metrics": arr_metrics_df.iloc[0].to_dict(),
        },
        "diagnosis_summary": {
            "best_route": best_name,
            "metrics": diagnosis_metrics_df.iloc[0].to_dict(),
        },
        "reconstruction_summary": latent_summary,
        "counterfactual_delta_summary": {
            "simple_mean_delta": float(patient_df["delta_arr_simple"].mean()),
            "deep_mean_delta": float(patient_df["delta_arr_deep"].mean()),
            "simple_median_delta": float(patient_df["delta_arr_simple"].median()),
            "deep_median_delta": float(patient_df["delta_arr_deep"].median()),
        },
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("医学去交杂器实验完成")
    print(arr_metrics_df.to_string(index=False))
    print(diagnosis_metrics_df[["route_name", "accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"]].to_string(index=False))
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
