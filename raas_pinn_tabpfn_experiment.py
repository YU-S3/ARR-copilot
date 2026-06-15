from __future__ import annotations

import json
import math
import os
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
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier
from tabpfn_client import set_access_token
from torch.utils.data import DataLoader, Dataset

from env_utils import get_tabpfn_token
from multiclass_ensemble_experiment import (
    TARGET_LABELS,
    ZERO_FILL_COLUMNS,
    build_preprocessor,
    evaluate_predictions,
    plot_confusion_heatmap,
    plot_multiclass_roc,
    prepare_data,
)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font="Microsoft YaHei")
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42
RENIN_COLUMNS = ["肾素", "肾素.1", "肾素.2", "肾素.3"]
ALDO_COLUMNS = [
    "立位醛固酮",
    "确诊后_卧位醛固酮醛固酮",
    "确诊后_卧位醛固酮醛固酮.1",
    "确诊后_卧位醛固酮醛固酮.2",
]
ARR_COLUMN = "ARR比值"


@dataclass
class SplitData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_dense: np.ndarray
    X_test_dense: np.ndarray
    feature_names: list[str]


class RAASDataset(Dataset):
    def __init__(
        self,
        x_dense: np.ndarray,
        y: np.ndarray,
        drug_total: np.ndarray,
        renin_values: np.ndarray,
        renin_mask: np.ndarray,
        aldo_values: np.ndarray,
        aldo_mask: np.ndarray,
        arr_values: np.ndarray,
        arr_mask: np.ndarray,
    ) -> None:
        self.x_dense = torch.tensor(x_dense, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.drug_total = torch.tensor(drug_total, dtype=torch.float32)
        self.renin_values = torch.tensor(renin_values, dtype=torch.float32)
        self.renin_mask = torch.tensor(renin_mask, dtype=torch.float32)
        self.aldo_values = torch.tensor(aldo_values, dtype=torch.float32)
        self.aldo_mask = torch.tensor(aldo_mask, dtype=torch.float32)
        self.arr_values = torch.tensor(arr_values, dtype=torch.float32)
        self.arr_mask = torch.tensor(arr_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x_dense[index],
            "y": self.y[index],
            "drug_total": self.drug_total[index],
            "renin_values": self.renin_values[index],
            "renin_mask": self.renin_mask[index],
            "aldo_values": self.aldo_values[index],
            "aldo_mask": self.aldo_mask[index],
            "arr_values": self.arr_values[index],
            "arr_mask": self.arr_mask[index],
        }


class RAASPINNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 3) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_head = nn.Linear(hidden_dim, 3)
        self.class_head = nn.Linear(hidden_dim + 3, n_classes)

        # 正参数，分别表示半衰期衰减、结合速率、解离速率、肾素/醛固酮动力学参数
        self.raw_lambda = nn.Parameter(torch.tensor(0.20))
        self.raw_kon = nn.Parameter(torch.tensor(0.15))
        self.raw_koff = nn.Parameter(torch.tensor(0.10))
        self.raw_s_r = nn.Parameter(torch.tensor(0.30))
        self.raw_k_r = nn.Parameter(torch.tensor(0.20))
        self.raw_beta_b = nn.Parameter(torch.tensor(0.15))
        self.raw_s_a = nn.Parameter(torch.tensor(0.25))
        self.raw_c_r = nn.Parameter(torch.tensor(0.25))
        self.raw_k_a = nn.Parameter(torch.tensor(0.20))
        self.raw_gamma_b = nn.Parameter(torch.tensor(0.12))

    def positive(self, value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value) + 1e-4

    def forward_flat(
        self, x_flat: torch.Tensor, t_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.feature_net(x_flat)
        t_feat = torch.cat([t_flat, torch.sin(t_flat), torch.cos(t_flat)], dim=1)
        z = self.trunk(torch.cat([h, t_feat], dim=1))
        raw_states = self.state_head(z)
        binding = torch.sigmoid(raw_states[:, :1])
        renin = F.softplus(raw_states[:, 1:2]) + 1e-4
        aldo = F.softplus(raw_states[:, 2:3]) + 1e-4
        logits = self.class_head(torch.cat([z, torch.log1p(renin), torch.log1p(aldo), binding], dim=1))
        return binding, renin, aldo, logits

    def forward_sequence(
        self, x: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        n_times = times.shape[0]
        x_rep = x.unsqueeze(1).repeat(1, n_times, 1).reshape(-1, x.shape[1])
        t_rep = times.view(1, n_times, 1).repeat(batch_size, 1, 1).reshape(-1, 1)
        t_rep = t_rep.clone().detach().requires_grad_(True)
        binding, renin, aldo, logits = self.forward_flat(x_rep, t_rep)

        binding = binding.view(batch_size, n_times)
        renin = renin.view(batch_size, n_times)
        aldo = aldo.view(batch_size, n_times)
        logits = logits.view(batch_size, n_times, -1)
        return binding, renin, aldo, logits, t_rep


def aggregate_patient_logits(logits: torch.Tensor) -> torch.Tensor:
    return 0.7 * logits[:, -1, :] + 0.3 * logits.mean(dim=1)


def build_dense_features(split_x_train: pd.DataFrame, split_x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prepared = prepare_data(Path(__file__).resolve().parent / "数据表格测试.xlsx")
    preprocessor = build_preprocessor(prepared.numeric_features, prepared.categorical_features)
    x_train_df, x_test_df = transform_to_dataframe(preprocessor, split_x_train, split_x_test)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_df)
    x_test_scaled = scaler.transform(x_test_df)
    return x_train_scaled, x_test_scaled, list(x_train_df.columns)


def transform_to_dataframe(
    preprocessor: Any, x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(x_train_t, columns=feature_names, index=x_train.index),
        pd.DataFrame(x_test_t, columns=feature_names, index=x_test.index),
    )


def prepare_split_data(random_state: int = RANDOM_STATE) -> SplitData:
    prepared = prepare_data(Path(__file__).resolve().parent / "数据表格测试.xlsx")
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=random_state,
        stratify=prepared.y,
    )
    x_train_dense, x_test_dense, feature_names = build_dense_features(x_train_raw, x_test_raw)
    return SplitData(
        X_train_raw=x_train_raw,
        X_test_raw=x_test_raw,
        y_train=y_train,
        y_test=y_test,
        X_train_dense=x_train_dense,
        X_test_dense=x_test_dense,
        feature_names=feature_names,
    )


def extract_raas_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    renin = df[RENIN_COLUMNS].astype(float).to_numpy()
    aldo = df[ALDO_COLUMNS].astype(float).to_numpy()
    arr = pd.to_numeric(df[ARR_COLUMN], errors="coerce").to_numpy()
    return {
        "renin_values": np.nan_to_num(np.log1p(np.clip(renin, a_min=0, a_max=None)), nan=0.0),
        "renin_mask": (~np.isnan(renin)).astype(float),
        "aldo_values": np.nan_to_num(np.log1p(np.clip(aldo, a_min=0, a_max=None)), nan=0.0),
        "aldo_mask": (~np.isnan(aldo)).astype(float),
        "arr_values": np.nan_to_num(np.log1p(np.clip(arr, a_min=0, a_max=None)), nan=0.0),
        "arr_mask": (~np.isnan(arr)).astype(float),
        "drug_total": df[ZERO_FILL_COLUMNS].sum(axis=1).astype(float).to_numpy(),
    }


def build_datasets(split_data: SplitData, random_state: int) -> tuple[RAASDataset, RAASDataset, np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(splitter.split(split_data.X_train_dense, split_data.y_train))

    train_targets = extract_raas_targets(split_data.X_train_raw.iloc[train_idx])
    val_targets = extract_raas_targets(split_data.X_train_raw.iloc[val_idx])

    train_ds = RAASDataset(
        x_dense=split_data.X_train_dense[train_idx],
        y=split_data.y_train.iloc[train_idx].to_numpy(),
        **train_targets,
    )
    val_ds = RAASDataset(
        x_dense=split_data.X_train_dense[val_idx],
        y=split_data.y_train.iloc[val_idx].to_numpy(),
        **val_targets,
    )
    return train_ds, val_ds, train_idx, val_idx


def compute_raas_loss(
    model: RAASPINNClassifier,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    class_weights: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    drug_total = batch["drug_total"].to(device).unsqueeze(1)
    renin_values = batch["renin_values"].to(device)
    renin_mask = batch["renin_mask"].to(device)
    aldo_values = batch["aldo_values"].to(device)
    aldo_mask = batch["aldo_mask"].to(device)
    arr_values = batch["arr_values"].to(device)
    arr_mask = batch["arr_mask"].to(device)

    binding, renin, aldo, logits, t_rep = model.forward_sequence(x, times)
    patient_logits = aggregate_patient_logits(logits)
    cls_loss = F.cross_entropy(patient_logits, y, weight=class_weights)

    renin_log = torch.log1p(renin)
    aldo_log = torch.log1p(aldo)
    renin_data_loss = ((renin_log - renin_values) ** 2 * renin_mask).sum() / renin_mask.sum().clamp_min(1.0)
    aldo_data_loss = ((aldo_log - aldo_values) ** 2 * aldo_mask).sum() / aldo_mask.sum().clamp_min(1.0)
    arr_hat = torch.log1p(aldo[:, 0] / (renin[:, 0] + 1e-4))
    arr_loss = (((arr_hat - arr_values) ** 2) * arr_mask).sum() / arr_mask.sum().clamp_min(1.0)

    batch_size, n_times = renin.shape
    binding_flat = binding.reshape(-1, 1)
    renin_flat = renin.reshape(-1, 1)
    aldo_flat = aldo.reshape(-1, 1)
    drug_rep = drug_total.repeat(1, n_times).reshape(-1, 1)

    d_binding_dt = torch.autograd.grad(binding_flat.sum(), t_rep, create_graph=True)[0]
    d_renin_dt = torch.autograd.grad(renin_flat.sum(), t_rep, create_graph=True)[0]
    d_aldo_dt = torch.autograd.grad(aldo_flat.sum(), t_rep, create_graph=True)[0]

    lambda_decay = model.positive(model.raw_lambda)
    kon = model.positive(model.raw_kon)
    koff = model.positive(model.raw_koff)
    s_r = model.positive(model.raw_s_r)
    k_r = model.positive(model.raw_k_r)
    beta_b = model.positive(model.raw_beta_b)
    s_a = model.positive(model.raw_s_a)
    c_r = model.positive(model.raw_c_r)
    k_a = model.positive(model.raw_k_a)
    gamma_b = model.positive(model.raw_gamma_b)

    drug_effect = drug_rep * torch.exp(-lambda_decay * t_rep)
    residual_b = d_binding_dt - (kon * drug_effect * (1.0 - binding_flat) - koff * binding_flat)
    residual_r = d_renin_dt - (s_r - k_r * renin_flat - beta_b * binding_flat)
    residual_a = d_aldo_dt - (s_a + c_r * renin_flat - k_a * aldo_flat - gamma_b * binding_flat)
    physics_loss = (residual_b.pow(2).mean() + residual_r.pow(2).mean() + residual_a.pow(2).mean()) / 3.0

    total_loss = cls_loss + 0.15 * (renin_data_loss + aldo_data_loss) + 0.05 * arr_loss + 0.05 * physics_loss
    diagnostics = {
        "cls_loss": float(cls_loss.detach().cpu()),
        "renin_loss": float(renin_data_loss.detach().cpu()),
        "aldo_loss": float(aldo_data_loss.detach().cpu()),
        "arr_loss": float(arr_loss.detach().cpu()),
        "physics_loss": float(physics_loss.detach().cpu()),
    }
    return total_loss, diagnostics, patient_logits


def train_raas_pinn(
    split_data: SplitData,
    output_dir: Path,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device("cpu")
    train_ds, val_ds, train_idx, val_idx = build_datasets(split_data, random_state)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    torch.manual_seed(random_state)
    model = RAASPINNClassifier(input_dim=split_data.X_train_dense.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device)

    counts = split_data.y_train.iloc[train_idx].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_idx) / (len(counts) * counts.get(i, 1)) for i in range(3)],
        dtype=torch.float32,
        device=device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_score = -np.inf
    patience = 40
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, 241):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics, _ = compute_raas_loss(model, batch, device, class_weights, times)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        scheduler.step()

        model.eval()
        val_logits_list: list[np.ndarray] = []
        val_y_list: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                binding, renin, aldo, logits, _ = model.forward_sequence(x, times)
                patient_logits = aggregate_patient_logits(logits)
                val_logits_list.append(patient_logits.cpu().numpy())
                val_y_list.append(batch["y"].numpy())

        val_logits = np.vstack(val_logits_list)
        val_y = np.concatenate(val_y_list)
        val_pred = val_logits.argmax(axis=1)
        val_macro_f1 = f1_score(val_y, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_macro_f1": float(val_macro_f1)})

        if val_macro_f1 > best_val_score:
            best_val_score = val_macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("RAAS-PINN 训练未得到有效模型。")

    model.load_state_dict(best_state)
    model.eval()

    x_test_tensor = torch.tensor(split_data.X_test_dense, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, _, test_logits, _ = model.forward_sequence(x_test_tensor, times)
        test_logits = aggregate_patient_logits(test_logits)
        test_proba = torch.softmax(test_logits, dim=1).cpu().numpy()

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "raas_pinn_training_history.csv", index=False, encoding="utf-8-sig")
    plot_training_history(history_df, output_dir / "raas_pinn_training_history.png")

    learned_params = {
        "lambda_decay": float(model.positive(model.raw_lambda).detach().cpu()),
        "kon": float(model.positive(model.raw_kon).detach().cpu()),
        "koff": float(model.positive(model.raw_koff).detach().cpu()),
        "s_r": float(model.positive(model.raw_s_r).detach().cpu()),
        "k_r": float(model.positive(model.raw_k_r).detach().cpu()),
        "beta_b": float(model.positive(model.raw_beta_b).detach().cpu()),
        "s_a": float(model.positive(model.raw_s_a).detach().cpu()),
        "c_r": float(model.positive(model.raw_c_r).detach().cpu()),
        "k_a": float(model.positive(model.raw_k_a).detach().cpu()),
        "gamma_b": float(model.positive(model.raw_gamma_b).detach().cpu()),
        "best_val_macro_f1": float(best_val_score),
        "epochs_trained": int(len(history_df)),
    }
    return test_proba, learned_params


def plot_training_history(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], color="#4C72B0", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["val_macro_f1"], color="#C44E52", label="Val Macro F1")
    ax2.set_ylabel("Val Macro F1", color="#C44E52")
    plt.title("RAAS-PINN 训练过程")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_tabpfn(split_data: SplitData) -> np.ndarray:
    token = get_tabpfn_token(Path(__file__).resolve().parent)
    if not token:
        raise RuntimeError("未找到 TabPFN access token。请设置 TABPFN_API_TOKEN 或 TABPFN_TOKEN。")

    set_access_token(token)
    model = ClientTabPFNClassifier(random_state=RANDOM_STATE)
    model.fit(split_data.X_train_dense, split_data.y_train.to_numpy())
    return np.asarray(model.predict_proba(split_data.X_test_dense))


def plot_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    melted = metrics_df.melt(
        id_vars="model_name",
        value_vars=["accuracy", "balanced_accuracy", "macro_f1", "ovr_roc_auc_macro"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model_name")
    plt.ylim(0, 1.05)
    plt.title("XGBoost / TabPFN-3 / RAAS-PINN 指标对比")
    plt.xlabel("指标")
    plt.ylabel("得分")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_roc_overview(y_true: pd.Series, probas: dict[str, np.ndarray], output_path: Path) -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for model_name, proba in probas.items():
        auc_score = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin[:, 1], proba[:, 1])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("模型 ROC 总览（以确诊类 OVR 曲线展示）")
    plt.xlabel("假阳性率")
    plt.ylabel("真正率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "pinn_tabpfn_outputs"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    split_data = prepare_split_data(RANDOM_STATE)
    baseline_summary = json.loads((project_dir / "ensemble_outputs_v2" / "experiment_summary.json").read_text(encoding="utf-8"))
    baseline_predictions = pd.read_excel(project_dir / "ensemble_outputs_v2" / "test_set_predictions.xlsx", index_col=0)
    baseline_metrics = baseline_summary["best_strategy_metrics"]
    baseline_proba = baseline_predictions[
        ["main_xgboost_prob_0", "main_xgboost_prob_1", "main_xgboost_prob_2"]
    ].to_numpy()

    tabpfn_proba: np.ndarray | None = None
    tabpfn_metrics: dict[str, Any] | None = None
    tabpfn_error: str | None = None
    try:
        tabpfn_proba = run_tabpfn(split_data)
        tabpfn_metrics = evaluate_predictions(split_data.y_test, tabpfn_proba, TARGET_LABELS, [0, 1, 2])
    except Exception as exc:  # pragma: no cover - 运行时依赖外部许可
        tabpfn_error = str(exc)

    raas_pinn_proba, raas_pinn_params = train_raas_pinn(split_data, output_dir, RANDOM_STATE)
    raas_pinn_metrics = evaluate_predictions(split_data.y_test, raas_pinn_proba, TARGET_LABELS, [0, 1, 2])

    comparison_rows = [
        {
            "model_name": "xgboost_baseline",
            "accuracy": baseline_metrics["accuracy"],
            "balanced_accuracy": baseline_metrics["balanced_accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
            "weighted_f1": baseline_metrics["weighted_f1"],
            "ovr_roc_auc_macro": baseline_metrics["ovr_roc_auc_macro"],
        },
        {
            "model_name": "raas_pinn",
            "accuracy": raas_pinn_metrics["accuracy"],
            "balanced_accuracy": raas_pinn_metrics["balanced_accuracy"],
            "macro_f1": raas_pinn_metrics["macro_f1"],
            "weighted_f1": raas_pinn_metrics["weighted_f1"],
            "ovr_roc_auc_macro": raas_pinn_metrics["ovr_roc_auc_macro"],
        },
    ]
    if tabpfn_metrics is not None:
        comparison_rows.append(
            {
                "model_name": "tabpfn_3",
                "accuracy": tabpfn_metrics["accuracy"],
                "balanced_accuracy": tabpfn_metrics["balanced_accuracy"],
                "macro_f1": tabpfn_metrics["macro_f1"],
                "weighted_f1": tabpfn_metrics["weighted_f1"],
                "ovr_roc_auc_macro": tabpfn_metrics["ovr_roc_auc_macro"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False
    )
    comparison_df.to_csv(tables_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")

    plot_comparison(comparison_df, figures_dir / "comparison_metrics.png")
    roc_models = {"XGBoost": baseline_proba, "RAAS-PINN": raas_pinn_proba}
    if tabpfn_proba is not None:
        roc_models["TabPFN-3"] = tabpfn_proba
    plot_model_roc_overview(split_data.y_test, roc_models, figures_dir / "comparison_roc_overview.png")

    # 单独绘制每个新模型与最优方案的混淆矩阵和 ROC
    plot_confusion_heatmap(
        split_data.y_test,
        np.argmax(raas_pinn_proba, axis=1),
        figures_dir / "raas_pinn_confusion_heatmap.png",
    )
    plot_multiclass_roc(split_data.y_test, raas_pinn_proba, figures_dir / "raas_pinn_roc_curve.png")
    if tabpfn_proba is not None:
        plot_confusion_heatmap(
            split_data.y_test,
            np.argmax(tabpfn_proba, axis=1),
            figures_dir / "tabpfn_confusion_heatmap.png",
        )
        plot_multiclass_roc(split_data.y_test, tabpfn_proba, figures_dir / "tabpfn_roc_curve.png")

    comparison_summary = {
        "baseline_xgboost_metrics": baseline_metrics,
        "tabpfn_metrics": tabpfn_metrics,
        "tabpfn_error": tabpfn_error,
        "raas_pinn_metrics": raas_pinn_metrics,
        "raas_pinn_learned_params": raas_pinn_params,
        "test_size": int(len(split_data.y_test)),
        "train_size": int(len(split_data.y_train)),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 保存测试集逐例预测
    predictions_df = split_data.X_test_raw.copy()
    predictions_df["true_label"] = split_data.y_test
    if tabpfn_proba is not None:
        predictions_df["tabpfn_pred"] = np.argmax(tabpfn_proba, axis=1)
        predictions_df["tabpfn_prob_0"] = tabpfn_proba[:, 0]
        predictions_df["tabpfn_prob_1"] = tabpfn_proba[:, 1]
        predictions_df["tabpfn_prob_2"] = tabpfn_proba[:, 2]
    predictions_df["raas_pinn_pred"] = np.argmax(raas_pinn_proba, axis=1)
    predictions_df["raas_pinn_prob_0"] = raas_pinn_proba[:, 0]
    predictions_df["raas_pinn_prob_1"] = raas_pinn_proba[:, 1]
    predictions_df["raas_pinn_prob_2"] = raas_pinn_proba[:, 2]
    predictions_df.to_excel(output_dir / "test_predictions.xlsx", index=True)

    print("实验完成")
    print(comparison_df.to_string(index=False))
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
