from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from frontier_augmentation import POST_TREATMENT_COLUMNS, SCREENING_COLUMNS, TREATMENT_COLUMNS, clip_numeric_value, infer_discrete_numeric_features
from frontier_scm_v2_experiment import DEFAULT_SEEDS, ProgressTracker, prepare_split_data
from frontier_scm_v3_experiment import fit_scm_v3_config, fit_xgb_with_safe_adasyn
from scm_v3_augmentation import evaluate_model_from_raw, summarize_model_metrics


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "tabddpm_proto_outputs"
DEVICE = torch.device("cpu")
SCM_V3_REFERENCE_CONFIG = {
    "seed_strategy": "hard_case_seed",
    "target_classes": "class0_only",
    "treat_mix_prob": 0.40,
    "sampler_strength": 0.65,
    "node_sampler": "conditional_kde",
    "rule_filter_mode": "medical_rules",
    "teacher_filter_mode": "single",
    "curriculum_mode": "off",
    "counterfactual_mode": "off",
}


class Denoiser(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden = max(32, input_dim * 2)
        self.time_embed = nn.Embedding(64, 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_feat = self.time_embed(t)
        return self.net(torch.cat([x, time_feat], dim=1))


@dataclass
class TabDDPMPrototype:
    feature_columns: list[str]
    mean_: np.ndarray
    std_: np.ndarray
    model: Denoiser

    def sample(self, n_samples: int, steps: int, seed: int) -> np.ndarray:
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(seed)
        x = torch.randn((n_samples, len(self.feature_columns)), generator=rng, device=DEVICE)
        for step in reversed(range(steps)):
            t = torch.full((n_samples,), step, dtype=torch.long, device=DEVICE)
            noise_pred = self.model(x, t)
            alpha = 0.90 + 0.09 * (step / max(steps - 1, 1))
            x = (x - (1 - alpha) * noise_pred) / max(alpha, 1e-4)
            if step > 0:
                x = x + 0.03 * torch.randn_like(x, generator=rng)
        sampled = x.detach().cpu().numpy()
        return sampled * self.std_ + self.mean_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabDDPM 原型实验。")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    return parser.parse_args()


def build_numeric_generation_columns(X_train_raw: pd.DataFrame) -> list[str]:
    ordered = TREATMENT_COLUMNS + SCREENING_COLUMNS + POST_TREATMENT_COLUMNS
    cols = [col for col in ordered if col in X_train_raw.columns and pd.api.types.is_numeric_dtype(X_train_raw[col])]
    return list(dict.fromkeys(cols))


def fit_tabddpm_proto(class_df: pd.DataFrame, feature_columns: list[str], epochs: int, seed: int) -> TabDDPMPrototype:
    rng = np.random.default_rng(seed)
    X = class_df[feature_columns].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    mean_ = X.mean(axis=0).to_numpy(dtype=np.float32)
    std_ = X.std(axis=0, ddof=0).replace(0, 1.0).to_numpy(dtype=np.float32)
    X_scaled = ((X - mean_) / std_).to_numpy(dtype=np.float32)

    torch.manual_seed(seed)
    model = Denoiser(input_dim=len(feature_columns)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
    n_steps = 24

    for _ in range(epochs):
        batch_indices = rng.integers(0, len(X_tensor), size=min(32, len(X_tensor)))
        batch = X_tensor[batch_indices]
        t = torch.randint(0, n_steps, (len(batch),), device=DEVICE)
        noise = torch.randn_like(batch)
        alpha = 0.90 + 0.09 * (t.float() / max(n_steps - 1, 1))
        alpha = alpha.unsqueeze(1)
        noisy = alpha * batch + (1 - alpha) * noise
        noise_pred = model(noisy, t)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return TabDDPMPrototype(feature_columns=feature_columns, mean_=mean_, std_=std_, model=model)


def generate_tabddpm_rows(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int,
    epochs: int,
    smoke_test: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    class_zero = X_train_raw.loc[y_train == 0].copy()
    if len(class_zero) < 6:
        return pd.DataFrame(columns=X_train_raw.columns), pd.Series(dtype=int, name=y_train.name)

    feature_columns = build_numeric_generation_columns(X_train_raw)
    if not feature_columns:
        return pd.DataFrame(columns=X_train_raw.columns), pd.Series(dtype=int, name=y_train.name)

    prototype = fit_tabddpm_proto(class_zero, feature_columns, epochs=20 if smoke_test else epochs, seed=seed)
    n_samples = min(18 if smoke_test else max(len(class_zero) * 2, 24), 96)
    sampled_numeric = prototype.sample(n_samples=n_samples, steps=12 if smoke_test else 24, seed=seed + 77)
    sampled_df = pd.DataFrame(sampled_numeric, columns=feature_columns)
    support_cache = {col: pd.to_numeric(X_train_raw[col], errors="coerce") for col in feature_columns}
    discrete_features = set(infer_discrete_numeric_features(X_train_raw))

    rows: list[pd.Series] = []
    rng = np.random.default_rng(seed + 99)
    for i in range(n_samples):
        base_row = class_zero.iloc[int(rng.integers(0, len(class_zero)))].copy()
        candidate = base_row.copy()
        for col in feature_columns:
            value = float(sampled_df.iloc[i][col])
            value = clip_numeric_value(value, support_cache[col])
            if col in discrete_features:
                value = float(np.round(value))
            candidate[col] = value
        rows.append(candidate)

    augmented_df = pd.DataFrame(rows).reset_index(drop=True)
    y_aug = pd.Series(np.zeros(len(augmented_df), dtype=int), name=y_train.name)
    return augmented_df, y_aug


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    tracker = ProgressTracker(OUTPUT_DIR, total_steps=len(seeds) * 3)
    rows: list[dict] = []

    for seed in seeds:
        split_data = prepare_split_data(PROJECT_DIR, seed)

        baseline = fit_xgb_with_safe_adasyn(
            split_data.X_train_raw,
            split_data.y_train,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed=seed,
        )
        rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_reference_adasyn",
                y_test=split_data.y_test,
                proba=baseline.proba,
                augmented_size=0,
                train_size=baseline.train_size,
                resampled_train_size=baseline.resampled_train_size,
            )
        )
        tracker.log("完成 baseline", stage="baseline_done", advance=1, context={"seed": seed})

        scm_v3_result, scm_v3_aug_size, _ = fit_scm_v3_config(split_data, seed, SCM_V3_REFERENCE_CONFIG)
        rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_scm_v3_reference",
                y_test=split_data.y_test,
                proba=scm_v3_result.proba,
                augmented_size=scm_v3_aug_size,
                train_size=scm_v3_result.train_size,
                resampled_train_size=scm_v3_result.resampled_train_size,
            )
        )
        tracker.log("完成 SCM_v3 参照", stage="scm_v3_done", advance=1, context={"seed": seed})

        X_aug, y_aug = generate_tabddpm_rows(
            split_data.X_train_raw,
            split_data.y_train,
            seed=seed,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
        )
        X_train_aug = pd.concat([split_data.X_train_raw, X_aug], axis=0, ignore_index=True)
        y_train_aug = pd.concat([split_data.y_train, y_aug], axis=0, ignore_index=True)
        ddpm_model = fit_xgb_with_safe_adasyn(
            X_train_aug,
            y_train_aug,
            split_data.X_test_raw,
            split_data.discrete_numeric_features,
            seed=seed,
        )
        rows.append(
            evaluate_model_from_raw(
                seed=seed,
                model_name="xgb_tabddpm_proto",
                y_test=split_data.y_test,
                proba=ddpm_model.proba,
                augmented_size=int(len(X_aug)),
                train_size=ddpm_model.train_size,
                resampled_train_size=ddpm_model.resampled_train_size,
            )
        )
        tracker.log("完成 TabDDPM 原型", stage="tabddpm_done", advance=1, context={"seed": seed, "augmented_size": len(X_aug)})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables_dir / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize_model_metrics(metrics_df, metrics_df["model_name"].unique().tolist())
    summary_df.to_csv(tables_dir / "metrics_mean_std.csv", index=False, encoding="utf-8-sig")
    summary = {
        "seeds": seeds,
        "smoke_test": args.smoke_test,
        "epochs": 20 if args.smoke_test else args.epochs,
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
