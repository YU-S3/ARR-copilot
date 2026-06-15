from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from tablegpt2_pa.common import (
    TARGET_COLUMN,
    build_binary_dataset,
    build_prompt_examples,
    compute_binary_metrics,
    ensure_directory,
    load_clean_frame,
    parse_json_response,
    rank_features,
    split_with_icl_pools,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="使用 TableGPT2/Qwen 公开 checkpoint 进行 PA QLoRA 微调。")
    parser.add_argument("--input", type=Path, default=project_dir / "数据表格测试.xlsx")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "tablegpt2_pa_outputs" / "qlora_main",
    )
    parser.add_argument("--model-name-or-path", default="tablegpt/TableGPT2-7B")
    parser.add_argument("--protocol", choices=["A", "B", "C"], default="A")
    parser.add_argument("--top-p", type=int, default=16)
    parser.add_argument("--k-shot", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=32)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="远程模型仓库镜像端点，默认使用 hf-mirror。",
    )
    return parser.parse_args()


def build_training_dataset(examples: list[Any]) -> Dataset:
    records = [{"text": f"{example.prompt}\n{example.response}"} for example in examples]
    return Dataset.from_list(records)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        result = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    resolved_model_path = resolve_model_path(args)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
        if args.use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
            model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["device_map"] = None
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(resolved_model_path, **model_kwargs)
    model.config.use_cache = False

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = infer_lora_target_modules(model)
    lora_config = LoraConfig(
        r=64 if not args.smoke_test else 8,
        lora_alpha=128 if not args.smoke_test else 16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    max_positions = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", args.max_seq_length
    )
    tokenizer_limit = getattr(tokenizer, "model_max_length", args.max_seq_length)
    if tokenizer_limit is None or tokenizer_limit > 100_000:
        tokenizer_limit = args.max_seq_length
    effective_max_length = min(args.max_seq_length, int(max_positions), int(tokenizer_limit))
    return tokenizer, model, effective_max_length


def resolve_model_path(args: argparse.Namespace) -> str:
    model_path = Path(args.model_name_or_path)
    if model_path.exists():
        return str(model_path)

    cache_dir = ensure_directory(args.output_dir / "hf_model_cache" / model_path.name)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    snapshot_download(
        repo_id=args.model_name_or_path,
        local_dir=str(cache_dir),
        token=args.hf_token,
        endpoint=args.hf_endpoint,
    )
    return str(cache_dir)


def infer_lora_target_modules(model: AutoModelForCausalLM) -> list[str]:
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    qwen_candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    gpt2_candidates = ["c_attn", "c_proj", "c_fc"]
    if {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(module_names):
        return [name for name in qwen_candidates if name in module_names]
    if "c_attn" in module_names:
        return [name for name in gpt2_candidates if name in module_names]
    raise ValueError("未识别到可用的 LoRA target modules，请检查模型结构。")


def generate_predictions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[Any],
    output_path: Path,
    max_eval_samples: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    records: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    selected_examples = examples[:max_eval_samples]
    device = next(model.parameters()).device
    for example in selected_examples:
        inputs = tokenizer(example.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        parse_error = None
        try:
            parsed = parse_json_response(generated)
            probability = float(parsed.get("probability", 0.5))
            pred_label = int(parsed.get("label", 0))
        except Exception as exc:  # pragma: no cover - 用于兼容小模型烟雾测试
            parsed = {"label": 0, "probability": 0.5, "reasoning": "模型输出未能解析为 JSON。"}
            probability = 0.5
            pred_label = 0
            parse_error = str(exc)

        y_true.append(int(example.label))
        y_pred.append(pred_label)
        y_prob.append(probability)
        records.append(
            {
                "sample_id": example.sample_id,
                "label_true": int(example.label),
                "label_pred": pred_label,
                "probability": probability,
                "prompt": example.prompt,
                "generated_text": generated,
                "parsed_json": json.dumps(parsed, ensure_ascii=False),
                "parse_error": parse_error,
            }
        )

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    prediction_df = pd.DataFrame(records)
    prediction_df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return metrics, prediction_df


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    predictions_dir = ensure_directory(output_dir / "predictions")
    artifacts_dir = ensure_directory(output_dir / "artifacts")

    frame = load_clean_frame(args.input)
    dataset = build_binary_dataset(frame, protocol=args.protocol)
    splits = split_with_icl_pools(dataset.frame, dataset.label_column, random_state=args.random_state)
    selected_features = rank_features(splits.train_df, dataset.label_column, top_p=args.top_p)

    train_examples = build_prompt_examples(
        splits.train_df,
        splits.icl_train_df,
        selected_features,
        dataset.label_column,
        k_shot=args.k_shot,
        seed=args.random_state,
    )
    val_examples = build_prompt_examples(
        splits.val_df,
        splits.icl_val_df,
        selected_features,
        dataset.label_column,
        k_shot=args.k_shot,
        seed=args.random_state + 1000,
    )
    test_examples = build_prompt_examples(
        splits.test_df,
        splits.icl_test_df,
        selected_features,
        dataset.label_column,
        k_shot=args.k_shot,
        seed=args.random_state + 2000,
    )

    if args.max_train_samples:
        train_examples = train_examples[: args.max_train_samples]
    if args.smoke_test:
        train_examples = train_examples[: min(12, len(train_examples))]
        val_examples = val_examples[: min(6, len(val_examples))]
        test_examples = test_examples[: min(6, len(test_examples))]

    tokenizer, model, effective_max_length = load_model_and_tokenizer(args)
    train_dataset = tokenize_dataset(
        build_training_dataset(train_examples),
        tokenizer=tokenizer,
        max_seq_length=effective_max_length,
    )
    val_dataset = tokenize_dataset(
        build_training_dataset(val_examples),
        tokenizer=tokenizer,
        max_seq_length=effective_max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1 if args.smoke_test else 10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=(not args.bf16) and torch.cuda.is_available(),
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    adapter_dir = ensure_directory(output_dir / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    test_metrics, prediction_df = generate_predictions(
        model=trainer.model,
        tokenizer=tokenizer,
        examples=test_examples,
        output_path=predictions_dir / "test_predictions.json",
        max_eval_samples=args.max_eval_samples if not args.smoke_test else min(args.max_eval_samples, 6),
    )

    gray_metrics = None
    gray_prediction_count = 0
    if dataset.gray_zone_frame is not None and not dataset.gray_zone_frame.empty:
        gray_frame = dataset.gray_zone_frame.copy()
        gray_frame[dataset.label_column] = 1
        gray_examples = build_prompt_examples(
            gray_frame,
            splits.icl_test_df,
            selected_features,
            dataset.label_column,
            k_shot=args.k_shot,
            seed=args.random_state + 3000,
        )
        gray_metrics, gray_predictions = generate_predictions(
            model=trainer.model,
            tokenizer=tokenizer,
            examples=gray_examples,
            output_path=predictions_dir / "gray_zone_predictions.json",
            max_eval_samples=args.max_eval_samples if not args.smoke_test else min(args.max_eval_samples, 6),
        )
        gray_prediction_count = len(gray_predictions)

    summary = {
        "input_file": str(args.input),
        "model_name_or_path": args.model_name_or_path,
        "protocol": args.protocol,
        "selected_features": selected_features,
        "effective_max_length": effective_max_length,
        "target_column": TARGET_COLUMN,
        "dataset_sizes": {
            "train": len(splits.train_df),
            "val": len(splits.val_df),
            "test": len(splits.test_df),
            "icl_train": len(splits.icl_train_df),
            "icl_val": len(splits.icl_val_df),
            "icl_test": len(splits.icl_test_df),
        },
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "gray_zone_predictions": gray_prediction_count,
        "test_metrics": test_metrics,
        "gray_zone_metrics": gray_metrics,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prediction_df.to_excel(output_dir / "test_predictions.xlsx", index=False)
    pd.DataFrame([summary]).to_json(artifacts_dir / "run_manifest.json", orient="records", force_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
