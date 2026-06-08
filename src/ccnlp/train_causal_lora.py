from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from ccnlp.causal_sft import preprocess_example


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen Causal LM with LoRA for Lu Xun style transfer.")
    parser.add_argument("--train_file", required=True, help="JSONL training file")
    parser.add_argument("--validation_file", default=None, help="Optional validation JSONL file")
    parser.add_argument("--dataset_format", choices=["filtered", "luxungpt"], default="filtered")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=DEFAULT_LORA_TARGET_MODULES,
        help="Transformer module names to adapt with LoRA.",
    )
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use reentrant checkpointing. Keep False for DDP + LoRA stability.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_path(args.train_file, "Training file")
    if args.validation_file:
        _validate_path(args.validation_file, "Validation file")

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing Qwen LoRA training dependencies. Install them first:\n"
            "pip install transformers datasets accelerate peft bitsandbytes"
        ) from exc

    local_rank = local_rank_from_env()
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    raw_datasets = load_dataset("json", data_files=data_files)
    if args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(args.max_train_samples, len(raw_datasets["train"])))
        )

    def preprocess(row: dict[str, Any]) -> dict[str, list[int]]:
        if args.dataset_format == "luxungpt":
            return _preprocess_luxungpt_example(row, tokenizer, args.max_seq_length)
        return preprocess_example(row, tokenizer, args.max_seq_length)

    tokenized = raw_datasets.map(
        preprocess,
        remove_columns=raw_datasets["train"].column_names,
    )
    has_eval = "validation" in tokenized

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map=resolve_device_map(),
        quantization_config=quantization_config,
    )
    model = prepare_model_for_training(model, args, prepare_model_for_kbit_training)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if has_eval else None,
        eval_strategy="steps" if has_eval else "no",
        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_eval else None,
        tokenizer=tokenizer,
        data_collator=CausalDataCollator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


class CausalDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _preprocess_luxungpt_example(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt = str(example["context"])
    target = str(example["target"]).strip()
    full_text = prompt + target
    if tokenizer.eos_token:
        full_text += tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
    label_start = min(len(prompt_ids), len(full_ids))
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * label_start + full_ids[label_start:],
    }


def _validate_path(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def prepare_model_for_training(model: Any, args: argparse.Namespace, prepare_model_for_kbit_training: Any) -> Any:
    if args.gradient_checkpointing:
        model.config.use_cache = False

    checkpointing_kwargs = {
        "use_reentrant": args.gradient_checkpointing_use_reentrant,
    }
    if args.load_in_4bit:
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=checkpointing_kwargs,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpointing_kwargs,
        )
    return model


def local_rank_from_env() -> int | None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return None
    return int(local_rank)


def resolve_device_map() -> str | dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = local_rank_from_env()
    if world_size > 1 and local_rank is not None:
        return {"": local_rank}
    return "auto"


if __name__ == "__main__":
    main()
