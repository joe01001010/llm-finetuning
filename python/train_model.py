"""
train_model.py
This file is designed to train the model
This will only train and save the lora parameters
"""
#!/usr/bin/env python


import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
DATA_PATH = f"{Path(__file__).resolve().parent.parent}/data/seattle_weather_chat_train.jsonl"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "adapter-weights"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA/QLoRA adapters on the Seattle weather chat dataset."
    )
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Adapter output directory. Defaults to ./adapter-weights/<mode>.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Token truncation length. 0 uses mode defaults (LoRA=512, QLoRA=1024).",
    )
    return parser.parse_args()


def get_16bit_dtype():
    """
    This function returns the highest-efficiency 16-bit dtype available
    This function falls back to float32 on CPU-only systems
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def resolve_output_dir(mode, output_dir):
    """
    This function returns the explicit output directory or a mode-specific default
    """
    if output_dir:
        return Path(output_dir)
    return OUTPUT_ROOT / mode


def load_full_precision_model(model_path, dtype):
    """
    This function loads the base model without quantization
    """
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=False,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )


def write_runtime_config(output_dir, mode, base_model_load_mode, compute_dtype):
    """
    This function writes the adapter runtime metadata for evaluation-time loading
    """
    runtime_config = {
        "adapter_mode": mode,
        "base_model_load_mode": base_model_load_mode,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "note": (
            "QLoRA uses 4-bit base-model loading. "
            "PEFT saves adapter deltas separately, so adapter file size stays similar."
        ),
    }
    runtime_path = Path(output_dir) / "adapter_runtime_config.json"
    runtime_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")


def create_model(mode, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    compute_dtype = get_16bit_dtype()

    if mode == "qlora":
        bits_and_bytes = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bits_and_bytes,
            device_map="auto"
        )
        base_model = prepare_model_for_kbit_training(base_model)
        base_model_load_mode = "4bit"
    else:
        base_model = load_full_precision_model(model_path, compute_dtype)
        base_model_load_mode = "16bit" if compute_dtype != torch.float32 else "32bit"

    adapter_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, adapter_cfg)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU is unavailable")
        sys.exit(67)

    return model, tokenizer, base_model_load_mode, compute_dtype


def train_lora_sft(
    model,
    tokenizer,
    data_path,
    output_dir,
    mode,
    base_model_load_mode,
    compute_dtype,
    max_length=1024,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    save_steps=200,
    logging_steps=10):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1) Load JSONL dataset (one JSON object per line)
    ds = load_dataset("json", data_files=data_path, split="train")

    # 2) Convert chat messages -> a single text string (prompt+response)
    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    # 3) Tokenize
    def tokenize_fn(batch):
        toks = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 4) Data collator (causal LM)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) Train
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    trainer.train()

    # 6) Save ONLY the LoRA adapters (small, reusable)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_runtime_config(output_dir, mode, base_model_load_mode, compute_dtype)

    print(f"Saved {mode.upper()} adapter to: {output_dir}")


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.mode, args.output_dir)
    max_length = args.max_length or (1024 if args.mode == "qlora" else 512)

    model, tokenizer, base_model_load_mode, compute_dtype = create_model(
        mode=args.mode,
        model_path=args.model_path,
    )
    try:
        train_lora_sft(
            model=model,
            tokenizer=tokenizer,
            data_path=args.data_path,
            output_dir=output_dir,
            mode=args.mode,
            base_model_load_mode=base_model_load_mode,
            compute_dtype=compute_dtype,
            max_length=max_length,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            suggested_max_length = max(256, max_length // 2)
            raise SystemExit(
                f"CUDA OOM during {args.mode.upper()} training.\n"
                f"Try rerunning with a smaller sequence length, for example:\n"
                f"  --max-length {suggested_max_length}\n"
                f"and optionally increase gradient accumulation (e.g. --gradient-accumulation-steps 16)."
            ) from exc
        raise


if __name__ == '__main__':
    main()
