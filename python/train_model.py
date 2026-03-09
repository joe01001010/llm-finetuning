"""
train_model.py
This file is designed to train the model
This will only train and save the lora parameters
"""
#!/usr/bin/env python


import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
DATA_PATH = f"{Path(__file__).resolve().parent.parent}/data/seattle_weather_chat.json"
OUTPUT_DIR = "./lora-weights"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA/QLoRA adapters on the Seattle weather chat dataset."
    )
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
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


def create_model(mode, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if mode == "qlora":
        bits_and_bytes = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bits_and_bytes,
            device_map="auto"
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
        )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_cfg)
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

    return model, tokenizer


def train_lora_sft(
    model,
    tokenizer,
    data_path,
    output_dir,
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
    #    Your dataset already includes the assistant response, so generation_prompt=False.
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
        # Trainer will use labels for causal LM; copy input_ids
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved LoRA adapter to: {output_dir}")


def main():
    args = parse_args()
    max_length = args.max_length or (512 if args.mode == "lora" else 1024)

    model, tokenizer = create_model(mode=args.mode, model_path=args.model_path)
    try:
        train_lora_sft(
            model=model,
            tokenizer=tokenizer,
            data_path=args.data_path,
            output_dir=args.output_dir,
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
