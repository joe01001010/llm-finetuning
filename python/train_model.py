"""
train_model.py
This file is designed to train the model
This will only train and save the lora parameters
"""
#!/usr/bin/env python


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


def create_model():
    bits_and_bytes = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config = bits_and_bytes,
        device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_cfg)
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
    max_length = 1024,
    num_train_epochs = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 2e-4,
    save_steps = 200,
    logging_steps = 10):

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
            padding="max_length",
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
    model, tokenizer = create_model()
    train_lora_sft(
        model=model,
        tokenizer=tokenizer,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
    )


if __name__ == '__main__':
    main()
