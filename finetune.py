#!/usr/bin/env python
# finetune.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import json
import pandas as pd

def load_dummy_data(file_path="dummy_training_data.json"):
  """Load dummy dataset"""
  try:
    with open(file_path, 'r') as f:
      data = json.load(f)
    
    # Format for training
    formatted_data = []
    for item in data:
      text = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Response: {item['output']}"
      formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)
  
  except FileNotFoundError:
    print(f"File {file_path} not found. Creating dummy data...")
    # Create dummy data on the fly
    from create_fake_data import create_dummy_dataset
    dummy_data = create_dummy_dataset(num_samples=20)
    
    formatted_data = []
    for item in dummy_data:
      text = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Response: {item['output']}"
      formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def main():
  print("Starting fine-tuning with dummy data...")
  
  # 1. Load model and tokenizer (use smaller model for testing)
  # Switch to smaller model for testing
  model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Much smaller, faster for testing
  # model_name = "Qwen/Qwen2.5-7B"  # Original - use for real training
  
  print(f"Loading model: {model_name}")
  
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  # Set padding token
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  
  # Load model with 4-bit quantization for memory efficiency
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      load_in_4bit=True,  # Use 4-bit for testing
      device_map="auto",
      torch_dtype=torch.float16,
      trust_remote_code=True
  )
  
  print("Model loaded successfully")
  
  # 2. Configure LoRA
  lora_config = LoraConfig(
      r=8,  # Smaller rank for testing
      lora_alpha=16,
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
      lora_dropout=0.1,
      bias="none",
      task_type="CAUSAL_LM"
  )
  
  # Prepare model for training
  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, lora_config)
  
  # Print trainable parameters
  model.print_trainable_parameters()
  
  # 3. Prepare dataset
  print("Loading dummy dataset...")
  dataset = load_dummy_data()
  
  # Split into train and eval
  split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
  train_dataset = split_dataset["train"]
  eval_dataset = split_dataset["test"]
  
  print(f"Train samples: {len(train_dataset)}")
  print(f"Eval samples: {len(eval_dataset)}")
  
  # 4. Tokenization function
  def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,  # Shorter for testing
        padding="max_length"
    )
  
  # Tokenize datasets
  print("Tokenizing datasets...")
  train_tokenized = train_dataset.map(tokenize_function, batched=True)
  eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
  
  # 5. Training arguments (optimized for testing)
  training_args = TrainingArguments(
      output_dir="./fine_tuned_model",
      num_train_epochs=1,  # Just 1 epoch for testing
      per_device_train_batch_size=1,  # Small batch for testing
      gradient_accumulation_steps=4,
      learning_rate=1e-4,
      fp16=True,
      logging_steps=5,
      save_steps=25,
      eval_steps=25,
      evaluation_strategy="steps",
      save_total_limit=2,
      warmup_steps=10,
      report_to="none",  # Disable wandb for testing
      optim="paged_adamw_8bit",
      gradient_checkpointing=True,
      remove_unused_columns=False
  )
  
  # 6. Data collator
  data_collator = DataCollatorForLanguageModeling(
      tokenizer=tokenizer,
      mlm=False
  )
  
  # 7. Create trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_tokenized,
      eval_dataset=eval_tokenized,
      data_collator=data_collator,
      tokenizer=tokenizer,
  )
  
  # 8. Train
  print("Starting training...")
  trainer.train()
  
  # 9. Save model
  print("Saving model...")
  model.save_pretrained("./fine_tuned_model")
  tokenizer.save_pretrained("./fine_tuned_model")
  
  # Save training arguments
  trainer.save_model("./fine_tuned_model")
  
  print("Fine-tuning complete! Model saved to './fine_tuned_model'")

if __name__ == '__main__':
  main()
