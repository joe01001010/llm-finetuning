#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig
import torch


def main():
  model_name = "Qwen/Qwen2.5-7B"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token
  
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4"
  )
  
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      device_map="auto",
      torch_dtype=torch.bfloat16
  )
  print("Model loaded successfully")


if __name__ == '__main__':
  main()
