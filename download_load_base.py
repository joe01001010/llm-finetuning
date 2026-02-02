#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def main():
  if torch.cuda.is_available():
    print("Using GPU")
  else:
    print("Using CPU")
  model_name = "Qwen/Qwen2.5-7B"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    dtype=torch.bfloat16
  )

  print("Model Loaded")


if __name__ == '__main__':
  main()
