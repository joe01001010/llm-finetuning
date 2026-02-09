#!/usr/bin/env python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
  model_name = "Qwen/Qwen2.5-7B"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=torch.float16,
      device_map="auto",
      low_cpu_mem_usage=True
    )
  print("Model Downloaded Successfully")


if __name__ == '__main__':
  main()
