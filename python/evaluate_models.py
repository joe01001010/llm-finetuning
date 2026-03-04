"""
This program will evaluate the models against the base model
"""
#!/usr/bin/env python


import json
import math
from typing import Dict, Any, List, Tuple
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"
ADAPTER_PATH = "./lora-weights"
DATA_PATH = "./data/seattle_weather_chat.json"
REQUIRED_KEYS = ["temp_max", "temp_min", "precipitation", "wind", "weather"]


def main():
    


if __name__ == '__main__':
    main()
