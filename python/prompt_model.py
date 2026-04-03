#!/usr/bin/env python
"""
Small smoke test for the locally pulled base model.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppo_utils import resolve_pretrained_source
from weather_agent import DEFAULT_MODEL_PATH


MODEL_PATH = DEFAULT_MODEL_PATH
TOKENIZER_SOURCE, TOKENIZER_IS_LOCAL = resolve_pretrained_source(MODEL_PATH, kind="tokenizer")
TOKENIZER = AutoTokenizer.from_pretrained(
    TOKENIZER_SOURCE,
    use_fast=True,
    local_files_only=TOKENIZER_IS_LOCAL,
)


def ask(user_text: str, model, max_new_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    model_source, is_local = resolve_pretrained_source(MODEL_PATH, kind="model")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=is_local,
    )
    model.eval()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print(ask("Give me a one-sentence summary of what PPO is.", model))


if __name__ == "__main__":
    main()
