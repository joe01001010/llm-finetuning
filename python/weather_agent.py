from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ppo_utils import (
    ensure_pad_token,
    find_lora_target_modules,
    get_compute_dtype,
    model_device,
    resolve_path,
    resolve_pretrained_source,
    tensor_dict_to_device,
)

DEFAULT_MODEL_PATH = "/local-containers/Qwen2-7B-Instruct"


class WeatherModelAgent:
    """
    Shared model/tokenizer/generation helper for SFT, PPO, and evaluation.

    The agent centralizes the transformer + PEFT loading conventions used
    throughout the repo so the training and evaluation entrypoints stay thin.
    """

    def __init__(
        self,
        *,
        base_model_path: str,
        tokenizer_path: str = "",
        trust_remote_code: bool = False,
        project_root: Path | None = None,
    ) -> None:
        self.base_model_path = base_model_path
        self.tokenizer_path = tokenizer_path
        self.trust_remote_code = trust_remote_code
        self.project_root = project_root

    def resolve_tokenizer_source(self, adapter_path: str | Path | None = None) -> str:
        if adapter_path is not None:
            resolved_adapter = resolve_path(adapter_path, self.project_root)
            if (resolved_adapter / "tokenizer_config.json").exists():
                return str(resolved_adapter)
        return self.tokenizer_path or self.base_model_path

    def load_tokenizer(
        self,
        *,
        adapter_path: str | Path | None = None,
        padding_side: str = "left",
    ):
        tokenizer_source, is_local = resolve_pretrained_source(
            self.resolve_tokenizer_source(adapter_path),
            kind="tokenizer",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
            local_files_only=is_local,
        )
        ensure_pad_token(tokenizer)
        tokenizer.padding_side = padding_side
        return tokenizer

    def load_causal_lm(
        self,
        *,
        load_mode: str,
        device_map: str | None = None,
    ) -> tuple[Any, str, torch.dtype]:
        model_source, is_local = resolve_pretrained_source(self.base_model_path, kind="model")
        compute_dtype = get_compute_dtype()

        if load_mode == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                quantization_config=quant_config,
                device_map=device_map or "auto",
                trust_remote_code=self.trust_remote_code,
                local_files_only=is_local,
            )
            return model, "4bit", compute_dtype

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": is_local,
        }
        if load_mode == "16bit" and compute_dtype != torch.float32:
            load_kwargs["torch_dtype"] = compute_dtype
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        if device_map is None and torch.cuda.is_available():
            model.to(torch.device("cuda"))
        resolved_mode = "16bit" if load_mode == "16bit" and compute_dtype != torch.float32 else load_mode
        return model, resolved_mode, compute_dtype

    def load_training_backbone(self, mode: str):
        load_mode = "4bit" if mode == "qlora" else "16bit"
        model, resolved_load_mode, compute_dtype = self.load_causal_lm(
            load_mode=load_mode,
            device_map="auto" if load_mode == "4bit" else None,
        )
        if load_mode == "4bit":
            model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        return model, resolved_load_mode, compute_dtype

    def load_model_variant(
        self,
        *,
        load_mode: str,
        adapter_path: str | Path | None = None,
        is_trainable: bool = False,
    ):
        model, resolved_load_mode, compute_dtype = self.load_causal_lm(
            load_mode=load_mode,
            device_map="auto" if load_mode == "4bit" else None,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                str(resolve_path(adapter_path, self.project_root)),
                is_trainable=is_trainable,
            )
        model.config.use_cache = False
        return model, resolved_load_mode, compute_dtype

    def enable_model_input_grads(self, model) -> None:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            return

        embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if embeddings is None:
            return

        def hook(_module, _inputs, output):
            output.requires_grad_(True)

        embeddings.register_forward_hook(hook)

    def create_lora_adapter(
        self,
        model,
        *,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        gradient_checkpointing: bool = True,
    ):
        target_modules = find_lora_target_modules(model)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        self.enable_model_input_grads(model)
        return model

    def build_bad_words_ids(self, tokenizer, forbidden_chars: str) -> list[list[int]]:
        bad_words: list[list[int]] = []
        for char in forbidden_chars:
            if char.isspace():
                continue
            token_ids = tokenizer.encode(char, add_special_tokens=False)
            if len(token_ids) == 1:
                bad_words.append(token_ids)
        return bad_words

    def build_generation_kwargs(
        self,
        tokenizer,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float = 1.0,
        forbidden_generation_chars: str = "",
        do_sample: bool | None = None,
    ) -> dict[str, Any]:
        sample = temperature > 0.0 if do_sample is None else do_sample
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": sample,
            "repetition_penalty": repetition_penalty,
            "renormalize_logits": True,
            "remove_invalid_values": True,
        }
        if sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        bad_words_ids = self.build_bad_words_ids(tokenizer, forbidden_generation_chars)
        if bad_words_ids:
            generation_kwargs["bad_words_ids"] = bad_words_ids
        return generation_kwargs

    def generate_text(
        self,
        *,
        model,
        tokenizer,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float = 1.0,
        forbidden_generation_chars: str = "",
        do_sample: bool | None = None,
    ) -> tuple[str, float]:
        inputs = tokenizer(prompt_text, return_tensors="pt")
        device = model_device(model)
        inputs = tensor_dict_to_device(inputs, device)
        prompt_width = int(inputs["input_ids"].shape[1])
        generation_kwargs = self.build_generation_kwargs(
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            forbidden_generation_chars=forbidden_generation_chars,
            do_sample=do_sample,
        )
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        latency = time.perf_counter() - start_time
        prediction = tokenizer.decode(outputs[0][prompt_width:], skip_special_tokens=True).strip()
        return prediction, latency


WeatherAgent = WeatherModelAgent
