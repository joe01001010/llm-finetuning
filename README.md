# llm-finetuning

End-to-end local workflow for Seattle weather fine-tuning with:
- Stage 1 supervised fine-tuning (SFT) using `LoRA` or `QLoRA`
- Stage 2 PPO post-training using structured weather JSON rewards
- Evaluation of base, SFT, and PPO-tuned adapters side by side

## Project Layout

- `python/pull_model.py`: pulls the base model locally.
- `python/data_formatting.py`: builds the Seattle weather chat datasets.
- `python/train_model.py`: Stage 1 SFT training for LoRA / QLoRA adapters.
- `python/train_ppo_weather.py`: Stage 2 PPO post-training on prompt-only weather data.
- `python/evaluate_models.py`: legacy SFT comparison for base vs LoRA vs QLoRA.
- `python/evaluate_ppo_weather.py`: unified comparison for base, SFT, and PPO models.
- `python/ppo_data.py`: prompt-only weather dataset loading for PPO and evaluation.
- `python/ppo_reward.py`: structured reward functions for weather JSON quality.
- `python/ppo_utils.py`: PPO masking, KL, logprob, and GAE helpers.
- `python/ppo_value_head.py`: transformer-native value head wrapper.
- `python/ppo_memory.py`: rollout container for PPO batches.
- `python/ppo_classes.py`: conceptual PPO reference from a separate non-LLM RL project. This is not used directly for language-model training.

## Workflow

### 1. Pull the base model

```bash
cd /llm-finetuning
python python/pull_model.py
```

### 2. Build the weather datasets

```bash
cd /llm-finetuning
python python/data_formatting.py
```

Outputs:
- `/llm-finetuning/data/seattle_weather_chat.jsonl`
- `/llm-finetuning/data/seattle_weather_chat_train.jsonl`
- `/llm-finetuning/data/seattle_weather_chat_eval.jsonl`

### 3. Train SFT adapters

#### QLoRA SFT

```bash
cd /llm-finetuning
python python/train_model.py \
  --mode qlora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --data-path /llm-finetuning/data/seattle_weather_chat_train.jsonl \
  --output-dir /llm-finetuning/adapter-weights/sft-qlora \
  --num-train-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-length 768
```

#### LoRA SFT

```bash
cd /llm-finetuning
python python/train_model.py \
  --mode lora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --data-path /llm-finetuning/data/seattle_weather_chat_train.jsonl \
  --output-dir /llm-finetuning/adapter-weights/sft-lora \
  --num-train-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-length 768
```

SFT outputs include:
- adapter weights
- tokenizer files
- `adapter_runtime_config.json`
- `sft_training_summary.json`

### 4. Run PPO post-training

PPO uses prompt-only rollouts. The reference JSON is used only for the reward function and evaluation, not as supervised labels during PPO optimization.

#### PPO from an SFT QLoRA adapter

```bash
cd /llm-finetuning
python python/train_ppo_weather.py \
  --mode qlora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --adapter-path /llm-finetuning/adapter-weights/sft-qlora \
  --data-path /llm-finetuning/data/seattle_weather_chat_train.jsonl \
  --output-dir /llm-finetuning/adapter-weights/ppo-qlora \
  --num-train-epochs 1 \
  --rollout-batch-size 1 \
  --mini-batch-size 1 \
  --ppo-epochs 4 \
  --max-prompt-length 512 \
  --max-new-tokens 48
```

#### PPO from an SFT LoRA adapter

```bash
cd /llm-finetuning
python python/train_ppo_weather.py \
  --mode lora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --adapter-path /llm-finetuning/adapter-weights/sft-lora \
  --data-path /llm-finetuning/data/seattle_weather_chat_train.jsonl \
  --output-dir /llm-finetuning/adapter-weights/ppo-lora \
  --num-train-epochs 1 \
  --rollout-batch-size 1 \
  --mini-batch-size 1 \
  --ppo-epochs 4 \
  --max-prompt-length 512 \
  --max-new-tokens 48
```

#### PPO directly from the base model with a fresh LoRA adapter

```bash
cd /llm-finetuning
python python/train_ppo_weather.py \
  --mode base \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --data-path /llm-finetuning/data/seattle_weather_chat_train.jsonl \
  --output-dir /llm-finetuning/adapter-weights/ppo-base \
  --num-train-epochs 1 \
  --rollout-batch-size 1 \
  --mini-batch-size 1 \
  --ppo-epochs 4
```

PPO outputs include:
- adapter weights
- tokenizer files
- `adapter_runtime_config.json`
- `ppo_runtime_config.json`
- `ppo_reward_config.json`
- `ppo_value_head.pt`
- `ppo_training_summary.json`

## Reward Design

The default PPO reward is modular and lives in `python/ppo_reward.py`. It reuses the repo’s structured JSON scoring style and includes:
- reward for valid JSON
- reward for schema coverage
- reward for field accuracy
- bonus for exact JSON match
- penalties for hallucinated / extra / missing fields
- penalties for malformed JSON and empty outputs
- optional verbosity penalty for non-JSON spillover text

All reward weights are configurable from the CLI, for example:
- `--json-valid-weight`
- `--schema-weight`
- `--field-accuracy-weight`
- `--hallucination-penalty`
- `--malformed-json-penalty`

## Evaluation

### Legacy SFT comparison

```bash
cd /llm-finetuning
python python/evaluate_models.py \
  --base-model /local-containers/Qwen2-7B-Instruct \
  --lora-adapter /llm-finetuning/adapter-weights/sft-lora \
  --qlora-adapter /llm-finetuning/adapter-weights/sft-qlora \
  --dataset /llm-finetuning/data/seattle_weather_chat_eval.jsonl \
  --output-report /llm-finetuning/reports/base_vs_sft.json \
  --output-predictions /llm-finetuning/reports/base_vs_sft_predictions.jsonl
```

### Unified base vs SFT vs PPO comparison

```bash
cd /llm-finetuning
python python/evaluate_ppo_weather.py \
  --base-model /local-containers/Qwen2-7B-Instruct \
  --dataset /llm-finetuning/data/seattle_weather_chat_eval.jsonl \
  --sft-lora-adapter /llm-finetuning/adapter-weights/sft-lora \
  --sft-qlora-adapter /llm-finetuning/adapter-weights/sft-qlora \
  --ppo-lora-adapter /llm-finetuning/adapter-weights/ppo-lora \
  --ppo-qlora-adapter /llm-finetuning/adapter-weights/ppo-qlora \
  --output-report /llm-finetuning/reports/weather_ppo_eval.json \
  --output-predictions /llm-finetuning/reports/weather_ppo_eval_predictions.jsonl \
  --max-samples 200
```

Evaluation metrics include:
- `accuracy`
- `hallucination_rate`
- `a1_score`
- `exact_json_match`
- `avg_latency_sec`
- `avg_reward`
- `exact_match`
- `token_f1`

Evaluation outputs:
- summary JSON report
- optional merged per-sample predictions JSONL

## Notes

- PPO is implemented as an additional stage and does not replace SFT.
- `train_ppo_weather.py` uses a transformer-native policy plus a scalar value head, not the MLP actor/critic from `ppo_classes.py`.
- PPO checkpoints remain adapter-based so the workflow stays memory-conscious for single-GPU use.
- For `LoRA` PPO, the reference model defaults to 4-bit loading when possible to reduce memory pressure.
