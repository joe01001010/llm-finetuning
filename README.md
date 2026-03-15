# llm-finetuning

End-to-end local workflow for:
- pulling a base LLM into a container-mounted path
- preparing a weather prediction chat dataset
- training adapters with PPO using `LoRA` and `QLoRA`
- evaluating base model vs adapter model outputs

## Project Capabilities

- Containerized GPU workflow using `nvcr.io/nvidia/pytorch:25.04-py3`
- Base model pull from Hugging Face (`Qwen/Qwen2-7B-Instruct`)
- Dataset pull + formatting from Kaggle into chat fine-tuning JSONL
- PPO-based adapter training with:
  - prompt-only rollouts
  - structured rewards computed against the target JSON
  - KL control against a frozen reference model
  - `LoRA` (16-bit base model loading)
  - `QLoRA` (4-bit base model loading)
- Side-by-side evaluation of base vs base+adapter with latency and optional EM/F1

## Repository Layout

- `containers/setup_container.sh`: installs Docker + NVIDIA container toolkit and pulls the base image.
- `python/pull_model.py`: downloads the base model to `/local-containers/Qwen2-7B-Instruct`.
- `python/data_formatting.py`: pulls weather dataset from Kaggle and writes combined, train, and eval chat datasets under `data/`.
- `python/train_model.py`: trains PPO adapters (`--mode lora` or `--mode qlora`).
- `python/evaluate_models.py`: compares base vs LoRA vs QLoRA in one run.
- `adapter-weights/`: default adapter output location.

## Prerequisites

- Linux host (Ubuntu/apt expected for `setup_container.sh`).
- NVIDIA GPU + drivers.
- Internet access for container/model/dataset downloads.
- Kaggle credentials for dataset pull.
  - Set env vars: `KAGGLE_USERNAME`, `KAGGLE_KEY`, or
  - place `kaggle.json` under `~/.kaggle/kaggle.json` on host/container.

## 1) Setup Container Runtime + Pull Base Container

From repo root:

```bash
cd llm-finetuning
sudo ./containers/setup_container.sh "$USER"
```

This script installs Docker + NVIDIA runtime, adds your user to the `docker` group, and pulls:
- `nvcr.io/nvidia/pytorch:25.04-py3`

If group changes do not apply immediately, log out and back in.

## 2) Start the Container

### Fresh run (new container)

```bash
cd llm-finetuning
docker rm -f llm-finetuning 2>/dev/null || true
docker run -it \
  -v /home/$USER/.cache:/root/.cache \
  -v /tmp:/local-containers \
  -v "$(pwd)":/llm-finetuning \
  --name llm-finetuning \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### Start an existing stopped container

```bash
docker start llm-finetuning
docker exec -it llm-finetuning /bin/bash
```

## 3) Once Inside the Container

```bash
cd /llm-finetuning
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 4) Pull the Base Model

```bash
cd /llm-finetuning
python python/pull_model.py
```

Default location:
- `/local-containers/Qwen2-7B-Instruct`

## 5) Pull + Format Dataset

```bash
cd /llm-finetuning
python python/data_formatting.py
```

Output:
- `/llm-finetuning/data/seattle_weather_chat.jsonl`
- `/llm-finetuning/data/seattle_weather_chat_train.jsonl`
- `/llm-finetuning/data/seattle_weather_chat_eval.jsonl`
- `/llm-finetuning/data/seattle_weather_chat.json` (legacy combined output)

## 6) Train PPO Adapters

`train_model.py` now uses PPO as the core training algorithm:
- input to the policy is prompt-only chat context
- the model samples a response
- the response is scored against the ground-truth weather JSON
- PPO updates the policy while a frozen reference model supplies KL control

Modes:
- `qlora`: 4-bit quantized base model with trainable LoRA adapters
- `lora`: 16-bit base model with trainable LoRA adapters

### Train QLoRA with PPO

```bash
cd /llm-finetuning
python python/train_model.py \
  --mode qlora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --data-path /llm-finetuning/data/seattle_weather_chat_train.jsonl \
  --output-dir /llm-finetuning/adapter-weights/qlora \
  --num-train-epochs 1 \
  --batch-size 1 \
  --mini-batch-size 1 \
  --max-prompt-length 512 \
  --max-new-tokens 48
```

### Train LoRA with PPO

```bash
cd /llm-finetuning
python python/train_model.py \
  --mode lora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --data-path /llm-finetuning/data/seattle_weather_chat_train.jsonl \
  --output-dir /llm-finetuning/adapter-weights/lora \
  --num-train-epochs 1 \
  --batch-size 1 \
  --mini-batch-size 1 \
  --max-prompt-length 512 \
  --max-new-tokens 48
```

Common tunables:
- `--batch-size`
- `--mini-batch-size`
- `--gradient-accumulation-steps`
- `--learning-rate`
- `--max-prompt-length`
- `--max-new-tokens`
- `--ppo-epochs`
- `--init-kl-coef`

Training artifacts:
- adapter weights and tokenizer files
- `adapter_runtime_config.json`
- `ppo_value_head.pt`
- `ppo_training_summary.json`

## 7) Evaluate Base vs LoRA vs QLoRA

`evaluate_models.py` runs all 3 variants in one pass:
- base model (`--base-model`)
- base + LoRA (`--lora-adapter`)
- base + QLoRA (`--qlora-adapter`)

It writes:
- a summary report with `accuracy`, `hallucination_rate`, `a1_score`, plus EM/F1
- optional per-sample predictions with side-by-side outputs

```bash
mkdir -p /llm-finetuning/reports
```

```bash
cd /llm-finetuning
python python/evaluate_models.py \
  --base-model /local-containers/Qwen2-7B-Instruct \
  --lora-adapter /llm-finetuning/adapter-weights/lora \
  --qlora-adapter /llm-finetuning/adapter-weights/qlora \
  --dataset /llm-finetuning/data/seattle_weather_chat_eval.jsonl \
  --output-report /llm-finetuning/reports/base_lora_qlora.json \
  --output-predictions /llm-finetuning/reports/base_lora_qlora_predictions.jsonl \
  --max-samples 200
```

By default evaluation now loads:
- base in 16-bit
- LoRA with a 16-bit base model
- QLoRA with a 4-bit base model

Optional for lower VRAM on the baseline model only:

```bash
python python/evaluate_models.py ... --base-load-mode 4bit
```

## Troubleshooting

- `GPU is unavailable`: confirm container has GPU access (`--gpus all`) and host NVIDIA runtime is installed.
- `kagglehub` auth errors: configure Kaggle credentials before running `data_formatting.py`.
- model not found at `/local-containers/...`: run `python python/pull_model.py`.
- out-of-memory: reduce `--batch-size`, `--mini-batch-size`, or `--max-new-tokens`, or use `--mode qlora`.
