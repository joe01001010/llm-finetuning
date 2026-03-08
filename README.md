# llm-finetuning

End-to-end local workflow for:
- pulling a base LLM into a container-mounted path
- preparing a weather prediction chat dataset
- training adapters with `LoRA` and `QLoRA`
- evaluating base model vs adapter model outputs

## Project Capabilities

- Containerized GPU workflow using `nvcr.io/nvidia/pytorch:25.04-py3`
- Base model pull from Hugging Face (`Qwen/Qwen2-7B-Instruct`)
- Dataset pull + formatting from Kaggle into chat fine-tuning JSONL
- Adapter training with:
  - `LoRA` (full-precision base model loading)
  - `QLoRA` (4-bit quantized base model loading)
- Side-by-side evaluation of base vs base+adapter with latency and optional EM/F1

## Repository Layout

- `containers/setup_container.sh`: installs Docker + NVIDIA container toolkit and pulls the base image.
- `python/pull_model.py`: downloads the base model to `/local-containers/Qwen2-7B-Instruct`.
- `python/data_formatting.py`: pulls weather dataset from Kaggle and writes `data/seattle_weather_chat.json`.
- `python/train_model.py`: trains adapter weights (`--mode lora` or `--mode qlora`).
- `python/evaluate_models.py`: compares base model vs one adapter.
- `lora-weights/`: default adapter output location.

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
- `/llm-finetuning/data/seattle_weather_chat.json`

## 6) Train Adapters

`train_model.py` supports both modes:
- `qlora`: 4-bit quantized base model loading (`BitsAndBytesConfig`)
- `lora`: non-quantized base model loading

### Train QLoRA

```bash
cd /llm-finetuning
python python/train_model.py \
  --mode qlora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --data-path /llm-finetuning/data/seattle_weather_chat.json \
  --output-dir /llm-finetuning/lora-weights/qlora \
  --num-train-epochs 1
```

### Train LoRA

```bash
cd /llm-finetuning
python python/train_model.py \
  --mode lora \
  --model-path /local-containers/Qwen2-7B-Instruct \
  --data-path /llm-finetuning/data/seattle_weather_chat.json \
  --output-dir /llm-finetuning/lora-weights/lora \
  --num-train-epochs 1
```

Common tunables:
- `--per-device-train-batch-size`
- `--gradient-accumulation-steps`
- `--learning-rate`
- `--max-length`

## 7) Evaluate Base vs Adapter

`evaluate_models.py` compares:
- base model (`--base-model`)
- base + one adapter (`--lora-adapter`)

It writes a summary report and optional per-sample predictions.

```bash
mkdir -p /llm-finetuning/reports
```

### Base vs QLoRA

```bash
cd /llm-finetuning
python python/evaluate_models.py \
  --base-model /local-containers/Qwen2-7B-Instruct \
  --lora-adapter /llm-finetuning/lora-weights/qlora \
  --dataset /llm-finetuning/data/seattle_weather_chat.json \
  --output-report /llm-finetuning/reports/base_vs_qlora.json \
  --output-predictions /llm-finetuning/reports/base_vs_qlora_predictions.jsonl \
  --max-samples 200
```

### Base vs LoRA

```bash
cd /llm-finetuning
python python/evaluate_models.py \
  --base-model /local-containers/Qwen2-7B-Instruct \
  --lora-adapter /llm-finetuning/lora-weights/lora \
  --dataset /llm-finetuning/data/seattle_weather_chat.json \
  --output-report /llm-finetuning/reports/base_vs_lora.json \
  --output-predictions /llm-finetuning/reports/base_vs_lora_predictions.jsonl \
  --max-samples 200
```

## 8) Compare LoRA vs QLoRA Reports

After both evaluations complete:

```bash
python - <<'PY'
import json
from pathlib import Path

lora = json.loads(Path("/llm-finetuning/reports/base_vs_lora.json").read_text())
qlora = json.loads(Path("/llm-finetuning/reports/base_vs_qlora.json").read_text())

print("LoRA summary:", json.dumps(lora["base_plus_lora"], indent=2))
print("QLoRA summary:", json.dumps(qlora["base_plus_lora"], indent=2))
print("Base summary (LoRA run):", json.dumps(lora["base_model"], indent=2))
PY
```

## Troubleshooting

- `GPU is unavailable`: confirm container has GPU access (`--gpus all`) and host NVIDIA runtime is installed.
- `kagglehub` auth errors: configure Kaggle credentials before running `data_formatting.py`.
- model not found at `/local-containers/...`: run `python python/pull_model.py`.
- out-of-memory: reduce batch size, increase gradient accumulation, reduce `--max-length`, or use `--mode qlora`.
