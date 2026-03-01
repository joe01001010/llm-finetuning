#!/usr/bin/env bash

cd ~/git/llm-finetuning
docker rm -f llm-finetuning
docker run -it -v ~/.cache:/root/.cache -v /tmp/:/local-containers -v $(pwd):/llm-finetuning --name=llm-finetuning --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 43e6802c38a5
