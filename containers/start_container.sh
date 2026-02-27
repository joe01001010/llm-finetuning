#!/usr/bin/env bash

docker rm -f llm-finetuning
docker run -it -v $(pwd):/llm-finetuning --name=llm-finetuning --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 43e6802c38a5
