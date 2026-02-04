#!/usr/bin/env bash

if [[ $UID -ne 0 ]]
then
  echo "Script must be executed as root"
  exit 1
fi

image=$(docker images | grep pytorch)

if [[ $? -ne 0 ]]
then
  docker pull pytorch/pytorch:latest
fi


docker rm -f $(sudo docker ps -a | grep -v NAMES | awk '{print $NF}')

docker run -d --name pytorch_test -v /home/joeweibel-laptop/git/llm-finetuning:/mnt/ pytorch/pytorch:latest sh
