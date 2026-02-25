* The below commands are helpful, or just run setup_container.sh as an idempotent start
  - sudo ./setup_container.sh USERNAME

* Pull the image:
  - docker pull docker pull nvcr.io/nvidia/pytorch:25.04-py3

* Remove all containers:
  - docker rm -f $(docker ps -a | awk '{print $NF}'  | grep -v NAMES)

* Run the container interactively from scratch:
  - docker run -it --name=llm-finetuning --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ${containers}

* Start a stopped container and the exec into it:
  - docker start llm-finetuning
  - docker exec -it llm-finetuning /bin/bash
