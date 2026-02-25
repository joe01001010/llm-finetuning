* Pull the image:
  - docker pull docker pull nvcr.io/nvidia/pytorch:25.04-py3

* Run the container interactively from scratch:
  - docker run -it --gpus all $(docker images | grep pytorch | awk '{print $3}')
  - or name it:
    - docker run -it --gpus all $(docker images | grep pytorch | awk '{print $3}') --name=llm-finetuning

* Remove all containers:
  - docker rm -f $(docker ps -a | awk '{print $NF}' | grep -v NAMES)
