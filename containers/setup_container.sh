#!/usr/bin/env bash
#
# This is designed for ubuntu using the apt package manager


DOCKER_IMAGE_NAME=nvcr.io/nvidia/pytorch
DOCKER_IMAGE_VERSION=25.04-py3


if [[ -z $1 ]]
then
  echo "Must send username as an argument"
  echo "Example: ${0} \$USER"
  exit 67
fi

if [[ ${UID} -ne 0 ]]
then
  echo "Script must be run as root"
  exit 67
fi

apt update
apt upgrade -y
apt install -y curl ca-certificates gnupg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

apt update
apt install -y nvidia-container-toolkit docker.io
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker 
systemctl enable docker
getent group docker

if [[ $? -ne 0 ]]
then
  echo "Docker group doesnt exist"
  echo "Something went wrong with the docker install"
else
  usermod -aG docker ${1}
  id -nG ${1} | grep docker
  group_status=$?
fi

containers=$(docker images | grep ${DOCKER_IMAGE_NAME} | grep ${DOCKER_IMAGE_VERSION} | awk '{print $3}')
if [[ $? -ne 0 ]]
then
  docker pull ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION}
fi

echo ""
echo "====================================="
echo "====================================="
if [[ $group_status -ne 0 ]]
then
  echo "You will need to completely log out of your session and log back in"
  echo "Your groups didnt update to show you being in the docker group"
  exit 67
else
  echo "Your container image is pulled"
  echo "Ensure existing containers of the same name are removed:"
  echo "docker rm -f $(docker ps -a | grep llm-finetuning)"
  echo "Run your container with the following command:"
  echo "docker run -it --name=llm-finetuning --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ${containers}"
  exit 0
fi
