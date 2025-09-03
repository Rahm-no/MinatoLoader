ARG FROM_IMAGE_NAME=nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu20.04
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d
RUN chmod +x /workspace/unet3d/run_and_time.sh

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio

# Install NVIDIA DALI
RUN pip3 install nvidia-dali-cuda120

# Install other requirements
RUN pip3 install --disable-pip-version-check -r requirements.txt
