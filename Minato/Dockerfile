#ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG FROM_IMAGE_NAME=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${FROM_IMAGE_NAME}




# Install the rest of the requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --prefer-binary -r /tmp/requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

#RUN apt-get update && apt-get install -y build-essential
# Update pip and install essential build tools
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel

# Install dllogger and mlcommons-logging separately
RUN pip install git+https://github.com/NVIDIA/dllogger
RUN pip install https://github.com/mlcommons/logging/archive/refs/tags/1.1.0-rc4.zip

# Install the rest of the requirements
#COPY requirements.txt /tmp/requirements.txt
#RUN pip install --prefer-binary -r /tmp/requirements.txt

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

#RUN pip install --upgrade pip
#RUN pip install --disable-pip-version-check -r requirements.txt

#RUN pip install torch==2.4.0




