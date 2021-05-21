FROM nvidia/cuda:11.0-base-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    git \
    vim \
    htop

# Create a working directory
RUN mkdir /aitp
WORKDIR /aitp

RUN sudo apt install python3-pip -y

COPY requirements.txt /aitp
RUN pip install -r /aitp/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
RUN alias python='python3'
