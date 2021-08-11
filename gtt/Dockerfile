FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt update -y

RUN apt install -y build-essential \ 
    cmake \
    git \
    wget \
    python3 \
    python3-pip \ 
    pkg-config \
    nano \
    tmux

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
RUN python -m pip install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

