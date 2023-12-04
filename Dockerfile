#!/bin/bash
FROM python:3.9.13

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install g++
RUN apt-get install gcc
RUN apt-get install -y gfortran
RUN apt-get install -y libmariadb-dev
RUN apt-get install -y pkg-config
RUN apt-get install -y make
RUN apt-get install -y cmake
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y python3-dev
RUN apt-get install -y musl-dev
RUN apt-get install -y build-essential
RUN apt-get install -y llvm
RUN apt-get install -y llvm-dev
RUN apt-get install -y libffi-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y openjdk-11-jdk

ENV LLVM_CONFIG=/usr/bin/llvm-config
RUN pip install llvmlite==0.41.0

RUN pip install --upgrade pip
COPY ./docker_requirements.txt .
RUN pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r ./docker_requirements.txt
RUN pip install scikit-learn==1.3.1
RUN pip install umap-learn==0.5.4
RUN pip install sentence-transformers==2.2.2
RUN pip install hdbscan==0.8.33
RUN pip install pynndescent==0.5.10

RUN mkdir -p src/main
COPY ./src/main ./src/main
WORKDIR /src/main

EXPOSE 8081
ENTRYPOINT ["python", "main.py"]