FROM nvcr.io/nvidia/cuda:11.4.0-runtime-ubuntu20.04

ARG version=${TZ}
RUN ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime

RUN apt-get update
RUN apt-get install -y build-essential libssl-dev libffi-dev libbz2-dev liblzma-dev libgl1-mesa-dev libncurses5-dev zlib1g zlib1g-dev libreadline-dev libsqlite3-dev libopencv-dev curl git
RUN apt-get clean

RUN curl --output Python-3.10.6.tgz https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tgz
RUN tar xvf Python-3.10.6.tgz
RUN cd Python-3.10.6 && ./configure --enable-optimizations && make altinstall
RUN rm -rf Python-3.10.6 Python-3.10.6.tgz

RUN git clone https://github.com/Stability-AI/stablediffusion.git app
WORKDIR /app

RUN pip3.10 install transformers==4.19.2 diffusers invisible-watermark
RUN pip3.10 install -r requirements.txt
RUN pip3.10 install -e .
RUN pip3.10 install xformers
