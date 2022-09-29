#FROM chainer/chainer

FROM nvidia/cuda:9.2-cudnn7-devel

WORKDIR /home/[username]
COPY ./Miniconda3-py38_4.8.2-Linux-x86_64.sh ./
RUN chmod o+x Miniconda3-py38_4.8.2-Linux-x86_64.sh
RUN ./Miniconda3-py38_4.8.2-Linux-x86_64.sh -b -p /home/[username]/Miniconda -f

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
#    python3-pip \
#    python3-wheel \
#    python3-setuptools \
    git \
    cmake \
    libblas3 \
    libblas-dev \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 Miniconda/bin/pip install --no-cache-dir cupy-cuda92==7.2.0 chainer==7.2.0

COPY ./pip.txt ./
RUN Miniconda/bin/pip install -r ./pip.txt
