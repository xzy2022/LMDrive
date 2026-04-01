FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Miniconda 安装位置
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -f /tmp/miniconda.sh && \
    conda clean -afy

RUN printf "%s\n" \
    "channels:" \
    "  - conda-forge" \
    "channel_priority: strict" \
    "default_channels: []" \
    "allowlist_channels:" \
    "  - conda-forge" \
    > /root/.condarc


RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /root/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]
