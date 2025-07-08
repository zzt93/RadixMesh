# 使用官方最小化的 Ubuntu 24.04 镜像
FROM ubuntu:24.04

# 避免交互
ENV DEBIAN_FRONTEND=noninteractive

# 更新并安装 Python3 以及常用依赖（pip, venv），并清理缓存
RUN apt-get update && \
    apt-get install -y python3 python3.12-venv python3-pip && \
    apt-get install -y build-essential \
                   cmake \
                   libibverbs-dev \
                   libgoogle-glog-dev \
                   libgtest-dev \
                   libjsoncpp-dev \
                   libnuma-dev \
                   libunwind-dev \
                   libpython3-dev \
                   libboost-all-dev \
                   libssl-dev \
                   pybind11-dev \
                   libcurl4-openssl-dev \
                   libhiredis-dev \
                   pkg-config \
                   patchelf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建工作目录并设为默认工作目录
WORKDIR /app

# 将 requirements.txt 复制到容器中
COPY python/requirements.txt .

# 安装 requirements.txt 依赖
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

# 默认进入 bash
CMD ["/bin/bash"]