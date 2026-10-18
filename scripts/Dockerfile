# CIと同じ環境 (Debian bookworm + lavapipe) でテストを動かす。
# build: docker build -f scripts/Dockerfile -t mjmlx-ci .
# run:   docker run --rm -v "$PWD":/work mjmlx-ci
FROM debian:bookworm
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates pkg-config \
        python3 python3-pip python3-venv \
        vulkan-tools mesa-vulkan-drivers libvulkan-dev glslang-tools \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install mujoco==3.5.0 cmake
ENV PATH=/opt/venv/bin:$PATH
ENV MUJOCO_ROOT=/opt/venv/lib/python3.11/site-packages/mujoco
WORKDIR /work
# 別buildディレクトリを使い、ホストのbuild/と衝突させない
CMD ["bash", "-c", "git config --global --add safe.directory '*' && BUILD_DIR=build-docker just check-ci"]
