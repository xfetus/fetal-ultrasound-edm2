# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

COPY pyproject.toml ./
COPY uv.lock* ./

RUN /root/.local/bin/uv pip install \
    --system \
    --python /opt/conda/bin/python3.11 \
    -r pyproject.toml \
    --extra test \
    --extra learning

COPY . .

RUN /opt/conda/bin/python3.11 -c "import sys; print(sys.executable)" && \
    /opt/conda/bin/python3.11 -c "import pandas as pd; print(pd.__version__)"
