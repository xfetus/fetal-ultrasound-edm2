# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/.venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

COPY pyproject.toml ./
COPY uv.lock* ./
COPY . .

RUN rm -rf /workspace/.venv && \
    /opt/conda/bin/python3.11 -m venv /workspace/.venv && \
    VIRTUAL_ENV=/workspace/.venv \
    /root/.local/bin/uv sync \
      --active \
      --frozen \
      --extra test \
      --extra learning

RUN /workspace/.venv/bin/python -c "import sys; print(sys.executable)" && \
    /workspace/.venv/bin/python -c "import encodings, pandas; print('ok')"