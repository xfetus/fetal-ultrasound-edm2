# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/.venv \
    PATH="/root/.local/bin:/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

COPY pyproject.toml ./
COPY uv.lock* ./

RUN /root/.local/bin/uv venv /.venv && \
    /root/.local/bin/uv sync --no-install-project --no-editable

COPY . .

RUN /root/.local/bin/uv sync

CMD ["python"]
