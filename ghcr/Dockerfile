# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

COPY pyproject.toml ./
COPY uv.lock* ./
COPY . .

# Make sure a stale local .venv is never copied into the image
RUN rm -rf /workspace/.venv

# Create the env outside /workspace so runtime mounts do not hide it
RUN /opt/conda/bin/python3.11 -m venv /opt/venv && \
    /root/.local/bin/uv sync \
      --active \
      --frozen \
      --extra test \
      --extra learning

RUN /opt/venv/bin/python -c "import sys; print(sys.executable)" && \
    /opt/venv/bin/python -c "import encodings, pandas; print('ok')"

CMD ["/opt/venv/bin/python", "--version"]