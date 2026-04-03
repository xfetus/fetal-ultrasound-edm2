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
COPY . .

# Install into the same interpreter used at runtime
RUN /root/.local/bin/uv pip install --system -r pyproject.toml \
    --python /opt/conda/bin/python3.11 \
    --extra test \
    --extra learning

# Install the project itself too, if train_edm2.py imports local package metadata/entry points
RUN /opt/conda/bin/python3.11 -m pip install -e .

# Hard checks
RUN /opt/conda/bin/python3.11 -c "import sys; print(sys.executable)" && \
    /opt/conda/bin/python3.11 -c "import pandas; print(pandas.__version__)"