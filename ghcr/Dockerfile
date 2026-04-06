# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /workspace && chmod -R 777 /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN /opt/conda/bin/python -m pip install --upgrade pip && \
    /opt/conda/bin/python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/opt/conda/bin/python"]
