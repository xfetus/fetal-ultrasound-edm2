# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    XDG_CACHE_HOME=/workspace/.cache \
    TORCH_HOME=/workspace/.cache/torch \
    VAE_DIR=/workspace/models/sd-vae-ft-mse

RUN mkdir -p /workspace && chmod -R 777 /workspace
WORKDIR /workspace

RUN mkdir -p \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    /workspace/models \
    "$VAE_DIR"

COPY requirements.txt .

RUN /opt/conda/bin/python -m pip install --upgrade pip && \
    /opt/conda/bin/python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/opt/conda/bin/python"]
