# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /workspace && chmod -R 777 /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN /opt/conda/bin/python -m pip install --upgrade pip && \
    /opt/conda/bin/python -m pip install --no-cache-dir -r requirements.txt && \
    /opt/conda/bin/python -m pip uninstall -y \
      opencv-python \
      opencv-contrib-python \
      opencv-python-headless \
      opencv-contrib-python-headless || true && \
    /opt/conda/bin/python -m pip install --no-cache-dir opencv-python-headless==4.12.0.88

RUN dpkg -s libxcb1 && \
    ls -l /usr/lib/x86_64-linux-gnu/libxcb.so.1 && \
    /opt/conda/bin/python -c "import cv2; print('cv2 ok:', cv2.__version__)"

COPY . .

CMD ["/opt/conda/bin/python"]
