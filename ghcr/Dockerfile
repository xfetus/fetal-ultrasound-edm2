# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /workspace && chmod -R 777 /workspace
WORKDIR /workspace

COPY requirements.txt .

RUN /opt/conda/bin/python -m pip install --upgrade pip && \
    /opt/conda/bin/python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/opt/conda/bin/python"]
