# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN mkdir -p /workspace && chmod -R 777 /workspace
WORKDIR /workspace

COPY requirements.txt .

RUN /opt/conda/bin/python -m pip install --upgrade pip && \
    /opt/conda/bin/python -m pip install -r requirements.txt

COPY . .

CMD ["/opt/conda/bin/python"]
