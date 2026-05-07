# syntax=docker/dockerfile:1.7
FROM docker.io/pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

RUN mkdir -p /workspace && chmod -R 777 /workspace
RUN mkdir -p /.cache/pip /.local && chmod -R 777 /.cache/pip /.local

WORKDIR /workspace
