# Base image with quant essentials
FROM python:3.10-slim-bullseye AS base

# System dependencies for numerical computing
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Project structure setup
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment configuration
ENV PYTHONPATH=/app/src \
    DATA_PATH=/app/data/processed \
    LOG_PATH=/app/logs \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m quant && \
    mkdir -p ${DATA_PATH} ${LOG_PATH} && \
    chown -R quant:quant /app
USER quant

# Development image with research tools
FROM base AS dev
RUN pip install --no-cache-dir \
    jupyterlab==4.0.3 \
    pytest==7.4.0 \
    ipdb==0.13.13 \
    && mkdir -p /app/notebooks

# Production image
FROM base AS prod
COPY --chown=quant:quant . /app