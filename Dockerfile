# ----------------------------------------------------
# Base Image
# ----------------------------------------------------
FROM python:3.11-slim

# ----------------------------------------------------
# Environment Variables
# ----------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# ----------------------------------------------------
# Working Directory
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# Install System Dependencies
# ----------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# Install Python Packages
# ----------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ----------------------------------------------------
# Copy Project
# ----------------------------------------------------
COPY . .

# ----------------------------------------------------
# Create Required Directories
# ----------------------------------------------------
RUN mkdir -p \
    results \
    .cache/huggingface

# ----------------------------------------------------
# Default Command
# ----------------------------------------------------
CMD ["python", "run.py"]