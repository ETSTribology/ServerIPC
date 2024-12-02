# Use a base image with CUDA support
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    libglu1-mesa-dev \
    libx11-dev \
    libxi-dev \
    libxmu-dev \
    libgl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ../requirements.txt /workspace/requirements.txt
COPY . /workspace/simulation

# Set working directory
WORKDIR /workspace

# Python package installation
RUN pip3 install --upgrade pip
RUN pip3 install uv
RUN pip3 install -r requirements.txt

# CUDA configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command
CMD ["python3", "/workspace/simulation/server.py"]
