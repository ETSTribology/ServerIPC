# Use a base image with CUDA support
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:${PATH}"
ENV VCPKG_ROOT=/opt/vcpkg
ENV IPC_TOOLKIT_ROOT=/opt/ipc-toolkit

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
    ninja-build \
    pkg-config \
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

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git ${VCPKG_ROOT} \
    && ${VCPKG_ROOT}/bootstrap-vcpkg.sh \

# Clone and build IPC Toolkit
RUN git clone https://github.com/ipc-sim/ipc-toolkit.git ${IPC_TOOLKIT_ROOT} \
    && cd ${IPC_TOOLKIT_ROOT} \
    && cmake -B build -S . \
        -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DIPC_TOOLKIT_WITH_TESTS=OFF \
    && cmake --build build --config Release \
    && cmake --install build \
    && cd .. \
    && pip install . --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Release -DIPC_TOOLKIT_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DIPC_TOOLKIT_WITH_TESTS=OFF" -v


# Clone and build Physics Based Animation
RUN mkdir -p ${PHYSICS_ANIMATION_ROOT} \
&& cd ${PHYSICS_ANIMATION_ROOT} \
&& git clone https://github.com/demirayonur/Physics-Based-Animation.git . \
&& cmake --preset=local-pip-cuda \
&& cmake --build build \
&& cd .. \
&& pip install . --config-settings=cmake.args="--preset=pip-cuda" -v

# CUDA configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set library paths
ENV LD_LIBRARY_PATH="${IPC_TOOLKIT_ROOT}/build/lib:${VCPKG_ROOT}/installed/x64-linux/lib:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="${IPC_TOOLKIT_ROOT}/include:${VCPKG_ROOT}/installed/x64-linux/include:${CPLUS_INCLUDE_PATH}"

# Default command
CMD ["python3", "/workspace/simulation/server.py"]
