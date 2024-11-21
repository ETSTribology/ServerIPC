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

# Install the latest CMake
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.sh -O cmake.sh && \
    chmod +x cmake.sh && ./cmake.sh --skip-license --prefix=/usr/local && rm cmake.sh

# Install Python dependencies
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install uv
RUN pip3 install -r requirements.txt

# Clone ipc-toolkit
RUN git clone --recursive https://github.com/ipc-sim/ipc-toolkit.git
RUN cd ipc-toolkit && pip3 install . --config-settings=cmake.args="-DCMAKE_BUILD_TYPE='Release' -DIPC_TOOLKIT_WITH_CUDA='ON' -DIPC_TOOLKIT_BUILD_PYTHON='ON' -DCMAKE_CUDA_ARCHITECTURES='native'" -v && cd ..

# Install VCPKG
RUN git clone https://github.com/microsoft/vcpkg.git
RUN cd vcpkg && ./bootstrap-vcpkg.sh && cd ..
# add vcpkg to PATH
ENV PATH="${PATH}:$(pwd)/vcpkg"
ENV VCPKG_ROOT="$(pwd)/vcpkg"

# Clone PhysicsBasedAnimationToolkit
RUN git clone --recursive https://github.com/Q-Minh/PhysicsBasedAnimationToolkit.git
RUN cd PhysicsBasedAnimationToolkit && pip3 install . --config-settings=cmake.args="-DCMAKE_TOOLCHAIN_FILE='$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake' -DCMAKE_CUDA_ARCHITECTURES='native' -DPBAT_BUILD_PYTHON_BINDINGS='ON' -DPBAT_USE_CUDA='ON' -DCMAKE_POSITION_INDEPENDENT_CODE='ON' -DPBAT_USE_SUITESPARSE='ON' -DCMAKE_CUDA_ARCHITECTURES='native' -DVCPKG_MANIFEST_FEATURES='cuda;python;suitesparse;'" -v && cd ..

# Set CUDA visible devices
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default workdir
WORKDIR /workspace
