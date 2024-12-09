# Use a base image with CUDA support
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:${PATH}"
ENV VCPKG_ROOT=/opt/vcpkg
ENV IPC_TOOLKIT_ROOT=/opt/ipc-toolkit
# CUDA configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Update package list
RUN apt-get update -y

# Install essential tools
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y curl
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-dev

# Install graphical dependencies
RUN apt-get install -y libglu1-mesa-dev
RUN apt-get install -y libx11-dev
RUN apt-get install -y libxi-dev
RUN apt-get install -y libxmu-dev
RUN apt-get install -y libgl-dev

# Install other essential libraries
RUN apt-get install -y ninja-build
RUN apt-get install -y pkg-config

# Additional dependencies
RUN apt-get install -y build-essential zip unzip wget curl tar
RUN apt-get install -y libxinerama-dev
RUN apt-get install -y libxrandr-dev
RUN apt-get install -y libxcursor-dev
RUN apt-get install -y liblapack-dev
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y libsuitesparse-dev
RUN apt-get install -y librange-v3-dev
RUN apt-get install -y libfmt-dev
RUN apt-get install -y doctest-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y bison
RUN apt-get install -y ccache
RUN apt-get install -y linux-libc-dev
RUN apt-get install -y gfortran
RUN apt-get install -y xorg-dev
RUN apt-get install -y freeglut3-dev
RUN apt-get install -y mesa-common-dev
RUN apt-get install -y libgmp3-dev

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

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
RUN git clone https://github.com/Microsoft/vcpkg.git ${VCPKG_ROOT}
RUN cd ${VCPKG_ROOT} && git pull && ${VCPKG_ROOT}/bootstrap-vcpkg.sh

# Clone and build IPC Toolkit
RUN git clone https://github.com/ipc-sim/ipc-toolkit.git ${IPC_TOOLKIT_ROOT} \
    && cd ${IPC_TOOLKIT_ROOT} \
    && pip install . --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Release -DIPC_TOOLKIT_WITH_CUDA=OFF -DIPC_TOOLKIT_WITH_TESTS=OFF" -v

# Clone and build Physics Based Animation
RUN mkdir -p ${PHYSICS_ANIMATION_ROOT} \
    && cd ${PHYSICS_ANIMATION_ROOT} \
    && git clone https://github.com/demirayonur/Physics-Based-Animation.git . \
    && cmake --preset=local-pip-cuda \
    && cmake --build build \
    && cd .. \
    && pip install . --config-settings=cmake.args="CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON CMAKE_TOOLCHAIN_FILE:FILEPATH=/home/antoine/ServerIPC/extern/vcpkg/scripts/buildsystems/vcpkg.cmake PBAT_BUILD_PYTHON_BINDINGS:BOOL=ON PBAT_USE_CUDA:BOOL=ON PBAT_USE_SUITESPARSE:BOOL=ON VCPKG_MANIFEST_FEATURES=python;cuda;suitesparse"

# Default command
CMD ["python3", "/workspace/simulation/server.py"]
