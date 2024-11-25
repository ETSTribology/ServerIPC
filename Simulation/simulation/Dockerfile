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
RUN wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | \
    gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/kitware.list > /dev/null

# Update package lists and install CMake
RUN apt-get update
RUN apt-get install -y cmake

# Install Python dependencies
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install uv
RUN pip3 install -r requirements.txt

RUN wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
RUN chmod +x /tmp/llvm.sh
RUN /tmp/llvm.sh 18 || true

RUN apt-get update && apt-get install -y clang-18 clang-tools-18 clang-format-18

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
RUN echo '{
    "version": 6,
    "configurePresets": [
        {
            "name": "local-all",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_CUDA_ARCHITECTURES": "native",
                "IPC_TOOLKIT_WITH_CUDA": {
                    "type": "BOOL",
                    "value": "ON"
                },
                "IPC_TOOLKIT_BUILD_PYTHON": {
                    "type": "BOOL",
                    "value": "ON"
                },
                "CMAKE_POSITION_INDEPENDENT_CODE": {
                    "type": "BOOL",
                    "value": "ON"
                },
                "CMAKE_CXX_COMPILER": {
                    "type": "FILEPATH",
                    "value": "/usr/bin/clang-18"
                },
                "CMAKE_CXX_FLAGS": {
                    "type": "STRING",
                    "value": "-Werror -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wuseless-cast -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough -Wno-inline"
                }
            }
        }
    ]
}' > PhysicsBasedAnimationToolkit/CMakeUserPresets.json
RUN cd PhysicsBasedAnimationToolkit && pip install . --config-settings=cmake.args="--preset=local-all" -v && cd ..


# Set CUDA visible devices
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default workdir
WORKDIR /workspace
