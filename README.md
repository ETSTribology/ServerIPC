# 3D Elastic Simulation with IPC

[![Build Status](https://img.shields.io/github/actions/workflow/status/YourUsername/YourRepo/build.yml?branch=main)](https://github.com/YourUsername/YourRepo/actions)
[![License](https://img.shields.io/github/license/YourUsername/YourRepo)](LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/your-package-name)](https://pypi.org/project/your-package-name/)
[![Documentation Status](https://readthedocs.org/projects/your-project-name/badge/?version=latest)](https://your-project-name.readthedocs.io/en/latest/?badge=latest)

## Overview

This project simulates 3D elastic bodies using linear Finite Element Method (FEM) with tetrahedral elements and Incremental Potential Contact (IPC). It integrates with Redis and MinIO for real-time communication, control, and data storage, enabling interactive simulations and data persistence.

## Features

- **Finite Element Method (FEM)**: Simulates elastic deformations using tetrahedral meshes.
- **Incremental Potential Contact (IPC)**: Handles collision detection and response efficiently.
- **Real-time Communication Network**: Utilizes Redis for real-time communication between the simulation server and visualization client.
- **Modular Design**: Organized into multiple modules for scalability and maintainability.
- **Configurable**: Utilizes Hydra-Core for flexible configuration management.

## Installation

### Prerequisites

- **Python 3.8+**
- **CUDA Toolkit** (if using CUDA acceleration)
- **CMake** (for building native extensions)
- **Docker** (for running the simulation server in a container)



### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install the Package

To install the package with CUDA support and build the necessary C++ extensions, run:

```bash
cd extern/ipc-toolkit
pip install . --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Release -DIPC_TOOLKIT_WITH_CUDA=ON -DIPC_TOOLKIT_BUILD_PYTHON=ON -DCMAKE_CUDA_ARCHITECTURES=native"
```


## Usage

### Running the Simulation Server

Use the `server.py` script to start the simulation server. The server can be configured via command-line arguments or a JSON/YAML configuration file.

### Controlling the Simulation via Redis

The simulation listens to the `simulation_commands` channel in Redis. You can send commands to control the simulation:

- **Start the Simulation**:

  ```bash
  redis-cli publish simulation_commands '{"command": "start"}'
  ```

- **Pause the Simulation**:

  ```bash
  redis-cli publish simulation_commands '{"command": "pause"}'
  ```

- **Resume the Simulation**:

  ```bash
  redis-cli publish simulation_commands '{"command": "resume"}'
  ```

- **Stop the Simulation**:

  ```bash
  redis-cli publish simulation_commands '{"command": "stop"}'
  ```

- **Kill the Simulation**:

  ```bash
  redis-cli publish simulation_commands '{"command": "kill"}'
  ```

### Receiving Simulation Updates

Subscribe to the `simulation_updates` channel in Redis to receive real-time updates:

```bash
redis-cli subscribe simulation_updates
```

### Running the Visualization Client

Use the `client.py` script to run the visualization client, which connects to the simulation server and displays the simulation in real-time.

```bash
python simulation/visualization/client.py \
  --redis-host localhost \
  --redis-port 6379 \
  --redis-db 0
```

### Client Controls

Within the visualization window, you can interact with the simulation:

- **Start**: Start the simulation.
- **Pause**: Pause the simulation.
- **Resume**: Resume the simulation.
- **Stop**: Stop the simulation and reset to the initial state.
- **Reset**: Reset the visualization to the initial mesh state.
- **Navigate Steps**: Use the step slider to navigate through simulation steps.

## Example JSON Configuration

Alternatively, you can use a JSON configuration file. Below is an example:

```json
{
  "simulation": {
    "input": "meshes/input_mesh.msh",
    "percent_fixed": 0.1,
    "mass_density": 1000.0,
    "young_modulus": 6e9,
    "poisson_ratio": 0.45,
    "copies": 2
  },
  "communication": {
    "redis": {
      "host": "localhost",
      "port": 6379,
      "db": 0
    }
  },
  "storage": {
    "minio": {
      "endpoint": "localhost:9000",
      "access_key": "minioadmin",
      "secret_key": "minioadminpassword",
      "bucket": "simulation-data"
    }
  }
}
```

Run the simulation with the JSON config:

```bash
python simulation/server.py --config config.json
```

## Docker Compose Setup

Start the services:

```bash
docker-compose up -d
```

## License

This project is licensed under the [MIT License](LICENSE).

### Additional Improvements

- **Configuration Management**: The project now uses Hydra-Core for flexible configuration management, allowing you to override configurations via command-line or environment variables.
- **Logging**: Enhanced logging configuration for better debugging and monitoring.
- **Error Handling**: Improved error handling for more robust execution.

### References

- **Hydra Documentation**: [https://hydra.cc/docs/intro/](https://hydra.cc/docs/intro/)
- **Redis Documentation**: [https://redis.io/documentation](https://redis.io/documentation)
- **MinIO Documentation**: [https://docs.min.io/](https://docs.min.io/)
- **IPC Toolkit**: [https://ipc-sim.github.io/](https://ipc-sim.github.io/)