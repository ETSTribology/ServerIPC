# 3D Elastic Simulation with IPC

## Overview

This project simulates 3D elastic bodies using linear FEM tetrahedra and Incremental Potential Contact (IPC). It leverages Redis for communication and control, enabling real-time interaction with the simulation.

## Features

- **Finite Element Method (FEM)**: Simulates elastic deformations using tetrahedral meshes.
- **Incremental Potential Contact (IPC)**: Handles collision detection and response.
- **Redis Integration**: Allows controlling the simulation (start, pause, stop, etc.) and streaming simulation data.
- **Minio Integration**: Allow to storage the screenshots and the meshes in a buckets
- **Modular Design**: Organized into multiple modules for scalability and maintainability.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ETSTribology/ServerIPC.git
   cd elastic-simulation-ipc
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Available Materials

| Material           | Young's Modulus (Pa) | Poisson's Ratio | Density (kg/m³) |
|--------------------|----------------------|-----------------|-----------------|
| **Wood**           | 1.0e10               | 0.35            | 600             |
| **Steel**          | 2.1e11               | 0.30            | 7850            |
| **Aluminum**       | 6.9e10               | 0.33            | 2700            |
| **Concrete**       | 3.0e10               | 0.20            | 2400            |
| **Rubber**         | 1.0e7                | 0.48            | 1100            |
| **Copper**         | 1.1e11               | 0.34            | 8960            |
| **Glass**          | 5.0e10               | 0.22            | 2500            |
| **Titanium**       | 1.16e11              | 0.32            | 4500            |
| **Brass**          | 1.0e11               | 0.34            | 8500            |
| **PLA**            | 4.4e9                | 0.30            | 1250            |
| **ABS**            | 2.3e9                | 0.35            | 1050            |
| **PETG**           | 2.59e9               | 0.40            | 1300            |
| **Hydrogel**       | 1.0e6                | 0.35            | 1000            |
| **Polyacrylamide** | 1.0e6                | 0.45            | 1050            |

## Usage

Run the simulation using the main.py script with appropriate arguments.

```bash
python simulation/server.py -i meshes/input_mesh.msh --percent-fixed 0.1 -m 1000.0 -Y 6e9 -n 0.45 -c 2 --redis-host localhost --redis-port 6379 --redis-db 0
```

### Arguments

- `-i`, `--input`: Path to the input mesh file.
- `--percent-fixed`: Percentage of the input mesh's bottom to fix (default: 0.1).
- `-m`, `--mass-density`: Mass density (default: 1000.0).
- `-Y`, `--young-modulus`: Young's modulus (default: 6e9).
- `-n`, `--poisson-ratio`: Poisson's ratio (default: 0.45).
- `-c`, `--copy`: Number of copies of the input model (default: 1).
- `--redis-host`: Redis host address (default: localhost).
- `--redis-port`: Redis port (default: 6379).
- `--redis-db`: Redis database number (default: 0).

### Redis Commands

The simulation listens to the `simulation_commands` channel in Redis. You can send commands like `start`, `pause`, `resume`, `stop`, `play`, and `kill` to control the simulation.

```bash
# Start the simulation
redis-cli publish simulation_commands start

# Pause the simulation
redis-cli publish simulation_commands pause

# Resume the simulation
redis-cli publish simulation_commands resume

# Stop the simulation
redis-cli publish simulation_commands stop

# Kill the simulation
redis-cli publish simulation_commands kill
```

### Simulation Updates

The simulation publishes mesh updates to the `simulation_updates` channel in Redis. You can subscribe to this channel to receive real-time updates.

```bash
redis-cli subscribe simulation_updates
```

## Client Application Controls

Within the Polyscope visualization window, you can interact with the simulation using the UI:

- **Pause**: Pauses the simulation.
- **Play**: Resumes the simulation.
- **Stop**: Stops the simulation and resets the mesh to the initial state.
- **Start**: Starts the simulation.
- **Kill Simulation**: Terminates the simulation server.
- **Reset to Initial State**: Resets the visualization to the initial mesh state.
- **Select Step**: Navigate through different simulation steps.

```bash
python simulation/visualization/client.py --redis-host localhost --redis-port 6379 --redis-db 0
```

## Example JSON Configuration

The simulation can also be controlled through a JSON configuration file. Below is an example configuration:

```json
{
  "name": "Sample Simulation Configuration",
  "inputs": [
    {
      "path": "meshes/input_mesh.msh",
      "percent_fixed": 0.1,
      "material": {
        "density": 1000.0,
        "young_modulus": 6e9,
        "poisson_ratio": 0.45,
        "color": [255, 255, 255, 1]
      },
      "transform": {
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "translation": [0.0, 0.0, 0.0]
      },
      "force": {
        "gravity": 9.81,
        "top_force": 10,
        "side_force": 0
      }
    }
  ],
  "friction": {
    "friction_coefficient": 0.3,
    "damping_coefficient": 1e-4
  },
  "simulation": {
    "dhat": 1e-3,
    "dmin": 1e-4,
    "dt": 0.016
  },
  "server": {
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 0
  },
  "initial_conditions": {
    "gravity": 9.81
  }
}
```

### How to Use the JSON Config

To run the simulation using the JSON config file, you can provide the path to the file using the `--json` argument:

```bash
python simulation/server.py --json config.json
```

This configuration file defines the mesh input, material properties, transformation settings, external forces, friction parameters, and server details.


pip install . --config-settings=cmake.args="-DCMAKE_BUILD_TYPE='Release' -DIPC_TOOLKIT_WITH_CUDA='ON' -DIPC_TOOLKIT_BUILD_PYTHON='ON' -DCMAKE_CUDA_ARCHITECTURES='native'"