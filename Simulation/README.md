# 3D Elastic Simulation with IPC

## Overview

This project simulates 3D elastic bodies using linear FEM tetrahedra and Incremental Potential Contact (IPC). It leverages Redis for communication and control, enabling real-time interaction with the simulation.

## Features

- **Finite Element Method (FEM)**: Simulates elastic deformations using tetrahedral meshes.
- **Incremental Potential Contact (IPC)**: Handles collision detection and response.
- **Redis Integration**: Allows controlling the simulation (start, pause, stop, etc.) and streaming simulation data.
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

3. Install Dependencies

    ```bash
    pip install -r requirements.txt
    ```


## Available Materials
- WOOD: Young's Modulus = 10e9 Pa, Poisson's Ratio = 0.35, Density = 600 kg/m³
- STEEL: Young's Modulus = 210e9 Pa, Poisson's Ratio = 0.3, Density = 7850 kg/m³
- ALUMINUM: Young's Modulus = 69e9 Pa, Poisson's Ratio = 0.33, Density = 2700 kg/m³
- CONCRETE: Young's Modulus = 30e9 Pa, Poisson's Ratio = 0.2, Density = 2400 kg/m³
- RUBBER: Young's Modulus = 0.01e9 Pa, Poisson's Ratio = 0.48, Density = 1100 kg/m³
- COPPER: Young's Modulus = 110e9 Pa, Poisson's Ratio = 0.34, Density = 8960 kg/m³
- GLASS: Young's Modulus = 50e9 Pa, Poisson's Ratio = 0.22, Density = 2500 kg/m³
- TITANIUM: Young's Modulus = 116e9 Pa, Poisson's Ratio = 0.32, Density = 4500 kg/m³
- BRASS: Young's Modulus = 100e9 Pa, Poisson's Ratio = 0.34, Density = 8500 kg/m³
- PLA: Young's Modulus = 4.4e9 Pa, Poisson's Ratio = 0.3, Density = 1250 kg/m³
- ABS: Young's Modulus = 2.3e9 Pa, Poisson's Ratio = 0.35, Density = 1050 kg/m³
- PETG: Young's Modulus = 2.59e9 Pa, Poisson's Ratio = 0.4, Density = 1300 kg/m³
- HYDROGEL: Young's Modulus = 1e6 Pa, Poisson's Ratio = 0.35, Density = 1000 kg/m³
- POLYACRYLAMIDE: Young's Modulus Young's Modulus = 1e6 Pa, Poisson's Ratio = 0.45, Density = - 1050 kg/m³

## Usage
Run the simulation using the main.py script with appropriate arguments.

    ```bash
    python simulation/server.py -i meshes/input_mesh.msh --percent-fixed 0.1 -m 1000.0 -Y 6e9 -n 0.45 -c 2 --redis-host localhost --redis-port 6379 --redis-db 0
    ```
###Arguments
    -i, --input: Path to the input mesh file.
    --percent-fixed: Percentage of the input mesh's bottom to fix (default: 0.1).
    -m, --mass-density: Mass density (default: 1000.0).
    -Y, --young-modulus: Young's modulus (default: 6e9).
    -n, --poisson-ratio: Poisson's ratio (default: 0.45).
    -c, --copy: Number of copies of the input model (default: 1).
    --redis-host: Redis host address (default: localhost).
    --redis-port: Redis port (default: 6379).
    --redis-db: Redis database number (default: 0).

### Redis Commands
The simulation listens to the simulation_commands channel in Redis. You can send commands like start, pause, resume, stop, play, and kill to control the simulation.


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
#### Simulation Updates

    The simulation publishes mesh updates to the simulation_updates channel in Redis. You can subscribe to this channel to receive real-time updates.

    ```bash
    redis-cli subscribe simulation_updates

    ```


### Client Application Controls
    Within the Polyscope visualization window, you can interact with the simulation using the UI:

    Pause: Pauses the simulation.
    Play: Resumes the simulation.
    Stop: Stops the simulation and resets the mesh to the initial state.
    Start: Starts the simulation.
    Kill Simulation: Terminates the simulation server.
    Reset to Initial State: Resets the visualization to the initial mesh state.
    Select Step: Navigate through different simulation steps.


    ```bash
    python simulation/visualization/client.py --redis-host localhost --redis-port 6379 --redis-db 0
    ```