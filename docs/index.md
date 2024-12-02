# 3D Elastic Simulation with IPC

[![Build Status](https://img.shields.io/github/actions/workflow/status/ETSTribology/ServerIPC/build.yml?branch=main)](https://github.com/ETSTribology/ServerIPC/actions)
[![License](https://img.shields.io/github/license/ETSTribology/ServerIPC)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/serverIPC/badge/?version=latest)](https://serverIPC.readthedocs.io/en/latest/?badge=latest)

## Project Overview

ServerIPC is an advanced 3D elastic simulation framework designed for precise mechanical behavior modeling. Leveraging Finite Element Method (FEM) and Incremental Potential Contact (IPC), this project enables high-fidelity simulation of complex material interactions.

### Scientific Context

This simulation framework is specifically developed for tribological research, focusing on calculating friction coefficients between different material interfaces. It provides a computational platform for understanding mechanical interactions at microscopic scales.

## Key Features

### Computational Mechanics
- **Finite Element Method (FEM)**: Advanced tetrahedral mesh-based elastic deformation simulation
- **Incremental Potential Contact (IPC)**: Robust collision detection and response mechanism
- **Multi-material Support**: Comprehensive material property database
- **High-Performance Computing**: CUDA acceleration for GPU-enabled simulations

### Networking and Data Management
- **Real-time Communication**: Redis-powered communication infrastructure
- **Distributed Storage**: MinIO integration for simulation data persistence
- **Flexible Configuration**: Hydra-Core based dynamic configuration management

### Technical Capabilities
- Simulate complex material interactions
- Calculate friction coefficients
- Model non-linear material behaviors
- Support for various material types (metals, polymers, composites)

## Installation

### Prerequisites
- Python 3.8+
- CUDA Toolkit (optional, for GPU acceleration)
- CMake
- Docker (recommended)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/ETSTribology/ServerIPC.git
cd ServerIPC

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install with CUDA support
cd extern/ipc-toolkit
pip install . --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Release -DIPC_TOOLKIT_WITH_CUDA=ON"
```

## Supported Materials

The simulation supports a wide range of material properties, including:
- Metals: Steel, Aluminum, Copper, Titanium
- Polymers: PLA, ABS, PETG
- Composites: Concrete, Glass
- Elastomers: Rubber

## Usage Example

```python
from simulation import SimulationManager

# Initialize simulation
sim_manager = SimulationManager()
sim_manager.configure(
    material_a='steel',
    material_b='rubber',
    contact_parameters={
        'friction_model': 'coulomb',
        'normal_stiffness': 1e5
    }
)

# Run simulation
results = sim_manager.run()
print(f"Friction Coefficient: {results.friction_coefficient}")
```

## Documentation

Comprehensive documentation is available at [ServerIPC Documentation](https://serverIPC.readthedocs.io/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Tribology Research Group, École de Technologie Supérieure
- IPC-Toolkit Contributors
- NVIDIA CUDA Platform
