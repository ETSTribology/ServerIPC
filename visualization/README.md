# Visualization Module

A powerful visualization system for 3D FEM (Finite Element Method) simulations with real-time updates through IPC (Inter-Process Communication).

## Features

- **Multi-Backend Support**
  - Redis (default)
  - gRPC
  - WebSocket
  - MinIO (for storage)

- **Real-Time Visualization**
  - Interactive 3D mesh visualization using Polyscope
  - Dynamic mesh updates
  - Multiple rendering modes (default, wireframe, solid)
  - Customizable color schemes (light/dark)

- **Configuration Management**
  - JSON-based configuration
  - Schema validation
  - Dynamic configuration updates
  - Extensible architecture

## Installation

Install dependencies:
```bash
pip install -r visualization/requirements.txt
```

## Usage

1. Start the visualization client:
```bash
python -m visualization.client --config path/to/config.json
```

Options:
- `--config`: Path to configuration file (default: visualization/config/config.json)
