# ServerIPC Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows (WSL2)
- **Python**: 3.9 - 3.11
- **RAM**: 16 GB recommended
- **Disk Space**: 10 GB

### Software Dependencies
- Python 3.9+
- Git
- Docker (optional)
- CUDA Toolkit (optional, for GPU acceleration)

## Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/ETSTribology/ServerIPC.git
cd ServerIPC
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv serverIPC_env
source serverIPC_env/bin/activate  # On Windows: serverIPC_env\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Optional: Install with development tools
poetry install --extras dev
```

### 4. Optional: CUDA Setup
```bash
# Install with CUDA support
poetry run pip install . --config-settings=cmake.args="-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda"
```

## Verification
```bash
# Check ServerIPC installation
poetry run python -c "import simulation; print(simulation.__version__)"
```

## Troubleshooting

### Common Issues
1. **Dependency Conflicts**
   ```bash
   poetry update
   poetry cache clear pypi --all
   ```

2. **Python Version**
   - Ensure Python 3.9+ is installed
   - Check `python3 --version`

## Environment Management

### Activate Environment
```bash
# Using Poetry
poetry shell

# Manually
source serverIPC_env/bin/activate
```

### Deactivate Environment
```bash
# Exit Poetry shell
exit

# Manually deactivate
deactivate
```

## Additional Resources
- [Poetry Documentation](https://python-poetry.org/docs/)
- [ServerIPC GitHub Repository](https://github.com/ETSTribology/ServerIPC)

## Support
For installation issues:
1. Check [GitHub Issues](https://github.com/ETSTribology/ServerIPC/issues)
2. Consult [Documentation](https://serverIPC.readthedocs.io/)
3. Open a new issue with detailed error logs

---

**Happy Simulating!** 🚀🔬
