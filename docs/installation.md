# ServerIPC Installation Guide

This guide is split into **Server Setup** and **Client Setup** for simplicity.

---

## Server Setup

### 1. Prerequisites

Ensure you have the following installed:

- **Python**: Version 3.9 or higher
- **Git**: For cloning the repository
- **Docker**: For running external services

### 2. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/ETSTribology/ServerIPC.git
cd ServerIPC
```

### 3. Install Poetry

Poetry manages the dependencies for the project. Install it with:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

Verify installation:

```bash
poetry --version
```

### 4. Install Dependencies

Run the following to install all server dependencies:

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

# ServerIPC Docker Setup

This guide explains the Docker-based setup for the **ServerIPC** project, including a table of services and a Mermaid diagram for visualization.

---

## Docker Services

The project uses the following services, which can be configured via `docker-compose.yml`:

| Service      | Description                                 | Ports         | Environment Variables                        | Volumes        |
|--------------|---------------------------------------------|---------------|----------------------------------------------|----------------|
| **Dragonfly** | Fast in-memory key-value store (Redis)      | `6379:6379`   | N/A                                          | `dragonfly-data:/data` |
| **MinIO**    | Object storage system                       | `9000:9000`, `9001:9001` | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`     | `minio_data:/data`, `minio_config:/root/.minio` |
| **SurrealDB**| Advanced database for the simulation server | `8000:8000`   | `SURREALDB_USER`, `SURREALDB_PASSWORD`       | `surrealdb_data:/data` |

---

## Mermaid Diagram

```mermaid
graph TD
    A[Client Application] -->|Commands/Updates| B[Dragonfly (Redis)]
    B -->|Pub/Sub Channel| C[ServerIPC Simulation]
    C -->|Stores Mesh Data| D[MinIO (Object Storage)]
    C -->|Stores Simulation Data| E[SurrealDB (Database)]
```

---

## Steps to Deploy Docker Services

### 2. Start Docker Services

Run the following command in your terminal to start all services:

```bash
docker-compose up -d
```

### 3. Check the Status

Verify that all services are running:

```bash
docker-compose ps
```

---

## Summary

This setup uses **Dragonfly**, **MinIO**, and **SurrealDB** to handle the simulation's data and communication needs. Use the table to understand each service's purpose and the Mermaid diagram to visualize their interaction.

Happy simulating! 🚀

Start the services:

```bash
docker-compose up -d
```

---

## Client Setup

### 1. Prerequisites

Ensure you have the following installed:

- **Python**: Version 3.9 or higher
- **Redis CLI**: For sending commands to the server

### 2. Connect to the Server

Make sure the server is running and accessible at the appropriate Redis host (`localhost` by default).

### 3. Run the Client

Run the client to interact with the simulation:

```bash
python simulation/visualization/client.py --redis-host localhost --redis-port 6379 --redis-db 0
```

---

You're now ready to use **ServerIPC**!