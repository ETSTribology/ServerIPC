# Solver Benchmarks

## Overview
This directory contains performance benchmarks for the ServerIPC project's solvers.

## Benchmark Categories
- `LinearSolverBenchmarks`: Performance tests for linear equation solving
- `OptimizerBenchmarks`: Optimization algorithm performance
- `LineSearchBenchmarks`: Line search algorithm efficiency
- `MemorySolverBenchmarks`: Memory usage measurements

## Running Benchmarks
To run benchmarks, use:
```bash
asv run
```

To generate an HTML report:
```bash
asv publish
```

## Benchmark Metrics
- Execution time
- Memory consumption
- Scalability across different problem sizes

## Requirements
- Python 3.8+
- NumPy
- ASV (Airspeed Velocity)
