# States

The `simulation/core/states` module provides the framework for managing and evolving states within the simulation. States represent the system's configuration and are updated iteratively through optimization and numerical methods.

---

## State

### Description

The `State` manages the evolution of the simulation's states. It applies numerical solvers and optimization algorithms to update states iteratively.

```mermaid
classDiagram
    class State {
        + current_state: State
        + initialize(state: State)
        + step(dt: float)
        + get_current_state() : State
    }
```

### Class

::: simulation.core.states.state

---
