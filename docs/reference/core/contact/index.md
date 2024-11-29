# Contact Module

The `simulation/core/contact` module provides implementations for handling contact-related computations in simulations, including barrier initializers, barrier updaters, and collision detection (CCD).

---

## Barrier Initializer

### Description

The `BarrierInitializer` class initializes barrier parameters based on input data. It is responsible for setting up stiffness and other barrier-related parameters essential for simulation stability and collision handling.

```mermaid
classDiagram
    class BarrierInitializerBase {
        <<abstract>>
        - params: ParametersBase
        + __call__(x: np.ndarray, gU: np.ndarray, gB: np.ndarray)
    }
    class BarrierInitializer {
        + __call__(x: np.ndarray, gU: np.ndarray, gB: np.ndarray)
    }
    BarrierInitializerBase <|-- BarrierInitializer : implements
```

### Class

::: simulation.core.contact.barrier_initializer

---

## Barrier Updater

### Description

The `BarrierUpdater` class updates barrier stiffness dynamically during simulations based on the current positions. This ensures stability and accuracy as the simulation evolves.

```mermaid
classDiagram
    class BarrierUpdaterBase {
        <<abstract>>
        - params: ParametersBase
        + __call__(xk: np.ndarray)
    }
    class BarrierUpdater {
        + __call__(xk: np.ndarray)
    }
    BarrierUpdaterBase <|-- BarrierUpdater : implements
```

### Class

::: simulation.core.contact.barrier_updater

---

## CCD (Collision Detection)

### Description

The `CCD` class calculates the maximum collision-free step size for position updates, ensuring no interpenetration of objects during simulations.

```mermaid
classDiagram
    class CCDBase {
        <<abstract>>
        - params : ParametersBase
        - broad_phase_method : str
        - alpha : float
        + __call__(x: np.ndarray, dx: np.ndarray) : float
    }
    class CCD {
        + __call__(x: np.ndarray, dx: np.ndarray) : float
    }
    CCDBase <|-- CCD : implements

```

### Class

::: simulation.core.contact.ccd
