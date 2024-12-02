# Registry

The `simulation/core/registry` module provides a flexible system for registering and retrieving components dynamically. It allows developers to create, manage, and access different registries for various simulation components.

---

## RegistryContainer

### Description

The `RegistryContainer` class is a singleton that manages multiple registries for different component types, such as `line_search`, `optimizer`, and others. It ensures that only one instance of each registry exists throughout the application.

```mermaid
classDiagram
    class RegistryContainer {
        - _instance: Optional[RegistryContainer]
        - _registries: Dict[str, Registry]
        + add_registry(registry_name: str, base_class_path: str)
        + get_registry(registry_name: str) : Registry
        + list_all()
    }
    RegistryContainer o-- Registry : manages
```

### Class

::: simulation.core.registry.container.RegistryContainer

---

## Registry

### Description

The `Registry` class is a generic system for registering and retrieving classes. It ensures that all registered classes inherit from a specified base class.

```mermaid
classDiagram
    class Registry {
        - base_class: Type
        - _registry: Dict[str, Callable[..., Any]]
        + register(name: str) : Callable
        + get(name: str) : Type
        + list() : str
    }
```

### Class

::: simulation.core.registry.registry.Registry

---

## Register Decorator

### Description

The `register` decorator is used to dynamically register a class in a specified registry. It associates a class with a name and ensures it is added to the correct registry.

```mermaid
classDiagram
    class register {
        + register(type: str, name: str) : Callable
    }
    register ..> RegistryContainer : uses
    register ..> Registry : adds
```

### Decorator

::: simulation.core.registry.decorators.register

---
