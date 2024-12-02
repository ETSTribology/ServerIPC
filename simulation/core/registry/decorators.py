import logging
from typing import Any, Type

from .container import RegistryContainer

logger = logging.getLogger(__name__)


def register(type: str, name: str):
    """Generalized decorator to register a class with a given component type and name."""

    def decorator(cls: Type[Any]):
        registry_container = RegistryContainer()
        type_lower = type.lower()
        try:
            # Debug logging
            logger.debug(f"Attempting to register '{cls.__name__}' as '{name}' in '{type_lower}' registry")
            logger.debug(f"Available registries: {list(registry_container._registries.keys())}")
            
            # Access the appropriate registry based on the component type
            registry = getattr(registry_container, type_lower)
            if registry is None:
                raise AttributeError(
                    f"No registry found for component type '{type_lower}'."
                )
            # Register the class using the registry's register method
            registry.register(name)(cls)
            logger.info(
                f"Registered '{cls.__name__}' as '{name}' in '{type_lower}' registry."
            )
            return cls
        except AttributeError as ae:
            logger.error(f"AttributeError: {str(ae)}")
            logger.error(f"Attempted type: {type_lower}")
            raise AttributeError(
                f"Component type '{type}' does not have an associated registry."
            ) from ae
        except Exception as e:
            logger.error(
                f"Failed to register class '{cls.__name__}' as '{name}' in '{type_lower}' registry: {e}"
            )
            raise e

    return decorator