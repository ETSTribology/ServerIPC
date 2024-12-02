from __future__ import annotations

from typing import (
    Any,
    Dict,
    Generic,
    Type,
    TypeVar,
    Callable
)

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic type variable

class Registry(Generic[T]):
    """A generic registry class to register and retrieve classes.

    Parameters
    ----------
    T : TypeVar
        The type of classes to register.

    """

    def __init__(self, base_class: Type[T]):
        """Initialize the registry with a base class.

        Parameters
        ----------
        base_class : Type[T]
            The base class that all registered classes must inherit from.

        """
        if not isinstance(base_class, type):
            raise TypeError("base_class must be a class type.")
        self.base_class = base_class
        self._registry: Dict[str, Callable[..., T]] = {}
        logger.debug(
            f"Initialized registry for base class '{self.base_class.__name__}'."
        )

    def register(self, name: str):
        """Decorator to register a class with a given name.

        Parameters
        ----------
        name : str
            The name identifier for the class.

        """

        def decorator(cls: Type[T]):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Registration name must be a non-empty string.")
            if not issubclass(cls, self.base_class):
                raise TypeError(
                    f"Cannot register {cls.__name__} as it is not a subclass of {self.base_class.__name__}."
                )
            name_lower = name.lower()
            if name_lower in self._registry:
                raise KeyError(
                    f"Class '{cls.__name__}' is already registered under name '{name_lower}'."
                )
            self._registry[name_lower] = cls
            logger.debug(f"Registered '{name_lower}' with class '{cls.__name__}'.")
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        """Retrieve a class from the registry by name.

        Parameters
        ----------
        name : str
            The name identifier for the class.

        Returns
        -------
        Type[T]
            The registered class.

        Raises
        ------
        ValueError
            If the name is not registered.

        """
        if not isinstance(name, str):
            raise TypeError("The name must be a string.")
        name_lower = name.lower()
        if name_lower not in self._registry:
            available = ", ".join(self._registry.keys())
            logger.error(
                f"Unsupported type: '{name_lower}'. Available types: {available}."
            )
            raise ValueError(
                f"Unsupported type: '{name_lower}'. Available types: {available}."
            )
        cls = self._registry[name_lower]
        logger.debug(f"Retrieved class '{cls.__name__}' for type '{name_lower}'.")
        return cls

    def list(self) -> str:
        """List all registered names.

        Returns
        -------
        str
            A comma-separated string of all registered names.

        """
        registered = ", ".join(self._registry.keys())
        logger.debug(f"Available registered types: {registered}.")
        return registered

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered.

        Parameters
        ----------
        name : str
            The name to check.

        Returns
        -------
        bool
            True if the name is registered, False otherwise.
        """
        return name.lower() in self._registry

    def __iter__(self):
        """Iterate over registered names.

        Returns
        -------
        Iterator[str]
            An iterator over registered names.
        """
        return iter(self._registry.keys())

    @property
    def registry(self) -> Dict[str, Callable[..., T]]:
        """Expose the internal registry dictionary.

        Returns
        -------
        Dict[str, Callable[..., T]]
            A dictionary of registered classes.
        """
        return self._registry