"""Backend registry for communication mechanisms."""

from typing import Dict, Type, Any
from .base import BaseCommunicationBackend


class CommunicationBackendRegistry:
    """
    Registry for managing communication backends.
    Implements the registry factory pattern.
    """
    
    _backends: Dict[str, Type[BaseCommunicationBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[BaseCommunicationBackend]) -> None:
        """
        Register a new communication backend.
        
        Args:
            name: Unique identifier for the backend
            backend_class: Backend class to register
        """
        cls._backends[name] = backend_class

    @classmethod
    def get(cls, name: str) -> Type[BaseCommunicationBackend]:
        """
        Retrieve a registered backend.
        
        Args:
            name: Backend identifier
        
        Returns:
            Registered backend class
        
        Raises:
            KeyError: If backend is not registered
        """
        if name not in cls._backends:
            raise KeyError(f"Backend '{name}' not registered")
        return cls._backends[name]

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseCommunicationBackend:
        """
        Create and configure a backend instance.
        
        Args:
            name: Backend identifier
            config: Configuration dictionary
        
        Returns:
            Configured backend instance
        """
        backend_class = cls.get(name)
        backend = backend_class()
        backend.connect(config)
        return backend

    @classmethod
    def list_backends(cls) -> list:
        """
        List all registered backends.
        
        Returns:
            List of registered backend names
        """
        return list(cls._backends.keys())
