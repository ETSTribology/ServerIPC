import logging

from .registry import Registry

logger = logging.getLogger(__name__)


class RegistryContainer:
    """A container class that holds multiple registries for different component types.
    Implemented as a singleton to ensure a single instance throughout the application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RegistryContainer, cls).__new__(cls)
            cls._instance._registries = {}
            logger.debug(
                "RegistryContainer instance created with all registries initialized."
            )
        return cls._instance

    def add_registry(self, registry_name: str, base_class_path: str):
        """Dynamically add a registry for a given component type.

        Parameters
        ----------
        registry_name : str
            The name of the registry to create.
        base_class_path : str
            The import path of the base class for this registry.

        """
        if registry_name in self._registries:
            logger.debug(f"Registry '{registry_name}' already exists.")
            return

        try:
            module_name, class_name = base_class_path.rsplit(".", 1)
            base_class = getattr(
                __import__(module_name, fromlist=[class_name]), class_name
            )
            self._registries[registry_name] = Registry(base_class)
            setattr(self, registry_name, self._registries[registry_name])
            logger.info(
                f"Registry '{registry_name}' initialized with base class '{base_class_path}'."
            )
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to initialize registry '{registry_name}': {e}")
            raise ImportError(
                f"Could not import base class for registry '{registry_name}': {e}"
            )

    def get_registry(self, registry_name: str):
        """Retrieve a registry by name.

        Parameters
        ----------
        registry_name : str
            The name of the registry.

        Returns
        -------
        Registry
            The requested registry instance.

        Raises
        ------
        KeyError
            If the registry is not found.

        """
        if registry_name not in self._registries:
            raise KeyError(f"Registry '{registry_name}' not found.")
        return self._registries[registry_name]

    def list_all(self):
        """List all registered components across all registries."""
        for name, registry in self._registries.items():
            logger.info(f"Registered {name.capitalize()} Components: {registry.list()}")
