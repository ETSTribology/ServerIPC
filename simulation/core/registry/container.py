import logging

from simulation.core.registry.registry import Registry

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
            
            # Add default registries
            default_registries = [
                # Board-related
                ("board", "simulation.board.board.BoardBase"),

                # Parameters
                ("parameters", "simulation.core.parameters.ParametersBase"),

                # Contact-related
                ("barrier_initializer", "simulation.core.contact.barrier_initializer.BarrierInitializerBase"),
                ("barrier_updater", "simulation.core.contact.barrier_updater.BarrierUpdaterBase"),
                ("ccd", "simulation.core.contact.ccd.CCDBase"),

                # Mathematical
                ("gradient", "simulation.core.math.gradient.GradientBase"),
                ("hessian", "simulation.core.math.hessian.HessianBase"),
                ("potential", "simulation.core.math.potential.PotentialBase"),

                # Solvers
                ("linear_solver", "simulation.core.solvers.linear.LinearSolverBase"),
                ("line_search", "simulation.core.solvers.line_search.LineSearchBase"),
                ("optimizer", "simulation.core.solvers.optimizer.OptimizerBase"),

                # Logging
                ("log_handler", "simulation.core.utils.logs.handler.LogHandlerBase"),

                # Network-related
                ("database", "simulation.nets.db.db.DatabaseBase"),
                ("storage", "simulation.nets.storage.storage.StorageBase")
            ]
            
            for registry_name, base_class_path in default_registries:
                try:
                    module_name, class_name = base_class_path.rsplit(".", 1)
                    base_class = getattr(
                        __import__(module_name, fromlist=[class_name]), class_name
                    )
                    cls._instance._registries[registry_name] = Registry(base_class)
                    setattr(cls._instance, registry_name, cls._instance._registries[registry_name])
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not initialize registry '{registry_name}': {e}")
            
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
