from simulation.nets.nets import Nets
from simulation.core.utils.singleton import SingletonMeta
import logging

logger = logging.getLogger(__name__)


class NetsFactory(metaclass=SingletonMeta):
    """Factory for creating communication client instances based on registered types."""

    def __init__(self):
        self.registry_container = RegistryContainer()
        self.logger = logger

    def create(self, method: str, **kwargs) -> Nets:
        """
        Factory method to create a communication client instance.

        Parameters
        ----------
        method : str
            The type of communication method (e.g., 'redis', 'grpc', 'websocket').
        kwargs : dict
            Additional keyword arguments required for the specific communication client.

        Returns
        -------
        Nets
            An instance of a communication client implementing the Nets interface.

        Raises
        ------
        ValueError
            If the specified communication method is not registered.
        """
        method_lower = method.lower()
        try:
            cls = self.registry_container.get_nets_class(method_lower)
            if not cls:
                self.logger.error(f"No communication client registered under name '{method_lower}'.")
                raise ValueError(f"No communication client registered under name '{method_lower}'.")
            self.logger.info(f"Creating communication client for method '{method_lower}'.")
            return cls(**kwargs)
        except Exception as e:
            self.logger.error(f"Error creating communication client '{method_lower}': {e}")
            raise ValueError(f"Failed to create communication client '{method_lower}': {e}")
