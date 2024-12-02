from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class Backend(ABC):
    """
    Abstract base class for backend  implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backend  with configuration.

        Args:
            config: A dictionary containing backend-specific configuration.
        """
        self.config = config
        self.connected = False

    @abstractmethod
    def connect(self):
        """
        Establish a connection to the backend.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Close the connection to the backend.
        """
        pass

    @abstractmethod
    def write(self, key: str, value: Any) -> None:
        """
        Write data to the backend .

        Args:
            key: The key under which the data will be stored.
            value: The data to store.
        """
        pass

    @abstractmethod
    def read(self, key: str) -> Any:
        """
        Read data from the backend .

        Args:
            key: The key of the data to retrieve.

        Returns:
            The retrieved data.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Delete data from the backend .

        Args:
            key: The key of the data to delete.
        """
        pass