import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from simulation.controller.model import Request, Response

logger = logging.getLogger(__name__)


class Backend(ABC):
    """
    Abstract base class for backend implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backend with configuration.

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
        Write data to the backend.

        Args:
            key: The key under which the data will be stored.
            value: The data to store.
        """
        pass

    @abstractmethod
    def read(self, key: str) -> Any:
        """
        Read data from the backend.

        Args:
            key: The key of the data to retrieve.

        Returns:
            The retrieved data.
        """
        pass

    @abstractmethod
    def send_response(self, response: Response) -> None:
        """
        Send a response message through the backend.

        Args:
            response: The ResponseMessage to send.
        """
        pass

    @abstractmethod
    def get_command(self) -> Optional[Request]:
        """
        Retrieve a command message from the backend.

        Returns:
            A Request if available, else None.
        """
        pass
