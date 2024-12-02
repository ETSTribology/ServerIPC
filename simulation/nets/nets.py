from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Nets(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establishes the connection."""
        pass

    @abstractmethod
    def listen_commands(self) -> None:
        """Listens for incoming commands."""
        pass

    @abstractmethod
    def get_command(self) -> Optional[str]:
        """Retrieves the latest command."""
        pass

    @abstractmethod
    def set_data(self, key: str, data: str) -> None:
        """Stores data with the specified key."""
        pass

    @abstractmethod
    def get_data(self, key: str) -> Optional[Any]:
        """Retrieves data associated with the specified key."""
        pass

    @abstractmethod
    def publish_data(self, channel: str, data: str) -> None:
        """Publishes data to the specified channel."""
        pass

    @abstractmethod
    def serialize_data(self, data: Dict[str, Any], method: str = "pickle") -> Optional[str]:
        """Serializes data using the specified method."""
        pass

    @abstractmethod
    def deserialize_data(self, data_str: str, method: str = "pickle") -> Dict[str, Any]:
        """Deserializes data using the specified method."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the connection and cleans up resources."""
        pass
