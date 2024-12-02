"""Base communication backend interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable


class BaseCommunicationBackend(ABC):
    """Abstract base class for communication backends."""

    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> None:
        """
        Establish connection with the backend.
        
        Args:
            config: Configuration dictionary for connection
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Terminate the connection."""
        pass

    @abstractmethod
    def send_message(self, topic: str, message: Any) -> None:
        """
        Send a message to a specific topic.
        
        Args:
            topic: Message destination/channel
            message: Payload to send
        """
        pass

    @abstractmethod
    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to a specific topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Function to call when message is received
        """
        pass

    @abstractmethod
    def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a specific topic.
        
        Args:
            topic: Topic to unsubscribe from
        """
        pass
