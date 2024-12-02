from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Storage(ABC):
    """
    Abstract base class for backend storage implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backend storage with configuration.

        Args:
            config: A dictionary containing backend-specific configuration.
        """
        self.config = config

    @abstractmethod
    def connect(self):
        """
        Establish a connection to the backend storage.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Close the connection to the backend storage.
        """
        pass

    @abstractmethod
    def write(self, location: str, filename: str, content: bytes) -> None:
        """
        Write data to the backend storage.

        Args:
            location: The location (e.g., directory or bucket name) where the file will be stored.
            filename: The name of the file to store.
            content: The content of the file to store (as bytes).
        """
        pass

    @abstractmethod
    def read(self, location: str, filename: str) -> bytes:
        """
        Read data from the backend storage.

        Args:
            location: The location (e.g., directory or bucket name) from where the file will be retrieved.
            filename: The name of the file to retrieve.

        Returns:
            The content of the file (as bytes).
        """
        pass

    @abstractmethod
    def delete(self, location: str, filename: str) -> None:
        """
        Delete a file from the backend storage.

        Args:
            location: The location (e.g., directory or bucket name) from where the file will be deleted.
            filename: The name of the file to delete.
        """
        pass

    @abstractmethod
    def list_files(self, location: str) -> List[str]:
        """
        List all files in a specific location in the backend storage.

        Args:
            location: The location (e.g., directory or bucket name) to list files.

        Returns:
            A list of filenames in the specified location.
        """
        pass

    @abstractmethod
    def get_directory(self, extension: str) -> str:
        """
        Get the directory path for a specific extension.

        Args:
            extension: The extension name.

        Returns:
            The directory path for the extension.
        """
        pass