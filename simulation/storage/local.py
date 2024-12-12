import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Local:
    """
    Local storage backend implementation for managing multiple directories and files.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Local storage backend with the provided configuration.

        Args:
            config: Configuration dictionary for the storage.
        """
        self.directory = Path(config.get("storage", {}).get("config", {}).get("directory", "storage"))
        self.extensions = config.get("extensions", {})
        self.connected = False

    def connect(self):
        """
        Establish a connection to the local storage backend.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        self._ensure_directories()
        self.connected = True
        logger.info(f"Connected to local storage backend. Base directory: {self.directory}")

    def _ensure_directories(self):
        """
        Create directories for each enabled extension.
        """
        for extension, ext_config in self.extensions.items():
            if ext_config.get("enabled", False):
                ext_dir = self.directory / ext_config["directory"].strip("/")
                ext_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created for extension '{extension}': {ext_dir}")

    def write(self, extension: str, filename: str, content: bytes) -> None:
        """
        Write any type of file to the directory for a specific extension.

        Args:
            extension: The extension name.
            filename: The filename to store the data.
            content: The content to store (as bytes).
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")

        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            logger.warning(f"{self.extensions}")
            raise ValueError(f"Extension '{extension}' is not enabled.")

        ext_dir = self.directory / self.extensions[extension]["directory"].strip("/")
        file_path = ext_dir / filename
        with file_path.open("wb") as f:
            f.write(content)
        logger.info(
            f"File '{filename}' written for extension '{extension}' in directory '{ext_dir}'."
        )

    def read(self, extension: str, filename: str) -> bytes:
        """
        Read any file from the directory for a specific extension.

        Args:
            extension: The extension name.
            filename: The filename to retrieve the data.

        Returns:
            The content of the file (as bytes).
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")

        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")

        ext_dir = self.directory / self.extensions[extension]["directory"].strip("/")
        file_path = ext_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        with file_path.open("rb") as f:
            return f.read()

    def delete(self, extension: str, filename: str) -> None:
        """
        Delete a file from the directory for a specific extension.

        Args:
            extension: The extension name.
            filename: The filename to delete.
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")

        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")

        ext_dir = self.directory / self.extensions[extension]["directory"].strip("/")
        file_path = ext_dir / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"File '{file_path}' deleted for extension '{extension}'.")
        else:
            logger.warning(f"File '{file_path}' does not exist.")

    def list_files(self, extension: str) -> list:
        """
        List all files in the directory for a specific extension.

        Args:
            extension: The extension name.

        Returns:
            A list of file paths in the directory for the extension.
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")

        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")

        ext_dir = self.directory / self.extensions[extension]["directory"].strip("/")
        if not ext_dir.exists():
            raise FileNotFoundError(
                f"Directory '{ext_dir}' does not exist for extension '{extension}'."
            )

        return [str(file) for file in ext_dir.iterdir() if file.is_file()]

    def get_directory(self, extension: str) -> str:
        """
        Get the directory path for a specific extension.

        Args:
            extension: The extension name.

        Returns:
            The directory path for the extension.
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")

        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")

        return self.directory / self.extensions[extension]["directory"].strip("/")
