from minio import Minio
from minio.error import S3Error
from typing import Any, Dict, List
import logging
import io

logger = logging.getLogger(__name__)

class MinIO:
    """
    MinIO-based storage backend implementation for managing multiple buckets and files.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MinIO backend with the provided configuration.

        Args:
            config: Configuration dictionary with MinIO details.
        """
        self.config = config
        self.client = Minio(
            endpoint=f"{config.get('storage', {}).get('host', 'localhost')}:{config.get('storage', {}).get('port', 9000)}",
            access_key=config.get("storage", {}).get("access_key", "minioadmin"),
            secret_key=config.get("storage", {}).get("secret_key", "minioadminpassword"),
            secure=config.get("storage", {}).get("secure", False)
        )
        logger.info(f"{config}")
        self.extensions = config.get("extensions", {})
        self.connected = False

    def connect(self):
        """
        Establish a connection to the MinIO instance and ensure buckets exist.
        """
        self._ensure_buckets()
        self.connected = True
        logger.info("Connected to MinIO instance and ensured all buckets exist.")

    def _ensure_buckets(self):
        """
        Ensure all buckets defined in the extensions exist in MinIO.
        """
        try:
            for extension, ext_config in self.extensions.items():
                if ext_config.get("enabled", False):
                    bucket_name = ext_config["directory"].strip("/")
                    if not self.client.bucket_exists(bucket_name):
                        self.client.make_bucket(bucket_name)
                        logger.info(f"Bucket '{bucket_name}' created for extension '{extension}'.")
                    else:
                        logger.info(f"Bucket '{bucket_name}' already exists for extension '{extension}'.")
        except S3Error as err:
            logger.error(f"Error ensuring buckets: {err}")
            raise ConnectionError("Failed to initialize MinIO buckets.")

    def write(self, extension: str, filename: str, content: bytes) -> None:
        """
        Write any type of file to the bucket for a specific extension.

        Args:
            extension: The extension name.
            filename: The filename (object name) to store the data.
            content: The content to store (as bytes).
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")
        
        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")
        
        bucket_name = self.extensions[extension]["directory"].strip("/")
        try:
            content_stream = io.BytesIO(content)
            self.client.put_object(bucket_name, filename, content_stream, len(content))
            logger.info(f"File '{filename}' written to bucket '{bucket_name}' for extension '{extension}'.")
        except S3Error as err:
            logger.error(f"Error writing file '{filename}' to bucket '{bucket_name}': {err}")
            raise

    def read(self, extension: str, filename: str) -> bytes:
        """
        Read any file from the bucket for a specific extension.

        Args:
            extension: The extension name.
            filename: The filename (object name) to retrieve.

        Returns:
            The retrieved file content (as bytes).
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")
        
        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")
        
        bucket_name = self.extensions[extension]["directory"].strip("/")
        try:
            response = self.client.get_object(bucket_name, filename)
            return response.read()
        except S3Error as err:
            logger.error(f"Error reading file '{filename}' from bucket '{bucket_name}': {err}")
            raise

    def delete(self, extension: str, filename: str) -> None:
        """
        Delete a file from the bucket for a specific extension.

        Args:
            extension: The extension name.
            filename: The filename (object name) to delete.
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")
        
        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")
        
        bucket_name = self.extensions[extension]["directory"].strip("/")
        try:
            self.client.remove_object(bucket_name, filename)
            logger.info(f"File '{filename}' deleted from bucket '{bucket_name}' for extension '{extension}'.")
        except S3Error as err:
            logger.error(f"Error deleting file '{filename}' from bucket '{bucket_name}': {err}")
            raise

    def list_files(self, extension: str) -> List[str]:
        """
        List all files (objects) in the bucket for a specific extension.

        Args:
            extension: The extension name.

        Returns:
            A list of file names in the bucket.
        """
        if not self.connected:
            raise ConnectionError("Storage backend is not connected.")
        
        if extension not in self.extensions or not self.extensions[extension].get("enabled", False):
            raise ValueError(f"Extension '{extension}' is not enabled.")
        
        bucket_name = self.extensions[extension]["directory"].strip("/")
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as err:
            logger.error(f"Error listing files in bucket '{bucket_name}': {err}")
            raise

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

        return self.extensions[extension]["directory"].strip("/")