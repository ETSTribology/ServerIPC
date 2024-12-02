import logging
import os
import shutil
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Type

from minio import Minio
from minio.error import S3Error

from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register
from simulation.core.utils.singleton import SingletonMeta

logger = logging.getLogger(__name__)


class StorageBase(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def create_bucket(self, bucket_name: str) -> None:
        """Create a storage bucket if it does not exist."""
        pass

    @abstractmethod
    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload a file to a storage bucket."""
        pass

    @abstractmethod
    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """Download a file from a storage bucket."""
        pass

    @abstractmethod
    def list_objects(self, bucket_name: str) -> list:
        """List objects in a storage bucket."""
        pass

    @abstractmethod
    def delete_object(self, bucket_name: str, object_name: str) -> None:
        """Delete an object from a storage bucket."""
        pass


@register(type="storage", name="minio")
class MinIOStorage(StorageBase):
    """MinIO storage implementation of StorageBase."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("MinIOStorage client initialized.")

    def create_bucket(self, bucket_name: str) -> None:
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Bucket '{bucket_name}' created successfully.")
            else:
                self.logger.info(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            self.logger.error(f"Failed to create bucket '{bucket_name}': {e}")
            raise

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        try:
            self.client.fput_object(bucket_name, object_name, file_path, content_type=content_type)
            self.logger.info(
                f"Uploaded '{file_path}' to bucket '{bucket_name}' as '{object_name}'."
            )
        except S3Error as e:
            self.logger.error(f"Failed to upload file '{file_path}' to bucket '{bucket_name}': {e}")
            raise

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            self.logger.info(
                f"Downloaded '{object_name}' from bucket '{bucket_name}' to '{file_path}'."
            )
        except S3Error as e:
            self.logger.error(
                f"Failed to download file '{object_name}' from bucket '{bucket_name}': {e}"
            )
            raise

    def list_objects(self, bucket_name: str) -> list:
        try:
            objects = self.client.list_objects(bucket_name, recursive=True)
            object_list = [obj.object_name for obj in objects]
            self.logger.info(f"Objects in bucket '{bucket_name}': {object_list}")
            return object_list
        except S3Error as e:
            self.logger.error(f"Failed to list objects in bucket '{bucket_name}': {e}")
            raise

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        try:
            self.client.remove_object(bucket_name, object_name)
            self.logger.info(f"Deleted object '{object_name}' from bucket '{bucket_name}'.")
        except S3Error as e:
            self.logger.error(
                f"Failed to delete object '{object_name}' from bucket '{bucket_name}': {e}"
            )
            raise


class StorageFactoryy(metaclass=SingletonMeta):
    """Factory for creating storage instances."""

    @lru_cache(maxsize=None)
    def get_class(self, type_lower: str):
        """Retrieve and cache the storage class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.storage.get(type_lower)

    def create(self, type: str, **kwargs) -> StorageBase:
        """Factory method to create a storage instance.

        :param type: The type of storage (e.g., 'minio').
        :param kwargs: Additional parameters for the storage backend.
        :return: An instance of StorageBase.
        """
        type_lower = type.lower()
        try:
            storage_cls = self.get_class(type_lower)
            storage_instance = storage_cls(**kwargs)
            logger.info(f"Storage '{type_lower}' created successfully.")
            return storage_instance
        except Exception as e:
            logger.error(f"Failed to create storage '{type_lower}': {e}")
            raise RuntimeError(f"Error during storage initialization for type '{type_lower}': {e}")


@register(type="storage", name="local")
class LocalStorage(StorageBase):
    """Local filesystem storage implementation of StorageBase."""

    def __init__(self, base_directory: str = "./storage"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_directory = base_directory
        self.logger.info(f"Initializing LocalStorage with base directory: {self.base_directory}")
        self._ensure_base_directory()

    def _ensure_base_directory(self) -> None:
        """Ensure that the base directory exists."""
        try:
            os.makedirs(self.base_directory, exist_ok=True)
            self.logger.info(f"Base directory '{self.base_directory}' is ready.")
        except Exception as e:
            self.logger.error(f"Failed to create base directory '{self.base_directory}': {e}")
            raise

    def _get_bucket_path(self, bucket_name: str) -> str:
        """Get the filesystem path for a given bucket."""
        return os.path.join(self.base_directory, bucket_name)

    def create_bucket(self, bucket_name: str) -> None:
        """Create a storage bucket (directory) if it does not exist."""
        bucket_path = self._get_bucket_path(bucket_name)
        try:
            os.makedirs(bucket_path, exist_ok=True)
            self.logger.info(f"Bucket '{bucket_name}' created at '{bucket_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to create bucket '{bucket_name}': {e}")
            raise

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload a file to a storage bucket (copy it to the bucket directory)."""
        bucket_path = self._get_bucket_path(bucket_name)
        destination_path = os.path.join(bucket_path, object_name)
        try:
            shutil.copy(file_path, destination_path)
            self.logger.info(
                f"Uploaded '{file_path}' to bucket '{bucket_name}' as '{object_name}'."
            )
        except Exception as e:
            self.logger.error(f"Failed to upload file '{file_path}' to bucket '{bucket_name}': {e}")
            raise

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> None:
        """Download a file from a storage bucket (copy it from the bucket directory)."""
        bucket_path = self._get_bucket_path(bucket_name)
        source_path = os.path.join(bucket_path, object_name)
        try:
            shutil.copy(source_path, file_path)
            self.logger.info(
                f"Downloaded '{object_name}' from bucket '{bucket_name}' to '{file_path}'."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to download file '{object_name}' from bucket '{bucket_name}': {e}"
            )
            raise

    def list_objects(self, bucket_name: str) -> List[str]:
        """List objects (files) in a storage bucket."""
        bucket_path = self._get_bucket_path(bucket_name)
        try:
            objects = os.listdir(bucket_path)
            self.logger.info(f"Objects in bucket '{bucket_name}': {objects}")
            return objects
        except Exception as e:
            self.logger.error(f"Failed to list objects in bucket '{bucket_name}': {e}")
            raise

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        """Delete an object (file) from a storage bucket."""
        bucket_path = self._get_bucket_path(bucket_name)
        object_path = os.path.join(bucket_path, object_name)
        try:
            os.remove(object_path)
            self.logger.info(f"Deleted object '{object_name}' from bucket '{bucket_name}'.")
        except Exception as e:
            self.logger.error(
                f"Failed to delete object '{object_name}' from bucket '{bucket_name}': {e}"
            )
            raise


class StorageFactory(metaclass=SingletonMeta):
    """Factory for creating storage instances."""

    def __init__(self):
        self.registry_container = RegistryContainer()
        self.logger = logging.getLogger(self.__class__.__name__)

    @lru_cache(maxsize=None)
    def get_class(self, type_lower: str) -> Type[StorageBase]:
        """Retrieve and cache the storage class from the registry."""
        storage_cls = self.registry_container.get_storage_class(type_lower)
        if not storage_cls:
            self.logger.error(f"No storage class registered under name '{type_lower}'.")
            raise ValueError(f"No storage class registered under name '{type_lower}'.")
        return storage_cls

    def create(self, type: str, **kwargs) -> StorageBase:
        """Factory method to create a storage instance.

        :param type: The type of storage (e.g., 'minio', 'local').
        :param kwargs: Additional parameters for the storage backend.
        :return: An instance of StorageBase.
        """
        type_lower = type.lower()
        try:
            storage_cls = self.get_class(type_lower)
            storage_instance = storage_cls(**kwargs)
            self.logger.info(f"Storage '{type_lower}' created successfully.")
            return storage_instance
        except Exception as e:
            self.logger.error(f"Failed to create storage '{type_lower}': {e}")
            raise RuntimeError(f"Error during storage initialization for type '{type_lower}': {e}")
