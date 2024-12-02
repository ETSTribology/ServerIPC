import logging
from typing import Any, Dict
from visualization.core.utils.singleton import SingletonMeta
from visualization.storage.local import Local
from visualization.storage.minio import MinIO

logger = logging.getLogger(__name__)


class StorageFactory(metaclass=SingletonMeta):
    """
    Factory for creating and managing storage backend instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a storage instance based on the configuration.

        Args:
            config: A dictionary containing the storage configuration.

        Returns:
            An instance of the storage class.

        Raises:
            ValueError: If the storage type is not recognized or required fields are missing.
        """
        logger.info("Creating storage...")
        storage_config = config.get("storage", {})
        storage_type = storage_config.get("backend", "local").lower()
        backend_config = storage_config.get("config", {})

        logger.info(f"Storage type: {storage_type}")
        logger.debug(f"Backend configuration: {backend_config}")

        if storage_type not in StorageFactory._instances:
            if storage_type == "local":
                storage_instance = Local(config)
                storage_instance.connect()
            elif storage_type == "minio":
                storage_instance = MinIO(config)
                storage_instance.connect()
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
            StorageFactory._instances[storage_type] = storage_instance
        else:
            logger.info("Storage instance already created.")
        return StorageFactory._instances[storage_type]
