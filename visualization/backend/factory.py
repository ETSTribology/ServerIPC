import logging
from typing import Any, Dict
from visualization.core.utils.singleton import SingletonMeta
from visualization.backend.redis import RedisBackend
from visualization.backend.websocket import WebSocketBackend
from visualization.backend.grpc import GrpcBackend

logger = logging.getLogger(__name__)

class BackendFactory(metaclass=SingletonMeta):
    """
    Factory for creating and managing backend instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a backend instance based on the configuration.

        Args:
            config: A dictionary containing the backend configuration.

        Returns:
            An instance of the backend class.

        Raises:
            ValueError: If the backend type is not recognized or required fields are missing.
        """
        logger.info("Creating backend...")
        backend_config = config.get("backend", {})
        backend_type = backend_config.get("backend", "redis")

        if backend_type not in BackendFactory._instances:
            if backend_type == "redis":
                backend_instance = RedisBackend(config)
            elif backend_type == "websocket":
                backend_instance = WebSocketBackend(config)
            elif backend_type == "grpc":
                backend_instance = GrpcBackend(config)
            else:
                raise ValueError(f"Unknown backend type: {backend_type}")
            
            backend_instance.connect()
            BackendFactory._instances[backend_type] = backend_instance
        else:
            logger.info(f"Backend instance already exists: {backend_type}")

        return BackendFactory._instances[backend_type]
