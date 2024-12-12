import logging
from typing import Any, Dict

from simulation.backend.redis import RedisBackend
from simulation.backend.websocket import WebSocketBackend
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import BackendInitializationError
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class BackendFactory(metaclass=SingletonMeta):
    """Factory for creating and managing backend instances."""

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """Create or retrieve a backend instance based on configuration.

        Args:
            config: Backend configuration dictionary.

        Returns:
            Backend instance.

        Raises:
            SimulationError: If backend type is invalid or creation fails.
        """
        backend_type = config.get("backend", {}).get("backend", "redis")

        # Return existing instance if available
        if backend_type in BackendFactory._instances:
            logger.debug(
                SimulationLogMessageCode.BACKEND_INITIALIZED.details(
                    f"Reusing existing {backend_type} backend instance"
                )
            )
            return BackendFactory._instances[backend_type]

        # Create new instance
        logger.info(
            SimulationLogMessageCode.BACKEND_INITIALIZED.details(
                f"Creating new {backend_type} backend instance"
            )
        )
        try:
            if backend_type == "redis":
                backend_instance = RedisBackend(config)
            elif backend_type == "websocket":
                backend_instance = WebSocketBackend(config)
            else:
                raise BackendInitializationError(f"Unknown backend type: {backend_type}")

            backend_instance.connect()
            BackendFactory._instances[backend_type] = backend_instance
            return backend_instance

        except Exception as e:
            logger.error(
                SimulationLogMessageCode.BACKEND_FAILED.details(
                    f"Failed to create {backend_type} backend: {str(e)}"
                )
            )
            raise BackendInitializationError(
                f"Failed to create {backend_type} backend", details=str(e)
            )
