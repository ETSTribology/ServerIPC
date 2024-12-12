import logging
from typing import Any, Dict, Optional

import redis

from simulation.backend.backend import Backend
from simulation.controller.model import Request, Response
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class RedisBackend(Backend):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RedisBackend with configuration.

        Args:
            config: A dictionary containing the Redis configuration.
        """
        super().__init__(config)
        self.backend_config = config.get("backend", {}).get("config", {})
        self.client = None
        self.pubsub = None
        self.connected = False

    def connect(self):
        """
        Connect to the Redis server.
        """
        try:
            self.client = redis.StrictRedis(
                host=self.backend_config.get("host", "localhost"),
                port=self.backend_config.get("port", 6379),
                db=self.backend_config.get("db", 0),
                password=self.backend_config.get("password"),
                ssl=self.backend_config.get("ssl", False),
                decode_responses=True,
            )
            # Test connection
            self.client.ping()
            self.pubsub = self.client.pubsub()
            self.connected = True
            logger.info(
                SimulationLogMessageCode.REDIS_CONNECTED.details("Connected to Redis backend.")
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_CONNECTION_FAILED.details(
                    f"Failed to connect to Redis: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.BACKEND_INITIALIZATION,
                "Failed to connect to Redis",
                details=str(e),
            )

    def disconnect(self):
        """
        Disconnect from the Redis server.
        """
        try:
            if self.pubsub:
                self.pubsub.close()
            if self.client:
                self.client.close()
                self.connected = False
                logger.info(
                    SimulationLogMessageCode.REDIS_DISCONNECTED.details(
                        "Disconnected from Redis backend."
                    )
                )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_DISCONNECTION_FAILED.details(
                    f"Error while disconnecting Redis backend: {e}"
                )
            )

    def write(self, key: str, value: Any) -> None:
        """
        Write data to Redis.

        Args:
            key: The key under which the data will be stored.
            value: The data to store.
        """
        try:
            self.client.set(key, value)
            logger.info(
                SimulationLogMessageCode.REDIS_WRITE_SUCCESS.details(
                    f"Written data to key '{key}'."
                )
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_WRITE_FAILURE.details(
                    f"Failed to write data to key '{key}': {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.FILE_IO, f"Failed to write data to key '{key}'", details=str(e)
            )

    def read(self, key: str) -> Any:
        """
        Read data from Redis.

        Args:
            key: The key of the data to retrieve.

        Returns:
            The retrieved data.
        """
        try:
            value = self.client.get(key)
            logger.info(
                SimulationLogMessageCode.REDIS_READ_SUCCESS.details(f"Read data from key '{key}'.")
            )
            return value
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                    f"Failed to read data from key '{key}': {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.FILE_IO, f"Failed to read data from key '{key}'", details=str(e)
            )

    def send_response(self, response: Response) -> None:
        """
        Publish a response message to the 'responses' channel.

        Args:
            response: The Response to send.
        """
        try:
            message = response.to_json()
            self.client.publish("responses", message)
            logger.info(
                SimulationLogMessageCode.REDIS_WRITE_SUCCESS.details(
                    "Published response to 'responses' channel."
                )
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_WRITE_FAILURE.details(
                    f"Failed to publish response: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to publish response",
                details=str(e),
            )

    def get_command(self) -> Optional[Request]:
        """
        Subscribe to the 'commands' channel and retrieve a command message.

        Returns:
            A Request if available, else None.
        """
        try:
            self.pubsub.subscribe("commands")
            message = self.pubsub.get_message(timeout=1)
            if message and message["type"] == "message":
                command_data = message["data"]
                command = Request.from_json(command_data)
                logger.info(
                    SimulationLogMessageCode.REDIS_READ_SUCCESS.details(
                        "Received command from 'commands' channel."
                    )
                )
                return command
            return None
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                    f"Failed to retrieve command: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to retrieve command",
                details=str(e),
            )
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Redis backend.

        Returns:
            A dictionary containing the backend status and connection details.
        """
        try:
            info = self.client.info() if self.connected else {}
            return {
                "connected": self.connected,
                "host": self.backend_config.get("host", "localhost"),
                "port": self.backend_config.get("port", 6379),
                "db": self.backend_config.get("db", 0),
                "info": info,
            }
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                    f"Failed to retrieve Redis status: {e}"
                )
            )
            return {
                "connected": self.connected,
                "error": str(e),
            }
        
    def is_connected(self) -> bool:
        """
        Check if the Redis backend is connected.

        Returns:
            True if connected, else False.
        """
        return self.connected
