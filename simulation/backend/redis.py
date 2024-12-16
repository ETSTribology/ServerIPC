import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import redis

from simulation.backend.backend import Backend
from simulation.controller.model import Request, Response
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class RedisBackend(Backend):
    def __init__(
        self,
        config: Dict[str, Any],
        command_handler: Optional[Callable[[Request], None]] = None,
        reconnect_interval: int = 5,
        listener_timeout: float = 1.0,
    ):
        """
        Initialize the RedisBackend with configuration.

        Args:
            config: A dictionary containing the Redis configuration.
            command_handler: A callable to handle incoming commands.
            reconnect_interval: Time in seconds between reconnection attempts.
            listener_timeout: Timeout in seconds for the listener's get_message.
        """
        super().__init__(config)
        self.backend_config = config.get("backend", {}).get("config", {})
        self.client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.connected = False
        self.reconnect_interval = reconnect_interval
        self.listener_timeout = listener_timeout
        self.command_handler = command_handler
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._listener_thread = threading.Thread(target=self._listen_commands, daemon=True)

    def connect(self):
        """
        Connect to the Redis server with retry logic.
        """
        with self._lock:
            while not self.connected and not self._stop_event.is_set():
                try:
                    self.client = redis.Redis(
                        host=self.backend_config.get("host", "localhost"),
                        port=self.backend_config.get("port", 6379),
                        db=self.backend_config.get("db", 0),
                        password=self.backend_config.get("password"),
                        ssl=self.backend_config.get("ssl", False),
                        decode_responses=True,
                        socket_connect_timeout=5,  # Connection timeout
                        retry_on_timeout=True,
                    )
                    # Test connection
                    self.client.ping()
                    self.pubsub = self.client.pubsub(ignore_subscribe_messages=True)
                    self.pubsub.subscribe("commands")
                    self.connected = True
                    logger.info(
                        SimulationLogMessageCode.REDIS_CONNECTED.details(
                            "Connected to Redis backend."
                        )
                    )
                except redis.ConnectionError as e:
                    logger.error(
                        SimulationLogMessageCode.REDIS_CONNECTION_FAILED.details(
                            f"Failed to connect to Redis: {e}. Retrying in {self.reconnect_interval} seconds..."
                        )
                    )
                    time.sleep(self.reconnect_interval)
                except Exception as e:
                    logger.error(
                        SimulationLogMessageCode.REDIS_CONNECTION_FAILED.details(
                            f"Unexpected error while connecting to Redis: {e}"
                        )
                    )
                    raise SimulationError(
                        SimulationErrorCode.BACKEND_INITIALIZATION,
                        "Failed to connect to Redis",
                        details=str(e),
                    )

        # Start listener thread if not already started
        if not self._listener_thread.is_alive():
            self._listener_thread.start()

    def disconnect(self):
        """Disconnect from the Redis server gracefully."""
        with self._lock:
            self._stop_event.set()  # Signal listener thread to stop
            if self.pubsub:
                try:
                    self.pubsub.unsubscribe()
                    self.pubsub.close()
                    logger.info("Unsubscribed and closed Redis pubsub.")
                except Exception as e:
                    logger.error(f"Error while unsubscribing Redis pubsub: {e}")
                finally:
                    self.pubsub = None  # Reset pubsub to prevent invalid access
            if self.client:
                try:
                    self.client.close()
                    self.connected = False
                    logger.info("Disconnected from Redis backend.")
                except Exception as e:
                    logger.error(f"Error while closing Redis client: {e}")
            # Ensure listener thread is stopped
            if self._listener_thread.is_alive():
                self._listener_thread.join(timeout=2)
                logger.info("Redis command listener thread has been stopped.")

    def safe_command_handler(self, command):
        try:
            if callable(self.command_handler):
                self.command_handler(command)
            else:
                logger.error("Command handler is not callable.")
        except Exception as e:
            logger.error(f"Error in command handler: {e}")

    def _listen_commands(self):
        logger.info("Redis command listener thread started.")
        while not self._stop_event.is_set():
            if not self.connected:
                logger.warning(
                    "Listener detected Redis is disconnected. Attempting to reconnect..."
                )
                self.connect()
                if not self.connected:
                    logger.error("Failed to reconnect to Redis. Listener will retry.")
                    time.sleep(self.reconnect_interval)
                    continue

            try:
                message = self.pubsub.get_message(timeout=self.listener_timeout)
                if message:
                    command_data = message.get("data")
                    if command_data:
                        try:
                            command = Request.from_json(command_data)
                            logger.info(f"Received command: {command.command_name}")
                            if callable(self.command_handler):
                                asyncio.run(self.command_handler(command))
                            else:
                                logger.error("Command handler is not callable.")
                        except Exception as e:
                            logger.error(f"Error processing command: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in listener thread: {e}")
                time.sleep(self.reconnect_interval)
        logger.info("Redis command listener thread stopped.")


    def write(self, key: str, value: Any) -> None:
        """
        Write data to Redis.

        Args:
            key: The key under which the data will be stored.
            value: The data to store.
        """
        if not self.connected:
            logger.warning(
                "Attempted to write to Redis while disconnected. Attempting to reconnect..."
            )
            self.connect()
            if not self.connected:
                raise SimulationError(
                    SimulationErrorCode.NETWORK_COMMUNICATION,
                    "Cannot write to Redis: Not connected.",
                )
        try:
            self.client.set(key, value)
            logger.info(
                SimulationLogMessageCode.REDIS_WRITE_SUCCESS.details(
                    f"Written data to key '{key}'."
                )
            )
        except redis.ConnectionError as e:
            logger.error(
                SimulationLogMessageCode.REDIS_WRITE_FAILURE.details(
                    f"Connection error while writing data to key '{key}': {e}"
                )
            )
            self.connected = False
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                f"Failed to write data to key '{key}' due to connection error.",
                details=str(e),
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
        if not self.connected:
            logger.warning(
                "Attempted to read from Redis while disconnected. Attempting to reconnect..."
            )
            self.connect()
            if not self.connected:
                raise SimulationError(
                    SimulationErrorCode.NETWORK_COMMUNICATION,
                    "Cannot read from Redis: Not connected.",
                )
        try:
            value = self.client.get(key)
            logger.info(
                SimulationLogMessageCode.REDIS_READ_SUCCESS.details(f"Read data from key '{key}'.")
            )
            return value
        except redis.ConnectionError as e:
            logger.error(
                SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                    f"Connection error while reading data from key '{key}': {e}"
                )
            )
            self.connected = False
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                f"Failed to read data from key '{key}' due to connection error.",
                details=str(e),
            )
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
        """
        if not self.connected:
            logger.warning(
                "Attempted to publish to Redis while disconnected. Attempting to reconnect..."
            )
            self.connect()
            if not self.connected:
                raise SimulationError(
                    SimulationErrorCode.NETWORK_COMMUNICATION,
                    "Cannot publish response to Redis: Not connected.",
                )
        try:
            message = response.to_json()
            self.client.publish("responses", message)  # Blocking publish method
            logger.info(
                SimulationLogMessageCode.REDIS_WRITE_SUCCESS.details(
                    "Published response to 'responses' channel."
                )
            )
        except redis.ConnectionError as e:
            logger.error(
                SimulationLogMessageCode.REDIS_WRITE_FAILURE.details(
                    f"Connection error while publishing response: {e}"
                )
            )
            self.connected = False
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to publish response due to connection error.",
                details=str(e),
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
        Retrieve a command message from the 'commands' channel.

        Returns:
            A Request if available, else None.
        """
        if not self.connected:
            logger.warning(
                "Attempted to read from Redis while disconnected. Attempting to reconnect..."
            )
            self.connect()
            if not self.connected:
                raise SimulationError(
                    SimulationErrorCode.NETWORK_COMMUNICATION,
                    "Cannot read from Redis: Not connected.",
                )
        try:
            message = self.pubsub.get_message(timeout=self.listener_timeout)
            if message:
                command_data = message.get("data")
                if command_data:
                    try:
                        command = Request.from_json(command_data)
                        logger.info(
                            SimulationLogMessageCode.REDIS_READ_SUCCESS.details(
                                f"Received command: {command.command_name}"
                            )
                        )
                        return command
                    except json.JSONDecodeError as e:
                        logger.error(
                            SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                                f"Invalid JSON format for command: {e}"
                            )
                        )
                    except Exception as e:
                        logger.error(
                            SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                                f"Error processing command: {e}"
                            )
                        )
            return None
        except redis.ConnectionError as e:
            logger.error(
                SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                    f"Connection error while reading command: {e}"
                )
            )
            self.connected = False
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to read command due to connection error.",
                details=str(e),
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.REDIS_READ_FAILURE.details(f"Failed to read command: {e}")
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to read command",
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
            status = {
                "connected": self.connected,
                "host": self.backend_config.get("host", "localhost"),
                "port": self.backend_config.get("port", 6379),
                "db": self.backend_config.get("db", 0),
                "info": info,
            }
            logger.debug(f"Redis status: {status}")
            return status
        except redis.ConnectionError as e:
            logger.error(
                SimulationLogMessageCode.REDIS_READ_FAILURE.details(
                    f"Connection error while retrieving Redis status: {e}"
                )
            )
            self.connected = False
            return {
                "connected": self.connected,
                "error": f"Connection error: {e}",
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

    def _handle_command(self, command: Request):
        """
        Handle a received command.

        Args:
            command: The Request object representing the command.
        """
        # Implement the logic to handle the command
        # For example, dispatch to a CommandDispatcher or process directly
        logger.info(
            f"Handling command: {command.command_name} with parameters: {command.parameters}"
        )
        if self.command_handler:
            try:
                self.command_handler(command)
            except Exception as e:
                logger.error(f"Error in command handler: {e}")
        else:
            logger.warning("No command handler provided to process the command.")

    def set_command_handler(self, handler: Callable[[Request], None]):
        """
        Set the command handler callable.

        Args:
            handler: A callable that takes a Request object and processes it.
        """
        if not callable(handler):
            raise TypeError("Command handler must be callable")
        self.command_handler = handler
        logger.info("Command handler has been set.")

    def start_listener(self):
        """
        Start the command listener thread.
        """
        if not self._listener_thread.is_alive():
            self._listener_thread = threading.Thread(target=self._listen_commands, daemon=True)
            self._listener_thread.start()
            logger.info("Redis command listener thread started.")

    def stop_listener(self):
        """
        Stop the command listener thread gracefully.
        """
        self._stop_event.set()
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2)
            logger.info("Redis command listener thread has been stopped.")
