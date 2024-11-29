# nets/redis.py

import logging
import queue
import threading
from typing import Any, Dict, Optional

import redis
from nets.messages import RequestMessage, ResponseMessage
from nets.nets import Nets
from nets.serialization.factory import SerializerFactory

logger = logging.getLogger(__name__)


class Redis(Nets):
    """Redis-based communication client implementing the Nets interface."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = None,
        serializer_method: str = "json",
    ):
        """Initializes the Redis client.

        Parameters
        ----------
        host : str
            Redis server host.
        port : int
            Redis server port.
        db : int
            Redis database number.
        password : str, optional
            Redis server password.
        serializer_method : str, optional
            Serialization method ('json' or 'pickle').

        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.serializer_method = serializer_method
        self.serializer = SerializerFactory.get_serializer(self.serializer_method)

        self.redis_client = None
        self.pubsub = None
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.listener_thread = None
        self.stop_event = threading.Event()

        self.connect()
        self.start_listener()

    def connect(self) -> None:
        """Establishes a connection to the Redis server."""
        try:
            self.redis_client = redis.StrictRedis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}, db={self.db}")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def start_listener(self) -> None:
        """Starts a listener thread for incoming Redis messages."""
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(
            "simulation_commands"
        )  # Subscribe to the simulation_commands channel
        self.pubsub.subscribe(
            "simulation_responses"
        )  # Subscribe to the simulation_responses channel
        self.listener_thread = threading.Thread(
            target=self.listen_commands, daemon=True
        )
        self.listener_thread.start()
        logger.info("RedisClient listener thread started.")

    def listen_commands(self) -> None:
        """Listens for incoming commands from Redis channels."""
        logger.info("Listener thread is now listening for messages.")
        try:
            for message in self.pubsub.listen():
                if self.stop_event.is_set():
                    break
                if message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    logger.debug(f"Received message from '{channel}': {data}")
                    if channel == "simulation_commands":
                        self.handle_request(data)
                    elif channel == "simulation_responses":
                        self.handle_response(data)
        except Exception as e:
            logger.error(f"Error in listener thread: {e}")

    def handle_request(self, data: str) -> None:
        """Processes incoming simulation commands."""
        try:
            request_dict = self.serializer.deserialize(data)
            if isinstance(request_dict, dict):
                request = RequestMessage(**request_dict)
            elif isinstance(request_dict, RequestMessage):
                request = request_dict
            else:
                raise ValueError("Invalid request data format")
            logger.debug(f"Deserialized RequestMessage: {request}")
            self.command_queue.put(request)
        except Exception as e:
            logger.error(f"Failed to deserialize RequestMessage: {e}")

    def handle_response(self, data: str) -> None:
        """Processes incoming simulation responses."""
        try:
            response_dict = self.serializer.deserialize(data)
            response = ResponseMessage(**response_dict)
            logger.debug(f"Deserialized ResponseMessage: {response}")
            self.response_queue.put(response)
        except Exception as e:
            logger.error(f"Failed to deserialize ResponseMessage: {e}")

    def get_command(self) -> Optional[RequestMessage]:
        """Retrieves the latest simulation command."""
        try:
            request = self.command_queue.get_nowait()
            logger.debug(f"Retrieved RequestMessage from queue: {request}")
            return request
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error retrieving command: {e}")
            return None

    def get_response(self) -> Optional[ResponseMessage]:
        """Retrieves the latest simulation response."""
        try:
            response = self.response_queue.get_nowait()
            logger.debug(f"Retrieved ResponseMessage from queue: {response}")
            return response
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error retrieving response: {e}")
            return None

    def set_data(self, key: str, data: str) -> None:
        """Stores data in Redis with the specified key."""
        try:
            logger.info(f"Storing data in Redis with key: {key}")
            self.redis_client.set(key, data)
            logger.debug(f"Data stored successfully for key: {key}")
        except redis.RedisError as e:
            logger.error(f"Failed to set data in Redis for key '{key}': {e}")

    def get_data(self, key: str) -> Optional[Any]:
        """Retrieves data from Redis using the specified key."""
        try:
            logger.info(f"Retrieving data from Redis with key: {key}")
            data = self.redis_client.get(key)
            logger.debug(f"Data retrieved for key '{key}': {data}")
            return data
        except redis.RedisError as e:
            logger.error(f"Failed to get data from Redis for key '{key}': {e}")
            return None

    def publish_data(self, channel: str, data: str) -> None:
        """Publishes data to the specified Redis channel."""
        try:
            logger.info(f"Publishing data to Redis channel: {channel}")
            self.redis_client.publish(channel, data)
            logger.debug(f"Data published successfully to Redis channel: {channel}")
        except redis.RedisError as e:
            logger.error(f"Failed to publish data to Redis channel '{channel}': {e}")

    def serialize_data(
        self, data: Dict[str, Any], method: str = "json"
    ) -> Optional[str]:
        """Serializes data using the specified serializer."""
        try:
            logger.info("Serializing data.")
            serialized = self.serializer.serialize(data)
            if serialized:
                logger.debug("Data serialized successfully.")
                return serialized
            logger.error("Serialization returned None.")
            return None
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            return None

    def deserialize_data(self, data_str: str, method: str = "json") -> Dict[str, Any]:
        """Deserializes data using the specified serializer."""
        try:
            logger.info("Deserializing data.")
            data = self.serializer.deserialize(data_str)
            logger.debug("Data deserialized successfully.")
            return data
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise

    def send_response(self, response: ResponseMessage) -> None:
        """Sends a ResponseMessage back to the 'simulation_responses' channel.

        Parameters
        ----------
        response : ResponseMessage
            The response message to send.

        """
        try:
            response_dict = response.__dict__
            serialized = self.serializer.serialize(response_dict)
            if serialized:
                self.publish_data("simulation_responses", serialized)
                logger.debug(f"Response sent successfully: {response}")
            else:
                logger.error("Failed to serialize ResponseMessage; response not sent.")
        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    def close(self) -> None:
        """Gracefully stops the listener thread and closes Redis connections."""
        logger.info("Closing RedisClient.")
        self.stop_event.set()
        if self.pubsub:
            self.pubsub.unsubscribe("simulation_commands")
            self.pubsub.unsubscribe("simulation_responses")
            self.pubsub.close()
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
            logger.info("Listener thread terminated.")
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis client connection closed.")
