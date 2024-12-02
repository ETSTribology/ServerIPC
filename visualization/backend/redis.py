import logging
from typing import Any, Callable, Dict

import redis

from visualization.backend.backend import Backend

logger = logging.getLogger(__name__)


class RedisBackend(Backend):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RedisBackend with configuration.

        Args:
            config: A dictionary containing the Redis configuration.
        """
        self.config = config
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
            )
            self.pubsub = self.client.pubsub()
            self.connected = True
            logger.info("Connected to Redis backend with Pub/Sub enabled.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def disconnect(self):
        """
        Disconnect from the Redis server.
        """
        if self.pubsub:
            self.pubsub.close()
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from Redis backend.")

    def publish(self, channel: str, message: str) -> None:
        """
        Publish a message to a Redis channel.

        Args:
            channel: The name of the Redis channel.
            message: The message to be published.
        """
        try:
            self.client.publish(channel, message)
            logger.info(f"Published message to channel '{channel}'.")
        except Exception as e:
            logger.error(f"Failed to publish message to channel '{channel}': {e}")
            raise

    def subscribe(self, channel: str, callback: Callable[[str, str], None]) -> None:
        """
        Subscribe to a Redis channel and process messages using a callback.

        Args:
            channel: The name of the Redis channel to subscribe to.
            callback: A callable that processes messages. It receives the message data.
        """
        try:

            def message_handler(message):
                if message["type"] == "message":
                    callback(message["channel"], message["data"].decode("utf-8"))

            self.pubsub.subscribe(**{channel: message_handler})
            logger.info(f"Subscribed to channel '{channel}'.")
        except Exception as e:
            logger.error(f"Failed to subscribe to channel '{channel}': {e}")
            raise

    def listen(self) -> None:
        """
        Start listening to subscribed channels.
        """
        if not self.pubsub:
            logger.error("Pub/Sub is not initialized. Call `connect` first.")
            return
        try:
            for message in self.pubsub.listen():
                if message["type"] == "message":
                    logger.info(
                        f"Received message on channel '{message['channel']}': {message['data'].decode('utf-8')}"
                    )
        except Exception as e:
            logger.error(f"Error while listening to Pub/Sub channels: {e}")
            raise

    def delete(self, channel: str) -> None:
        """
        Delete a Redis channel.

        Args:
            channel: The name of the Redis channel to delete.
        """
        try:
            self.client.delete(channel)
            logger.info(f"Deleted channel '{channel}'.")
        except Exception as e:
            logger.error(f"Failed to delete channel '{channel}': {e}")
            raise

        if self.pubsub:
            self.pubsub.unsubscribe(channel)

        if self.client:
            self.client.close()

        self.connected = False
        logger.info("Disconnected from Redis backend.")

    def read(self, key: str) -> Any:
        """
        Read data from the Redis pub/sub channel.

        Args:
            key: The key of the data to retrieve.

        Returns:
            The retrieved data.
        """
        return self.client.get(key)

    def write(self, key: str, value: Any) -> None:
        """
        Write data to the Redis pub/sub channel.

        Args:
            key: The key under which the data will be stored.
            value: The data to store.
        """
        self.client.set(key, value)
