"""Redis communication backend."""

import json
import redis
from typing import Any, Dict, Callable

from .base import BaseCommunicationBackend
from .registry import CommunicationBackendRegistry


class RedisCommunicationBackend(BaseCommunicationBackend):
    """Redis-based communication backend."""

    def __init__(self):
        """Initialize Redis backend."""
        self._redis_client = None
        self._pubsub = None
        self._subscriptions = {}

    def connect(self, config: Dict[str, Any]) -> None:
        """
        Establish Redis connection.
        
        Args:
            config: Redis connection configuration
        """
        try:
            self._redis_client = redis.Redis(
                host=config.get('host', 'localhost'),
                port=config.get('port', 6379),
                db=config.get('db', 0)
            )
            self._pubsub = self._redis_client.pubsub()
        except Exception as e:
            raise ConnectionError(f"Redis connection failed: {e}")

    def disconnect(self) -> None:
        """Close Redis connection."""
        if self._pubsub:
            self._pubsub.close()
        if self._redis_client:
            self._redis_client.close()

    def send_message(self, topic: str, message: Any) -> None:
        """
        Publish a message to a Redis channel.
        
        Args:
            topic: Channel name
            message: Message to publish
        """
        if not self._redis_client:
            raise RuntimeError("Redis client not connected")
        
        serialized_message = json.dumps(message)
        self._redis_client.publish(topic, serialized_message)

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to a Redis channel.
        
        Args:
            topic: Channel to subscribe to
            callback: Function to handle received messages
        """
        if not self._pubsub:
            raise RuntimeError("Redis pubsub not initialized")
        
        def _message_handler(message):
            try:
                decoded_message = json.loads(message['data'].decode('utf-8'))
                callback(decoded_message)
            except json.JSONDecodeError:
                print(f"Failed to decode message from {topic}")

        self._pubsub.subscribe(**{topic: _message_handler})
        self._subscriptions[topic] = _message_handler
        
        # Start listening in a separate thread
        self._pubsub.run_in_thread(sleep_time=0.1)

    def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a Redis channel.
        
        Args:
            topic: Channel to unsubscribe from
        """
        if not self._pubsub:
            return
        
        if topic in self._subscriptions:
            self._pubsub.unsubscribe(topic)
            del self._subscriptions[topic]


# Register the Redis backend with the registry
CommunicationBackendRegistry.register('redis', RedisCommunicationBackend)
