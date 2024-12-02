"""
Minio-integrated communication backend for visualization.

Provides a comprehensive backend that supports:
- Minio object storage
- Redis pub/sub messaging
- gRPC communication
"""

import json
import logging
from typing import Any, Dict, Callable, Optional

import redis
import grpc
import minio
from minio import Minio
from minio.error import S3Error

from .base import BaseCommunicationBackend
from .registry import CommunicationBackendRegistry


class MinioIntegratedBackend(BaseCommunicationBackend):
    """
    Integrated communication backend with Minio, Redis, and gRPC support.
    
    Provides a comprehensive solution for distributed visualization
    with object storage, messaging, and RPC capabilities.
    """

    def __init__(self):
        """Initialize backend components."""
        self._redis_client = None
        self._minio_client = None
        self._grpc_channel = None
        self._logger = logging.getLogger(__name__)
        
        # Tracking for subscriptions
        self._subscriptions = {}
        self._pubsub = None

    def connect(self, config: Dict[str, Any]) -> None:
        """
        Establish connections to Minio, Redis, and gRPC.
        
        Args:
            config: Configuration dictionary with connection details
        """
        try:
            # Minio Connection
            self._minio_client = Minio(
                endpoint=config.get('minio_endpoint', 'localhost:9000'),
                access_key=config.get('minio_access_key', 'minioadmin'),
                secret_key=config.get('minio_secret_key', 'minioadmin'),
                secure=config.get('minio_secure', False)
            )
            
            # Ensure visualization bucket exists
            bucket_name = config.get('minio_bucket', 'visualization')
            if not self._minio_client.bucket_exists(bucket_name):
                self._minio_client.make_bucket(bucket_name)
            
            # Redis Connection
            self._redis_client = redis.Redis(
                host=config.get('redis_host', 'localhost'),
                port=config.get('redis_port', 6379),
                db=config.get('redis_db', 0)
            )
            self._pubsub = self._redis_client.pubsub()
            
            # gRPC Channel
            target = f"{config.get('grpc_host', 'localhost')}:{config.get('grpc_port', 50051)}"
            self._grpc_channel = grpc.insecure_channel(target)
            
            self._logger.info("Successfully connected to Minio, Redis, and gRPC")
        
        except (S3Error, redis.RedisError, grpc.RpcError) as e:
            self._logger.error(f"Connection error: {e}")
            raise ConnectionError(f"Failed to establish backend connections: {e}")

    def disconnect(self) -> None:
        """Close all backend connections."""
        try:
            # Close Redis pubsub
            if self._pubsub:
                self._pubsub.close()
            
            # Close Redis connection
            if self._redis_client:
                self._redis_client.close()
            
            # Close gRPC channel
            if self._grpc_channel:
                self._grpc_channel.close()
            
            self._logger.info("Backend connections closed successfully")
        
        except Exception as e:
            self._logger.error(f"Error during disconnection: {e}")

    def upload_to_minio(
        self, 
        bucket: str, 
        object_name: str, 
        data: bytes, 
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload data to Minio object storage.
        
        Args:
            bucket: Minio bucket name
            object_name: Name of the object to store
            data: Byte data to upload
            content_type: Optional content type
        
        Returns:
            Object URL
        """
        try:
            # Ensure bucket exists
            if not self._minio_client.bucket_exists(bucket):
                self._minio_client.make_bucket(bucket)
            
            # Upload data
            self._minio_client.put_object(
                bucket, 
                object_name, 
                data, 
                length=len(data),
                content_type=content_type
            )
            
            # Generate URL (optional, depends on Minio configuration)
            url = self._minio_client.presigned_get_object(bucket, object_name)
            
            self._logger.info(f"Uploaded {object_name} to {bucket}")
            return url
        
        except S3Error as e:
            self._logger.error(f"Minio upload error: {e}")
            raise

    def send_message(self, topic: str, message: Any) -> None:
        """
        Send a message via Redis pub/sub.
        
        Args:
            topic: Message channel/topic
            message: Payload to send
        """
        try:
            serialized_message = json.dumps(message)
            self._redis_client.publish(topic, serialized_message)
            self._logger.debug(f"Message sent to topic {topic}")
        except Exception as e:
            self._logger.error(f"Message sending error: {e}")

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to a Redis pub/sub topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Function to handle received messages
        """
        def _message_handler(message):
            try:
                # Decode and parse message
                data = message['data']
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                
                parsed_message = json.loads(data)
                callback(parsed_message)
            except json.JSONDecodeError:
                self._logger.warning(f"Invalid JSON in message from {topic}")
        
        # Subscribe and store handler
        self._pubsub.subscribe(**{topic: _message_handler})
        self._subscriptions[topic] = _message_handler
        
        # Start listening in a separate thread
        self._pubsub.run_in_thread(sleep_time=0.1)

    def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
        """
        if topic in self._subscriptions:
            self._pubsub.unsubscribe(topic)
            del self._subscriptions[topic]


# Register the Minio-integrated backend
CommunicationBackendRegistry.register('minio', MinioIntegratedBackend)
