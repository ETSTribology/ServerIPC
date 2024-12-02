"""
Unit tests for Redis client in visualization module.
"""

import pytest
import redis
from unittest.mock import Mock, patch

from visualization.backends.redis_backend import RedisCommunicationBackend as RedisClient


class TestRedisClient:
    @pytest.fixture
    def mock_redis_connection(self):
        """Create a mock Redis connection."""
        with patch('redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            yield mock_instance

    def test_redis_client_initialization(self, mock_redis_connection):
        """Test Redis client initialization."""
        client = RedisClient(
            host='localhost', 
            port=6379, 
            db=0
        )
        
        assert client is not None
        assert hasattr(client, '_client')
        assert isinstance(client._client, Mock)

    def test_publish_message(self, mock_redis_connection):
        """Test publishing a message to a channel."""
        client = RedisClient(
            host='localhost', 
            port=6379, 
            db=0
        )
        
        test_channel = 'test_channel'
        test_message = {'key': 'value'}
        
        client.publish(test_channel, test_message)
        
        mock_redis_connection.publish.assert_called_once_with(
            test_channel, 
            str(test_message)
        )

    def test_subscribe(self, mock_redis_connection):
        """Test subscribing to a channel."""
        client = RedisClient(
            host='localhost', 
            port=6379, 
            db=0
        )
        
        test_channel = 'test_channel'
        mock_callback = Mock()
        
        client.subscribe(test_channel, mock_callback)
        
        # Verify subscription logic
        mock_redis_connection.pubsub.assert_called_once()
        
    def test_unsubscribe(self, mock_redis_connection):
        """Test unsubscribing from a channel."""
        client = RedisClient(
            host='localhost', 
            port=6379, 
            db=0
        )
        
        test_channel = 'test_channel'
        
        client.unsubscribe(test_channel)
        
        # Verify unsubscription logic
        mock_redis_connection.pubsub().unsubscribe.assert_called_once_with(test_channel)

    def test_connection_error(self):
        """Test handling of Redis connection errors."""
        with patch('redis.Redis', side_effect=redis.ConnectionError):
            with pytest.raises(ConnectionError):
                RedisClient(
                    host='invalid_host', 
                    port=6379, 
                    db=0
                )

    def test_close_connection(self, mock_redis_connection):
        """Test closing Redis connection."""
        client = RedisClient(
            host='localhost', 
            port=6379, 
            db=0
        )
        
        client.close()
        
        mock_redis_connection.close.assert_called_once()
