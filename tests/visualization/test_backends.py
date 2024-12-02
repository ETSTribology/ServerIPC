"""
Unit tests for communication backends.
"""

import pytest
import time
from unittest.mock import Mock, patch

from visualization.backends.registry import CommunicationBackendRegistry
from visualization.backends.base import BaseCommunicationBackend
from visualization.backends.redis_backend import RedisCommunicationBackend
from visualization.backends.websocket_backend import WebSocketCommunicationBackend
from visualization.backends.grpc_backend import GRPCCommunicationBackend


class TestCommunicationBackendRegistry:
    def test_backend_registration(self):
        """Test backend registration and retrieval."""
        # Create a mock backend
        class MockBackend(BaseCommunicationBackend):
            def connect(self, config): pass
            def disconnect(self): pass
            def send_message(self, topic, message): pass
            def subscribe(self, topic, callback): pass
            def unsubscribe(self, topic): pass

        # Register the mock backend
        CommunicationBackendRegistry.register('mock', MockBackend)
        
        # Verify registration
        assert 'mock' in CommunicationBackendRegistry.list_backends()
        retrieved_backend = CommunicationBackendRegistry.get('mock')
        assert retrieved_backend == MockBackend

    def test_backend_creation(self):
        """Test backend instance creation."""
        # Use Redis backend for testing
        config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
        
        backend = CommunicationBackendRegistry.create('redis', config)
        
        assert isinstance(backend, RedisCommunicationBackend)

    def test_nonexistent_backend(self):
        """Test handling of nonexistent backend."""
        with pytest.raises(KeyError, match="Backend 'nonexistent' not registered"):
            CommunicationBackendRegistry.get('nonexistent')


class TestRedisCommunicationBackend:
    @pytest.fixture
    def redis_backend(self):
        """Create a Redis backend for testing."""
        backend = RedisCommunicationBackend()
        backend.connect({'host': 'localhost', 'port': 6379})
        yield backend
        backend.disconnect()

    def test_redis_connection(self, redis_backend):
        """Test Redis backend connection."""
        assert redis_backend._redis_client is not None
        assert redis_backend._pubsub is not None

    def test_message_sending_and_receiving(self, redis_backend):
        """Test message publishing and subscribing."""
        # Mock callback
        callback = Mock()
        
        # Subscribe to a test topic
        redis_backend.subscribe('test_topic', callback)
        
        # Send a message
        test_message = {'key': 'value'}
        redis_backend.send_message('test_topic', test_message)
        
        # Wait for message processing
        time.sleep(0.5)
        
        # Verify callback was called
        callback.assert_called_once_with(test_message)

    def test_unsubscribe(self, redis_backend):
        """Test unsubscribing from a topic."""
        callback = Mock()
        redis_backend.subscribe('test_topic', callback)
        redis_backend.unsubscribe('test_topic')
        
        # Verify the subscription is removed
        assert 'test_topic' not in redis_backend._subscriptions


class TestWebSocketCommunicationBackend:
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket backend connection."""
        backend = WebSocketCommunicationBackend()
        
        # Mock configuration
        config = {
            'host': 'localhost',
            'port': 8765
        }
        
        # Patch websockets.connect to avoid actual network connection
        with patch('websockets.connect') as mock_connect:
            mock_websocket = Mock()
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            # Connect
            backend.connect(config)
            
            # Verify connection
            assert backend._websocket is not None
            assert backend._uri == 'ws://localhost:8765'

    @pytest.mark.asyncio
    async def test_websocket_messaging(self):
        """Test WebSocket message sending and receiving."""
        backend = WebSocketCommunicationBackend()
        
        # Mock configuration and websocket
        config = {'host': 'localhost', 'port': 8765}
        
        # Mock callback
        callback = Mock()
        
        # Patch websockets to simulate connection and message receiving
        with patch('websockets.connect') as mock_connect, \
             patch('websockets.WebSocketClientProtocol.recv', return_value='{"topic": "test", "payload": {"key": "value"}}') as mock_recv:
            
            mock_websocket = Mock()
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            # Connect and subscribe
            backend.connect(config)
            backend.subscribe('test', callback)
            
            # Simulate message receipt
            await backend._receive_messages()
            
            # Verify callback
            callback.assert_called_once_with({'key': 'value'})


class TestGRPCCommunicationBackend:
    def test_grpc_backend_initialization(self):
        """Test gRPC backend initialization."""
        backend = GRPCCommunicationBackend()
        
        config = {
            'host': 'localhost',
            'port': 50051
        }
        
        # Connect
        backend.connect(config)
        
        # Verify basic initialization
        assert backend._channel is not None
        assert backend._server is not None

    def test_grpc_backend_disconnection(self):
        """Test gRPC backend disconnection."""
        backend = GRPCCommunicationBackend()
        
        config = {
            'host': 'localhost',
            'port': 50051
        }
        
        # Connect and then disconnect
        backend.connect(config)
        backend.disconnect()
        
        # Additional verification can be added based on specific implementation
