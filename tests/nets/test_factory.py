import pytest

from simulation.nets.factory import NetsFactory
from simulation.nets.live.grpc import GRPC
from simulation.nets.live.redis import Redis
from simulation.nets.live.websocket import WebSocket


class TestNetsFactory:
    def test_create_redis_client(self):
        """Test creating a Redis client."""
        redis_client = NetsFactory.create_client("redis", host="localhost", port=6379, db=0)

        assert isinstance(redis_client, Redis)
        assert redis_client.host == "localhost"
        assert redis_client.port == 6379
        assert redis_client.db == 0

    def test_create_websocket_client(self):
        """Test creating a WebSocket client."""
        websocket_client = NetsFactory.create_client(
            "websocket", uri="ws://localhost:8080", serializer_method="json"
        )

        assert isinstance(websocket_client, WebSocket)
        assert websocket_client.uri == "ws://localhost:8080"
        assert websocket_client.serializer_method == "json"

    def test_create_grpc_client(self):
        """Test creating a gRPC client."""
        grpc_client = NetsFactory.create_client(
            "grpc", host="localhost", port=50051, serializer_method="protobuf"
        )

        assert isinstance(grpc_client, GRPC)
        assert grpc_client.host == "localhost"
        assert grpc_client.port == 50051
        assert grpc_client.serializer_method == "protobuf"

    def test_create_client_invalid_method(self):
        """Test creating a client with an invalid method."""
        with pytest.raises(ValueError, match="Unsupported communication method"):
            NetsFactory.create_client("invalid_method")

    def test_create_client_default_parameters(self):
        """Test creating clients with default parameters."""
        # Redis
        redis_client = NetsFactory.create_client("redis")
        assert redis_client.host == "localhost"
        assert redis_client.port == 6379
        assert redis_client.db == 0

        # WebSocket
        websocket_client = NetsFactory.create_client("websocket", uri="ws://test")
        assert websocket_client.serializer_method == "pickle"

        # gRPC
        grpc_client = NetsFactory.create_client("grpc")
        assert grpc_client.host == "localhost"
        assert grpc_client.port == 50051
        assert grpc_client.serializer_method == "pickle"
