import asyncio
import threading
from unittest.mock import Mock, patch

import pytest

from simulation.nets.live.grpc import GRPC
from simulation.nets.serialization.factory import SerializerFactory


class TestGRPC:
    @pytest.fixture
    def grpc_client(self):
        """Create a GRPC client for testing."""
        client = GRPC(host="localhost", port=50051, serializer_method="pickle")
        yield client
        client.stop_event.set()  # Stop the listener thread

    def test_grpc_initialization(self, grpc_client):
        """Test GRPC client initialization."""
        assert grpc_client.host == "localhost"
        assert grpc_client.port == 50051
        assert grpc_client.serializer_method == "pickle"

        # Check serializer factory
        assert isinstance(grpc_client.serializer_factory, SerializerFactory)

        # Check thread and queue
        assert isinstance(grpc_client.listener_thread, threading.Thread)
        assert grpc_client.listener_thread.daemon
        assert not grpc_client.command_queue.empty()

    @pytest.mark.asyncio
    async def test_connect_async(self, grpc_client):
        """Test async connection method."""
        with patch("grpc.aio.insecure_channel") as mock_channel:
            mock_channel_instance = Mock()
            mock_channel.return_value = mock_channel_instance
            mock_channel_instance.channel_ready = asyncio.coroutine(lambda: None)

            await grpc_client.connect_async()

            mock_channel.assert_called_once_with("localhost:50051")

    def test_stop_listener(self, grpc_client):
        """Test stopping the listener thread."""
        grpc_client.stop_event.clear()
        grpc_client.stop_listener()

        assert grpc_client.stop_event.is_set()
        assert not grpc_client.listener_thread.is_alive()

    def test_serializer_selection(self, grpc_client):
        """Test serializer method selection."""
        serializer_methods = ["pickle", "json", "protobuf"]

        for method in serializer_methods:
            client = GRPC(serializer_method=method)
            serializer = client.serializer_factory.create(method)
            assert serializer is not None

    def test_invalid_serializer_method(self):
        """Test creating a GRPC client with an invalid serializer method."""
        with pytest.raises(ValueError, match="Unsupported serialization method"):
            GRPC(serializer_method="invalid_method")

    def test_command_queue_operations(self, grpc_client):
        """Test command queue operations."""
        test_command = {"action": "test"}

        # Put a command in the queue
        grpc_client.command_queue.put(test_command)

        # Retrieve the command
        retrieved_command = grpc_client.command_queue.get()

        assert retrieved_command == test_command
        assert grpc_client.command_queue.empty()
