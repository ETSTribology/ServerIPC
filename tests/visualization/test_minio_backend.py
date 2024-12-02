"""
Comprehensive unit tests for Minio-integrated communication backend.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from minio.error import S3Error

from visualization.backends.minio_backend import MinioIntegratedBackend


class TestMinioIntegratedBackend:
    @pytest.fixture
    def backend_config(self):
        """Provide a standard test configuration."""
        return {
            "minio_endpoint": "localhost:9000",
            "minio_access_key": "minioadmin",
            "minio_secret_key": "minioadmin",
            "minio_secure": False,
            "minio_bucket": "test-bucket",
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            "grpc_host": "localhost",
            "grpc_port": 50051,
        }

    @pytest.fixture
    def minio_backend(self, backend_config):
        """Create a MinioIntegratedBackend instance for testing."""
        backend = MinioIntegratedBackend()
        backend.connect(backend_config)
        yield backend
        backend.disconnect()

    def test_backend_connection(self, minio_backend, backend_config):
        """Test successful backend connection."""
        assert minio_backend._minio_client is not None
        assert minio_backend._redis_client is not None
        assert minio_backend._grpc_channel is not None

    def test_minio_upload(self, minio_backend, backend_config):
        """Test Minio object upload."""
        # Prepare test data
        test_data = b"Test upload content"
        bucket = backend_config["minio_bucket"]
        object_name = "test_upload.txt"

        # Upload to Minio
        url = minio_backend.upload_to_minio(
            bucket, object_name, test_data, content_type="text/plain"
        )

        # Verify upload
        assert url is not None
        assert isinstance(url, str)

    def test_minio_upload_error(self, minio_backend):
        """Test Minio upload error handling."""
        with pytest.raises(S3Error):
            # Attempt upload with invalid bucket name
            minio_backend.upload_to_minio("!@#$%^&*", "invalid_object", b"Test data")

    def test_redis_messaging(self, minio_backend):
        """Test Redis pub/sub messaging."""
        # Prepare test message and topic
        test_topic = "test_channel"
        test_message = {"key": "value"}

        # Mock callback
        mock_callback = Mock()

        # Subscribe to topic
        minio_backend.subscribe(test_topic, mock_callback)

        # Send message
        minio_backend.send_message(test_topic, test_message)

        # Wait for message processing
        import time

        time.sleep(0.5)

        # Verify callback was called
        mock_callback.assert_called_once_with(test_message)

    def test_unsubscribe(self, minio_backend):
        """Test topic unsubscription."""
        # Prepare test topic and callback
        test_topic = "test_unsubscribe"
        mock_callback = Mock()

        # Subscribe
        minio_backend.subscribe(test_topic, mock_callback)

        # Unsubscribe
        minio_backend.unsubscribe(test_topic)

        # Verify unsubscription
        assert test_topic not in minio_backend._subscriptions

    @patch("minio.Minio.make_bucket")
    @patch("minio.Minio.bucket_exists", return_value=False)
    def test_minio_bucket_creation(self, mock_bucket_exists, mock_make_bucket, backend_config):
        """Test automatic bucket creation."""
        backend = MinioIntegratedBackend()
        backend.connect(backend_config)

        # Verify bucket creation was attempted
        mock_make_bucket.assert_called_once_with(backend_config["minio_bucket"])

    def test_connection_error_handling(self):
        """Test connection error handling."""
        backend = MinioIntegratedBackend()

        # Provide invalid configuration
        invalid_config = {
            "minio_endpoint": "invalid_endpoint",
            "minio_access_key": "invalid_key",
            "minio_secret_key": "invalid_secret",
        }

        # Expect connection error
        with pytest.raises(ConnectionError):
            backend.connect(invalid_config)

    def test_disconnection(self, minio_backend):
        """Test backend disconnection."""
        try:
            minio_backend.disconnect()
        except Exception as e:
            pytest.fail(f"Disconnection raised an unexpected error: {e}")

    def test_logging(self, minio_backend, caplog):
        """Test logging functionality."""
        # Send a message to trigger logging
        minio_backend.send_message("log_test", {"test": "message"})

        # Check log records
        assert any("Message sent to topic" in record.message for record in caplog.records)

    def test_minio_presigned_url(self, minio_backend, backend_config):
        """Test generation of presigned URL."""
        bucket = backend_config["minio_bucket"]
        object_name = "test_presigned.txt"

        # Upload a test object first
        test_data = b"Presigned URL test content"
        minio_backend.upload_to_minio(bucket, object_name, test_data)

        # Generate presigned URL
        presigned_url = minio_backend.get_presigned_url(bucket, object_name, expiry=3600)  # 1 hour

        assert presigned_url is not None
        assert isinstance(presigned_url, str)
        assert presigned_url.startswith("http")

    def test_grpc_communication(self, minio_backend):
        """Test basic gRPC communication."""
        # Mock gRPC stub and method
        mock_stub = MagicMock()
        mock_request = MagicMock()

        # Simulate gRPC call
        try:
            minio_backend.grpc_call(mock_stub, mock_request)
            mock_stub.assert_called_once_with(mock_request)
        except Exception as e:
            pytest.fail(f"gRPC communication test failed: {e}")

    def test_invalid_backend_configuration(self):
        """Test handling of incomplete backend configuration."""
        backend = MinioIntegratedBackend()

        # Provide partial configuration
        incomplete_config = {"minio_endpoint": "localhost:9000"}

        with pytest.raises(ValueError, match="Missing required configuration"):
            backend.connect(incomplete_config)
