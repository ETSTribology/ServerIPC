import os
import pytest
import logging
import tempfile
import json
import threading
import time
from unittest.mock import MagicMock, patch

from simulation.core.utils.logs.log import LoggingManager
from simulation.core.utils.logs.handler import (
    LogHandlerBase, 
    HandlerFactory, 
    DatabaseLogHandler, 
    MinIOHandler
)
from simulation.core.registry.container import RegistryContainer


class TestLoggingManager:
    def test_logging_manager_initialization(self):
        """Test basic initialization of LoggingManager."""
        logging_manager = LoggingManager()
        
        # Verify default logging level
        assert logging_manager.logger.level == logging.INFO
        
        # Verify at least one handler (console handler)
        assert len(logging_manager.logger.handlers) > 0

    def test_logging_manager_config_application(self):
        """Test applying a custom logging configuration."""
        config = {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                }
            }
        }

        logging_manager = LoggingManager(config)
        
        # Verify logging level
        assert logging_manager.logger.level == logging.DEBUG

    def test_remove_existing_handlers(self):
        """Test removal of existing handlers."""
        logging_manager = LoggingManager()
        
        # Add a dummy handler
        dummy_handler = logging.StreamHandler()
        logging_manager.logger.addHandler(dummy_handler)
        
        # Trigger handler removal
        logging_manager._remove_existing_handlers()
        
        # Verify no handlers remain
        assert len(logging_manager.logger.handlers) == 1  # Default console handler


class TestLogHandlerBase:
    def test_abstract_base_class(self):
        """Test that LogHandlerBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LogHandlerBase()

    def test_custom_handler_implementation(self):
        """Test creating a custom log handler by subclassing LogHandlerBase."""
        class CustomHandler(LogHandlerBase):
            def __init__(self):
                super().__init__()
                self.log_records = []

            def emit(self, record):
                self.log_records.append(record)

        handler = CustomHandler()
        
        # Create a log record
        logger = logging.getLogger('test_logger')
        record = logger.makeRecord(
            'test_logger', logging.INFO, None, None, 'Test message', None, None
        )
        
        # Emit the record
        handler.emit(record)
        
        # Verify record was captured
        assert len(handler.log_records) == 1
        assert handler.log_records[0].getMessage() == 'Test message'


class TestHandlerFactory:
    def test_handler_factory_registration(self):
        """Test handler registration and retrieval."""
        registry = RegistryContainer()
        factory = HandlerFactory()

        # Verify console handler can be created
        console_handler_config = {
            "type": "console",
            "level": "DEBUG",
            "formatter": "standard",
            "params": {"stream": "ext://sys.stdout"}
        }

        handler = factory.create(console_handler_config)
        assert isinstance(handler, logging.StreamHandler)

    def test_handler_factory_invalid_type(self):
        """Test creating a handler with an invalid type."""
        factory = HandlerFactory()
        
        with pytest.raises(ValueError):
            factory.create({"type": "non_existent_handler"})


class TestCustomLogHandlers:
    @pytest.mark.skipif(
        "SKIP_EXTERNAL_TESTS" in os.environ, 
        reason="Skipping tests requiring external services"
    )
    def test_database_handler_initialization(self):
        """Test DatabaseLogHandler initialization (requires mock database client)."""
        mock_database_client = MagicMock()
        
        handler = DatabaseLogHandler(mock_database_client)
        
        # Verify basic handler properties
        assert isinstance(handler, LogHandlerBase)
        assert handler.level == logging.NOTSET

    @pytest.mark.skipif(
        "SKIP_EXTERNAL_TESTS" in os.environ, 
        reason="Skipping tests requiring external services"
    )
    def test_minio_handler_initialization(self):
        """Test MinIOHandler initialization (requires mock MinIO client)."""
        mock_minio_client = MagicMock()
        mock_minio_client.bucket_exists.return_value = True
        
        handler = MinIOHandler(
            minio_client=mock_minio_client, 
            bucket_name="test-logs", 
            object_name="test-log.json"
        )
        
        # Verify basic handler properties
        assert isinstance(handler, LogHandlerBase)
        assert handler.bucket_name == "test-logs"
        assert handler.object_name == "test-log.json"
