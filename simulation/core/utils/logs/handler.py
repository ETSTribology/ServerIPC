import io
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from threading import Lock, Thread
from typing import Any, Dict

from core.registry.container import RegistryContainer
from core.registry.decorators import register
from core.utils.logs.message import *
from minio import Minio
from minio.error import S3Error
from nets.db.surreal import Surreal

logger = logging.getLogger(__name__)


class LogHandlerBase(logging.Handler, ABC):
    """Abstract base class for custom logging handlers.

    Ensures that all custom handlers implement the emit method.
    """

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    @abstractmethod
    def emit(self, record: logging.LogRecord):
        """Handle a single logging event.

        Must be implemented by all subclasses.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be handled.

        """
        pass


registry_container = RegistryContainer()
registry_container.add_registry("log_handler", "core.utils.logs.handler.LogHandlerBase")


@register(type="log_handler", name="surrealdb")
class SurrealDBHandler(LogHandlerBase):
    """Custom logging handler that sends log records to SurrealDB."""

    HANDLER_NAME = "SurrealDBHandler"

    def __init__(self, surrealdb_client: Surreal, table: str = "logs"):
        """Initializes the SurrealDBHandler.

        Parameters
        ----------
        - surrealdb_client: An instance of SurrealDBClient.
        - table: The SurrealDB table where logs will be stored.

        """
        super().__init__()
        self.surrealdb_client = surrealdb_client
        self.table = table
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            SURREALDB_HANDLER_INITIALIZED,
            self.table,
            extra={"handler_name": self.HANDLER_NAME},
        )

    def emit(self, record: logging.LogRecord):
        """Overrides the emit method to send log records to SurrealDB.

        Parameters
        ----------
        - record: The log record.

        """
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "funcName": record.funcName,
                "lineno": record.lineno,
                "threadName": record.threadName,
                "process": record.process,
                "run_number": getattr(record, "run_number", None),
            }
            self.surrealdb_client.create_record(self.table, log_entry)
            self.logger.debug(
                SURREALDB_EMIT_SUCCESS,
                log_entry,
                extra={"handler_name": self.HANDLER_NAME},
            )
        except Exception as e:
            self.handleError(record)
            self.logger.error(
                SURREALDB_EMIT_FAILURE, e, extra={"handler_name": self.HANDLER_NAME}
            )


@register(type="log_handler", name="minio")
class MinIOHandler(LogHandlerBase):
    """Custom logging handler that uploads log records to a MinIO bucket."""

    HANDLER_NAME = "MinIOHandler"

    def __init__(
        self,
        minio_client: Minio,
        bucket_name: str = "logs",
        object_name: str = "log.json",
        batch_size: int = 100,
        flush_interval: float = 10.0,
    ):
        """Initializes the MinIOHandler.

        Parameters
        ----------
        - minio_client: An instance of Minio client.
        - bucket_name: The MinIO bucket where logs will be stored.
        - object_name: The MinIO object name for storing logs.
        - batch_size: Number of log records to batch before uploading.
        - flush_interval: Time interval (in seconds) to flush logs.

        """
        super().__init__()
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.log_buffer = []
        self.lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            MINIO_HANDLER_INITIALIZED,
            self.bucket_name,
            self.object_name,
            extra={"handler_name": self.HANDLER_NAME},
        )

        # Ensure the bucket exists
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                self.logger.info(
                    MINIO_BUCKET_CREATED,
                    self.bucket_name,
                    extra={"handler_name": self.HANDLER_NAME},
                )
            else:
                self.logger.info(
                    MINIO_BUCKET_EXISTS,
                    self.bucket_name,
                    extra={"handler_name": self.HANDLER_NAME},
                )
        except S3Error as e:
            self.logger.error(
                "Failed to ensure bucket '%s' exists: %s",
                self.bucket_name,
                e,
                extra={"handler_name": self.HANDLER_NAME},
            )
            raise

        # Start a background thread for periodic flushing
        self._stop_event = False
        self.flush_thread = Thread(target=self._flush_periodically, daemon=True)
        self.flush_thread.start()

    def emit(self, record: logging.LogRecord):
        """Overrides the emit method to buffer log records.

        Parameters
        ----------
        - record: The log record.

        """
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "funcName": record.funcName,
                "lineno": record.lineno,
                "threadName": record.threadName,
                "process": record.process,
                "run_number": getattr(record, "run_number", None),
                # Add more fields as necessary
            }
            with self.lock:
                self.log_buffer.append(log_entry)
                if len(self.log_buffer) >= self.batch_size:
                    self._flush()
        except Exception as e:
            self.handleError(record)
            self.logger.error(
                "Failed to buffer log record to MinIO: %s",
                e,
                extra={"handler_name": self.HANDLER_NAME},
            )

    def _flush_periodically(self):
        """Periodically flushes the log buffer to MinIO."""
        while not self._stop_event:
            time.sleep(self.flush_interval)
            with self.lock:
                self._flush()

    def _flush(self):
        """Uploads the buffered log records to MinIO."""
        if not self.log_buffer:
            return
        try:
            # Retrieve existing logs if any
            existing_logs = []
            try:
                response = self.minio_client.get_object(
                    self.bucket_name, self.object_name
                )
                existing_logs = json.loads(response.read().decode("utf-8"))
                response.close()
                response.release_conn()
            except S3Error as e:
                if e.code != "NoSuchKey":
                    self.logger.error(
                        "Failed to retrieve existing logs from MinIO: %s",
                        e,
                        extra={"handler_name": self.HANDLER_NAME},
                    )
                    return

            # Append new logs
            existing_logs.extend(self.log_buffer)

            # Upload updated logs
            log_data = json.dumps(existing_logs, indent=2)
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=self.object_name,
                data=io.BytesIO(log_data.encode("utf-8")),
                length=len(log_data),
                content_type="application/json",
            )
            self.logger.debug(
                MINIO_FLUSH_UPLOAD,
                len(self.log_buffer),
                self.object_name,
                extra={"handler_name": self.HANDLER_NAME},
            )
            self.log_buffer.clear()
        except Exception as e:
            self.logger.error(
                MINIO_FLUSH_FAILURE, e, extra={"handler_name": self.HANDLER_NAME}
            )

    def close(self):
        """Flushes any remaining logs and stops the background thread."""
        self.acquire()
        try:
            self._stop_event = True
            self.flush_thread.join()
            with self.lock:
                self._flush()
            super().close()
        finally:
            self.release()


class HandlerFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the handler class based on the provided type."""
        registry_container = RegistryContainer()
        return registry_container.line_search.get(type_lower)

    def create(self, handler_config: Dict[str, Any]) -> logging.Handler:
        """Create a logging handler based on the provided configuration.

        Parameters
        ----------
        handler_config : Dict[str, Any]
            Configuration dictionary for the handler.

        Returns
        -------
        logging.Handler
            The created logging handler instance.

        Raises
        ------
        ValueError
            If the handler configuration is invalid or the handler creation fails.

        """
        handler_type = handler_config.get("type")
        if not handler_type:
            raise ValueError("Handler configuration must include a 'type' field.")

        handler_class = self.registry.get(handler_type.lower())

        # Extract parameters specific to the handler
        params = handler_config.get("params", {}).copy()  # Copy to avoid mutation

        try:
            if handler_type.lower() == "console":
                stream = params.get("stream", "ext://sys.stdout")
                if stream == "ext://sys.stdout":
                    stream_obj = sys.stdout
                elif stream == "ext://sys.stderr":
                    stream_obj = sys.stderr
                else:
                    raise ValueError(f"Unsupported stream: {stream}")
                handler = handler_class(stream=stream_obj)
            else:
                # For all other handlers, pass parameters directly
                handler = handler_class(**params)

            # Set handler level if specified
            level = handler_config.get("level")
            if level:
                handler.setLevel(level)

            # Set formatter if specified
            formatter = handler_config.get("formatter")
            if formatter:
                formatter_obj = logging.Formatter(formatter)
                handler.setFormatter(formatter_obj)

            logger.debug(
                HANDLER_FACTORY_CREATION_SUCCESS,
                handler_type,
                handler_config,
                extra={"handler_name": "HandlerFactory"},
            )
            return handler
        except Exception as e:
            logger.error(
                HANDLER_FACTORY_CREATION_FAILURE,
                handler_type,
                e,
                extra={"handler_name": "HandlerFactory"},
            )
            raise
