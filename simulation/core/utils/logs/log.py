import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from core.utils.logs.handler import HandlerFactory
from core.utils.logs.message import *

logger = logging.getLogger(__name__)


class LoggingManager:
    """Manages the logging configuration for the application.

    Initializes with default settings and allows applying a configuration dictionary to customize logging behavior.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **handler_clients: Any):
        """Initializes the LoggingManager.

        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            A dictionary containing logging configuration. If None, default logging settings are used.
        handler_clients : Any
            Handler-specific client instances (e.g., surrealdb_client, minio_client).

        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Default log level

        # Remove any existing handlers
        self._remove_existing_handlers()

        # Set up default logging (console handler)
        self._setup_default_logging()

        if config:
            self.apply_config(config, **handler_clients)

    def _remove_existing_handlers(self):
        """Removes all existing handlers from the root logger."""
        while self.logger.handlers:
            handler = self.logger.handlers[0]
            self.logger.removeHandler(handler)
            handler.close()
            logger.debug(
                LOGGING_MANAGER_REMOVE_HANDLER, extra={"handler_name": "LoggingManager"}
            )

    def _setup_default_logging(self):
        """Sets up default logging with a console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        logger.debug(
            LOGGING_MANAGER_SETUP_DEFAULT, extra={"handler_name": "LoggingManager"}
        )

    def apply_config(self, config: Dict[str, Any], **handler_clients: Any):
        """Applies a logging configuration from a dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            The logging configuration dictionary.
        handler_clients : Any
            Handler-specific client instances.

        """
        handler_configs = config.get("handlers", {})
        formatter_configs = config.get("formatters", {})
        log_level = config.get("level", "INFO")

        # Update root logger level
        self.logger.setLevel(log_level)
        logger.debug(
            LOGGING_MANAGER_SET_LEVEL,
            log_level,
            extra={"handler_name": "LoggingManager"},
        )

        # Set up formatters
        formatters = {}
        for fmt_name, fmt_props in formatter_configs.items():
            fmt_format = fmt_props.get(
                "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            formatters[fmt_name] = logging.Formatter(fmt_format)
            logger.debug(
                LOGGING_MANAGER_SETUP_FORMATTER,
                fmt_name,
                extra={"handler_name": "LoggingManager"},
            )

        # Initialize the HandlerFactory
        factory = HandlerFactory()

        # Iterate through handler configurations and create handlers
        for handler_name, handler_props in handler_configs.items():
            try:
                handler_type = handler_props.get("type")
                if not handler_type:
                    logger.warning(
                        "Handler '%s' does not have a 'type' specified. Skipping.",
                        handler_name,
                        extra={"handler_name": "LoggingManager"},
                    )
                    continue

                # Prepare parameters for the handler
                params = handler_props.get(
                    "params", {}
                ).copy()  # Copy to avoid mutation

                # Inject handler-specific clients if required
                if handler_type.lower() == "surrealdb":
                    surrealdb_client = handler_clients.get("surrealdb_client")
                    if not surrealdb_client:
                        logger.warning(
                            "Handler '%s' of type 'surrealdb' requires 'surrealdb_client'. Skipping.",
                            handler_name,
                            extra={"handler_name": "LoggingManager"},
                        )
                        continue
                    params["surrealdb_client"] = surrealdb_client
                elif handler_type.lower() == "minio":
                    minio_client = handler_clients.get("minio_client")
                    if not minio_client:
                        logger.warning(
                            "Handler '%s' of type 'minio' requires 'minio_client'. Skipping.",
                            handler_name,
                            extra={"handler_name": "LoggingManager"},
                        )
                        continue

                handler = factory.create(
                    type=handler_type,
                    name=handler_name,
                    level=handler_props.get("level", "INFO"),
                    formatter=formatters.get(handler_props.get("formatter", "default")),
                    **params,
                )

                # Add the handler to the root logger
                self.logger.addHandler(handler)
                logger.debug(
                    LOGGING_MANAGER_ADD_HANDLER,
                    handler_name,
                    handler_type,
                    extra={"handler_name": "LoggingManager"},
                )
            except Exception as e:
                logger.error(
                    LOGGING_MANAGER_FAILED_CONFIGURE_HANDLER,
                    handler_name,
                    e,
                    extra={"handler_name": "LoggingManager"},
                )

    @staticmethod
    def load_config_file(file_path: str) -> Dict[str, Any]:
        """Loads a logging configuration from a YAML or JSON file.

        Parameters
        ----------
        file_path : str
            Path to the configuration file.

        Returns
        -------
        Dict[str, Any]
            The logging configuration dictionary.

        Raises
        ------
        ValueError
            If the file extension is not supported or parsing fails.

        """
        if not Path(file_path).is_file():
            raise ValueError(f"Configuration file does not exist: {file_path}")

        try:
            with open(file_path, "r") as f:
                if file_path.endswith((".yaml", ".yml")):
                    config = yaml.safe_load(f)
                elif file_path.endswith(".json"):
                    config = json.load(f)
                else:
                    raise ValueError(
                        "Unsupported configuration file format. Use YAML or JSON."
                    )
            logger.debug(
                LOGGING_MANAGER_LOADED_CONFIG,
                file_path,
                extra={"handler_name": "LoggingManager"},
            )
            return config
        except Exception as e:
            raise ValueError(
                f"Failed to load logging configuration from {file_path}: {e}"
            )
