import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from jsonschema import Draft7Validator, RefResolver, ValidationError

from simulation.core.utils.singleton import SingletonMeta

CURRENT_DIR = Path(__file__).parent
SCHEMAS_DIR = Path(CURRENT_DIR, "schemas")


class ConfigManager(metaclass=SingletonMeta):
    """
    A singleton class for managing visualization configurations.
    """

    def __init__(self, config_path: str, schema_name: str = "main"):
        """
        Initialize the VisualizationConfigManager.

        Args:
            config_path: Path to the configuration file (JSON or YAML).
            schema_name: Name of the schema to load for validation.
        """
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.config_path = Path(config_path)
            self.schema = self.load_schema(schema_name)
            self._validator = Draft7Validator(
                self.schema,
                resolver=RefResolver(base_uri=f"file://{SCHEMAS_DIR}/", referrer=self.schema),
            )
            self._config = self.load_config()

    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Load a JSON schema from the schemas directory.

        Args:
            schema_name: Name of the schema file (without extension).

        Returns:
            A dictionary representing the loaded schema.

        Raises:
            FileNotFoundError: If the schema file is not found.
            ValueError: If the schema file contains invalid JSON.
        """
        schema_path = SCHEMAS_DIR / f"{schema_name}.json"
        try:
            with open(schema_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found at {schema_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file: {e}")

    def load_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load a configuration file (JSON or YAML).

        Args:
            filepath: Path to the configuration file.

        Returns:
            A dictionary representing the configuration.

        Raises:
            ValueError: If the file format is unsupported.
        """
        filepath_str = str(filepath)
        with open(filepath, "r") as f:
            if filepath_str.endswith(".json"):
                return json.load(f)
            elif filepath_str.endswith((".yaml", ".yml")):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format: only JSON and YAML are supported")

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration file and validate it.

        Returns:
            The loaded configuration as a dictionary.

        Raises:
            ValidationError: If the configuration fails validation.
        """
        config = self.load_file(self.config_path)
        self.validate(config)
        return config

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration against the schema.

        Args:
            config: The configuration dictionary to validate.

        Raises:
            ValidationError: If the configuration does not match the schema.
        """
        errors = sorted(self._validator.iter_errors(config), key=lambda e: e.path)
        if errors:
            error_messages = "\n".join(
                f"{'.'.join(map(str, error.path))}: {error.message}" for error in errors
            )
            raise ValidationError(
                f"Configuration validation failed against schema '{self.schema['title']}':\n{error_messages}"
            )

    def update(self, **kwargs) -> None:
        """
        Update specific configuration attributes and validate.

        Args:
            kwargs: Key-value pairs of configuration attributes to update.

        Raises:
            AttributeError: If a configuration attribute does not exist.
            ValidationError: If the updated configuration is invalid.
        """
        for key, value in kwargs.items():
            keys = key.split(".")
            target = self._config
            for k in keys[:-1]:
                target = target.get(k, {})
            if keys[-1] not in target:
                raise AttributeError(f"Invalid configuration attribute: {key}")
            target[keys[-1]] = value
        self.validate(self._config)

    def save(self, filepath: Optional[str] = None, file_format: str = "json") -> None:
        """
        Save the current configuration to a file.

        Args:
            filepath: Optional path to save configuration. Defaults to `self.config_path`.
            file_format: Format to save the file ('json' or 'yaml').

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        filepath = filepath or self.config_path
        with open(filepath, "w") as f:
            if file_format == "json":
                json.dump(self._config, f, indent=4)
            elif file_format in ("yaml", "yml"):
                yaml.safe_dump(self._config, f, default_flow_style=False)
            else:
                raise ValueError("Unsupported file format: only 'json' and 'yaml' are supported")

    def get(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            A dictionary containing the current configuration.
        """
        return self._config
