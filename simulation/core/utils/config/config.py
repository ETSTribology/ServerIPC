import json
import logging
import os
from typing import Any, Dict, List, Tuple

import yaml
from core.utils.singleton import SingletonMeta

logger = logging.getLogger(__name__)


class ConfigManager(metaclass=SingletonMeta):
    """Singleton class for managing simulation configurations.
    Provides easy access, validation, and dynamic updates of configuration values.
    """

    def __init__(self):
        self.config = self.generate_default_config()

    @staticmethod
    def generate_default_config() -> Dict[str, Any]:
        """Generates the default configuration dictionary.

        Returns:
            dict: Default configuration dictionary.

        """
        return {
            "name": "Default Configuration",
            "inputs": [
                {
                    "path": "meshes/rectangle.mesh",
                    "percent_fixed": 0.0,
                    "material": "default",
                    "transform": {
                        "scale": [1.0, 1.0, 1.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "translation": [0.0, 0.0, 0.0],
                    },
                    "force": {
                        "gravity": 9.81,
                        "top_force": 10,
                        "side_force": 0,
                    },
                }
            ],
            "materials": [
                {
                    "name": "default",
                    "density": 1000.0,
                    "young_modulus": 1e6,
                    "poisson_ratio": 0.45,
                    "color": [255, 255, 255, 1],
                }
            ],
            "friction": {
                "friction_coefficient": 0.3,
                "damping_coefficient": 1e-4,
            },
            "simulation": {"dhat": 1e-3, "dmin": 1e-4, "dt": 1 / 60},
            "communication": {
                "method": "redis",
                "settings": {
                    "redis": {
                        "host": "localhost",
                        "port": 6379,
                        "db": 0,
                        "password": None,
                    }
                },
            },
            "serialization": {"default_method": "json"},
            "optimizer": {
                "type": "newton",
                "params": {
                    "max_iterations": 100,
                    "rtol": 1e-5,
                    "n_threads": 1,
                    "reg_param": 1e-4,
                    "m": 10,
                },
            },
            "linear_solver": {
                "type": "ldlt",
                "params": {"tolerance": 1e-6, "max_iterations": 1000},
            },
            "ipc": {"enabled": False, "method": "default", "params": {}},
            "initial_conditions": {"gravity": 9.81},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "DEBUG",
                        "formatter": "standard",
                        "stream": "ext://sys.stdout",
                    },
                    "file": {
                        "class": "logging.FileHandler",
                        "level": "INFO",
                        "formatter": "standard",
                        "filename": "simulation.log",
                    },
                },
            },
        }

    def load_config(self, config_path: str) -> None:
        """Loads configuration from a JSON or YAML file, falling back to default.

        Args:
            config_path (str): Path to the configuration file.

        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as file:
                    if config_path.endswith(".json"):
                        self.config = json.load(file)
                    elif config_path.endswith((".yaml", ".yml")):
                        self.config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from '{config_path}'.")
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            logger.warning("Using default configuration.")

    def get_param(self, key: str, default: Any = None) -> Any:
        """Retrieves a parameter value using a dotted key path.

        Args:
            key (str): Dotted key path (e.g., 'simulation.dt').
            default (Any): Default value if key not found.

        Returns:
            Any: Value of the parameter or default.

        """
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Parameter '{key}' not found. Returning default: {default}.")
            return default

    def set_param(self, key: str, value: Any) -> None:
        """Sets a parameter value using a dotted key path.

        Args:
            key (str): Dotted key path (e.g., 'simulation.dt').
            value (Any): Value to set.

        """
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        logger.info(f"Parameter '{key}' set to {value}.")

    def load_material_properties(self, input_entry: dict) -> dict:
        """Loads material properties from the input entry.

        Args:
            input_entry (dict): The input entry containing material details.

        Returns:
            dict: Material properties including density, Young's modulus, etc.

        """
        material_props = input_entry.get("material", {})
        return {
            "density": material_props.get("density", 1000.0),
            "young_modulus": material_props.get("young_modulus", 1e6),
            "poisson_ratio": material_props.get("poisson_ratio", 0.45),
            "color": material_props.get("color", [255, 255, 255, 1]),
        }

    def load_transform_properties(
        self, input_entry: dict
    ) -> Tuple[List[float], List[float], List[float]]:
        """Loads transformation properties (scale, rotation, translation) from the input entry.

        Args:
            input_entry (dict): Input entry containing transformation details.

        Returns:
            Tuple[List[float], List[float], List[float]]: Scale, rotation, and translation properties.

        """
        transform = input_entry.get("transform", {})
        return (
            transform.get("scale", [1.0, 1.0, 1.0]),
            transform.get("rotation", [0.0, 0.0, 0.0, 1.0]),
            transform.get("translation", [0.0, 0.0, 0.0]),
        )
