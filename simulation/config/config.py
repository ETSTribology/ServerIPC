import json
import logging
import os
from dataclasses import field, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from jsonschema import Draft7Validator, RefResolver, ValidationError
from omegaconf import DictConfig, OmegaConf

from simulation.core.utils.singleton import SingletonMeta

logger = logging.getLogger(__name__)




@dataclass
class ConfigManager(metaclass=SingletonMeta):
    """
    Singleton class for managing simulation configurations.
    Provides easy access, validation, and dynamic updates of configuration values.
    """

    config: DictConfig = field(default_factory=lambda: OmegaConf.create())

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            # Define schema directories
            self.current_dir = Path(__file__).parent
            self.schemas_dir = self.current_dir / "schemas"
            # Initialize configuration with defaults
            self.config = self.generate_default_config()
            # Initialize schema
            self.schema = self.load_schema("main")
            self.validator = Draft7Validator(
                self.schema,
                resolver=RefResolver(
                    base_uri=f"file://{self.schemas_dir.resolve()}/",
                    referrer=self.schema,
                ),
            )
            logger.debug("ConfigManager initialized with default configuration.")

    def initialize(self, config_name: str = "config.yaml", config_path: str = "./configs") -> None:
        """
        Initializes the ConfigManager by loading the configuration file.

        Args:
            config_name (str): Name of the configuration file.
            config_path (str): Path to the configuration directory.
        """
        full_path = Path(config_path) / config_name
        self.load_config(full_path)
        logger.info(f"Configuration loaded:\n{OmegaConf.to_yaml(self.config)}")

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
        schema_path = self.schemas_dir / f"{schema_name}.json"
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)
                logger.info(f"Schema '{schema_name}.json' loaded successfully.")
                return schema
        except FileNotFoundError:
            logger.error(f"Schema file not found at {schema_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file: {e}")
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
        filepath_path = Path(filepath)
        filepath_str = str(filepath_path)
        with open(filepath_path, "r") as f:
            if filepath_str.endswith(".json"):
                return json.load(f)
            elif filepath_str.endswith((".yaml", ".yml")):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format: only JSON and YAML are supported")

    def load_config(self, config_path: str) -> None:
        """
        Loads configuration from a YAML or JSON file, validating against the schema.

        Args:
            config_path (str): Path to the configuration file.
        """
        if config_path and Path(config_path).is_file():
            try:
                config_dict = self.load_file(config_path)
                self.validate(config_dict)
                self.config = OmegaConf.create(config_dict)
                logger.info(f"Configuration loaded from '{config_path}'.")
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                logger.error(f"Error loading configuration: {e}")
                logger.warning("Using default configuration due to loading error.")
            except ValidationError as ve:
                logger.error(f"Configuration validation error: {ve}")
                logger.warning("Using default configuration due to validation error.")
        else:
            if config_path:
                logger.error(f"Configuration file '{config_path}' does not exist.")
            logger.warning("Using default configuration.")

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration against the schema.

        Args:
            config: The configuration dictionary to validate.

        Raises:
            ValidationError: If the configuration does not match the schema.
        """
        errors = sorted(self.validator.iter_errors(config), key=lambda e: e.path)
        if errors:
            error_messages = "\n".join(
                f"{'.'.join(map(str, error.path))}: {error.message}" for error in errors
            )
            logger.error(
                f"Configuration validation failed against schema '{self.schema.get('title', 'Unknown')}':\n{error_messages}"
            )
            raise ValidationError(
                f"Configuration validation failed against schema '{self.schema.get('title', 'Unknown')}':\n{error_messages}"
            )
        logger.info("Configuration validation passed successfully.")

    def generate_default_config(self) -> DictConfig:
        """
        Generates the default configuration dictionary by combining multiple sub-configurations.

        Returns:
            DictConfig: Default configuration.
        """
        default_config = {
            "name": "Default Configuration",
            "inputs": self._generate_inputs_config(),
            "materials": self._generate_materials_config(),
            "friction": self._generate_friction_config(),
            "simulation": self._generate_simulation_config(),
            "communication": self._generate_communication_config(),
            "serialization": self._generate_serialization_config(),
            "optimizer": self._generate_optimizer_config(),
            "linear_solver": self._generate_linear_solver_config(),
            "ipc": self._generate_ipc_config(),
            "initial_conditions": self._generate_initial_conditions_config(),
            "logging": self._generate_logging_config(),
            "board": self._generate_board_config(),
            "output": self._generate_output_config(),
        }
        return OmegaConf.create(default_config)

    def _generate_inputs_config(self) -> List[Dict[str, Any]]:
        """
        Generates the default inputs configuration.

        Returns:
            List[Dict[str, Any]]: List of input configurations.
        """
        return [
            {
                "path": "meshes/rectangle.mesh",
                "percent_fixed": 0.0,
                "material": "default",
                "transform": self._generate_transform_config(),
                "force": self._generate_force_config(),
            }
        ]

    def _generate_materials_config(self) -> List[Dict[str, Any]]:
        """
        Generates the default materials configuration.

        Returns:
            List[Dict[str, Any]]: List of material configurations.
        """
        return [
            {
                "name": "default",
                "density": 1000.0,
                "young_modulus": 1e6,
                "poisson_ratio": 0.45,
                "color": [255, 255, 255, 1],
            }
        ]

    def _generate_friction_config(self) -> Dict[str, Any]:
        """
        Generates the default friction configuration.

        Returns:
            Dict[str, Any]: Friction configuration.
        """
        return {
            "friction_coefficient": 0.3,
            "damping_coefficient": 1e-4,
        }

    def _generate_simulation_config(self) -> Dict[str, Any]:
        """
        Generates the default simulation configuration.

        Returns:
            Dict[str, Any]: Simulation configuration.
        """
        return {
            "dhat": 1e-3,
            "dmin": 1e-4,
            "dt": 1 / 60,
            "total_time": 10.0,
            "output_frequency": 60,
            "gravity": [0.0, -9.81, 0.0],
            "damping": 0.1,
        }

    def _generate_communication_config(self) -> Dict[str, Any]:
        """
        Generates the default communication configuration.

        Returns:
            Dict[str, Any]: Communication configuration.
        """
        return {
            "method": "redis",
            "settings": {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "password": None,
                }
            },
        }

    def _generate_serialization_config(self) -> Dict[str, Any]:
        """
        Generates the default serialization configuration.

        Returns:
            Dict[str, Any]: Serialization configuration.
        """
        return {
            "default_method": "json",
        }

    def _generate_optimizer_config(self) -> Dict[str, Any]:
        """
        Generates the default optimizer configuration.

        Returns:
            Dict[str, Any]: Optimizer configuration.
        """
        return {
            "type": "newton",
            "params": {
                "max_iterations": 100,
                "rtol": 1e-5,
                "n_threads": 1,
                "reg_param": 1e-4,
                "m": 10,
            },
            "line_search": {
                "method": "armijo",
                "c": 0.5,
                "tau": 0.5,
                "max_iterations": 20,
            },
        }

    def _generate_linear_solver_config(self) -> Dict[str, Any]:
        """
        Generates the default linear solver configuration.

        Returns:
            Dict[str, Any]: Linear solver configuration.
        """
        return {
            "type": "ldlt",
            "params": {"tolerance": 1e-6, "max_iterations": 1000},
            "preconditioner": {
                "type": "diagonal",
                "omega": 0.5,
            },
            "direct_solver": {
                "type": "cholmod",
                "ordering": "amd",
            },
        }

    def _generate_ipc_config(self) -> Dict[str, Any]:
        """
        Generates the default IPC (Intersection Penalty Contact) configuration.

        Returns:
            Dict[str, Any]: IPC configuration.
        """
        return {
            "enabled": False,
            "method": "default",
            "params": {},
        }

    def _generate_initial_conditions_config(self) -> Dict[str, Any]:
        """
        Generates the default initial conditions configuration.

        Returns:
            Dict[str, Any]: Initial conditions configuration.
        """
        return {
            "gravity": 9.81,
        }

    def _generate_logging_config(self) -> Dict[str, Any]:
        """
        Generates the default logging configuration.

        Returns:
            Dict[str, Any]: Logging configuration.
        """
        return {
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
        }

    def _generate_board_config(self) -> Dict[str, Any]:
        """
        Generates the default board (e.g., TensorBoard) configuration.

        Returns:
            Dict[str, Any]: Board configuration.
        """
        return {
            "type": "tensorboard",  # Default board type
            "enabled": True,
            "log_level": "INFO",
            "extra_config": {
                "log_dir": "./runs",
                "tensor_data_subdir": "tensor_data",
                "comment": "",
                "purge_step": None,
                "max_queue": 10,
                "flush_secs": 120,
                "filename_suffix": "",
            },
        }

    def _generate_output_config(self) -> Dict[str, Any]:
        """
        Generates the default output and visualization configuration.

        Returns:
            Dict[str, Any]: Output configuration.
        """
        return {
            "format": "vtk",
            "directory": "results",
            "frequency": 60,
            "compression": True,
            "visualization": {
                "enabled": True,
                "backend": "opengl",
                "window_size": [1024, 768],
                "camera": {
                    "position": [0, 0, 5],
                    "target": [0, 0, 0],
                    "up": [0, 1, 0],
                },
            },
            "metrics": {
                "enabled": True,
                "types": ["timing", "memory", "convergence"],
                "log_file": "performance.log",
            },
        }

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a parameter value using a dotted key path.

        Args:
            key (str): Dotted key path (e.g., 'simulation.dt').
            default (Any): Default value if key not found.

        Returns:
            Any: Value of the parameter or default.
        """
        try:
            return self.config.select(key)
        except Exception:
            logger.debug(f"Parameter '{key}' not found. Returning default: {default}.")
            return default

    def set_param(self, key: str, value: Any) -> None:
        """
        Sets a parameter value using a dotted key path.

        Args:
            key (str): Dotted key path (e.g., 'simulation.dt').
            value (Any): Value to set.
        """
        try:
            OmegaConf.update(self.config, key, value, force=True)
            logger.info(f"Parameter '{key}' set to {value}.")
        except Exception as e:
            logger.error(f"Failed to set parameter '{key}': {e}")

    def load_material_properties(self, material_name: str = "default") -> Dict[str, Any]:
        """
        Load material properties for a given material name.

        Args:
            material_name (str): Name of the material to load properties for.

        Returns:
            Dict[str, Any]: Dictionary containing material properties.
        """
        default_material = {
            "density": 1000.0,
            "young_modulus": 1e6,
            "poisson_ratio": 0.45,
            "color": [255, 255, 255, 1],
        }

        materials = self.config.get("materials", [])
        for material in materials:
            if material.get("name") == material_name:
                return {
                    "density": material.get("density", default_material["density"]),
                    "young_modulus": material.get("young_modulus", default_material["young_modulus"]),
                    "poisson_ratio": material.get("poisson_ratio", default_material["poisson_ratio"]),
                    "color": material.get("color", default_material["color"]),
                }

        logger.warning(f"Material '{material_name}' not found. Using default material properties.")
        return default_material

    def load_transform_properties(
        self, input_entry: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[float, List[float]], List[float], List[float]]:
        """
        Load transformation properties from input entry or configuration.

        Args:
            input_entry (Optional[Dict[str, Any]]): Input entry containing transform properties.

        Returns:
            Tuple[Union[float, List[float]], List[float], List[float]]: Tuple containing scale, rotation, and translation.
        """
        default_transform = {
            "scale": [1.0, 1.0, 1.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }

        if not input_entry or "transform" not in input_entry:
            logger.warning("No transform configuration found. Using default transform properties.")
            return (
                default_transform["scale"],
                default_transform["rotation"],
                default_transform["translation"],
            )

        transform = input_entry["transform"]
        return (
            transform.get("scale", default_transform["scale"]),
            transform.get("rotation", default_transform["rotation"]),
            transform.get("translation", default_transform["translation"]),
        )

    def load_simulation_parameters(self) -> Dict[str, Any]:
        """
        Load simulation parameters from configuration.

        Returns:
            Dict[str, Any]: Dictionary containing simulation parameters
                - dt: Time step size
                - total_time: Total simulation time
                - output_frequency: Frequency of output saves
                - gravity: Gravity vector [x, y, z]
                - dhat: Parameter dhat
                - dmin: Parameter dmin
                - damping: Damping coefficient
        """
        sim_config = self.config.get("simulation", {})
        return {
            "dt": sim_config.get("dt", 1 / 60),
            "total_time": sim_config.get("total_time", 10.0),
            "output_frequency": sim_config.get("output_frequency", 60),
            "gravity": sim_config.get("gravity", [0.0, -9.81, 0.0]),
            "dhat": sim_config.get("dhat", 1e-3),
            "dmin": sim_config.get("dmin", 1e-4),
            "damping": sim_config.get("damping", 0.1),
        }

    def load_optimizer_settings(self) -> Dict[str, Any]:
        """
        Load optimizer settings from configuration.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer parameters
                - type: Optimizer type (newton, gradient_descent, etc.)
                - max_iterations: Maximum number of iterations
                - rtol: Relative tolerance
                - n_threads: Number of threads
                - reg_param: Regularization parameter
                - line_search: Line search settings
        """
        opt_config = self.config.get("optimizer", {})
        return {
            "type": opt_config.get("type", "newton"),
            "max_iterations": opt_config.get("params", {}).get("max_iterations", 100),
            "rtol": opt_config.get("params", {}).get("rtol", 1e-5),
            "n_threads": opt_config.get("params", {}).get("n_threads", 1),
            "reg_param": opt_config.get("params", {}).get("reg_param", 1e-4),
            "line_search": {
                "method": opt_config.get("line_search", {}).get("method", "armijo"),
                "c": opt_config.get("line_search", {}).get("c", 0.5),
                "tau": opt_config.get("line_search", {}).get("tau", 0.5),
                "max_iterations": opt_config.get("line_search", {}).get("max_iterations", 20),
            },
        }

    def load_collision_settings(self) -> Dict[str, Any]:
        """
        Load collision detection and handling settings.

        Returns:
            Dict[str, Any]: Dictionary containing collision parameters
                - method: Collision detection method
                - friction_coefficient: Coefficient of friction
                - restitution: Coefficient of restitution
                - contact_stiffness: Contact stiffness parameter
                - contact_parameters: Additional contact parameters
                - friction_parameters: Additional friction parameters
        """
        collision_config = self.config.get("collision", {})
        return {
            "method": collision_config.get("method", "ipc"),
            "friction_coefficient": collision_config.get("friction_coefficient", 0.3),
            "restitution": collision_config.get("restitution", 0.5),
            "contact_stiffness": collision_config.get("contact_stiffness", 1e4),
            "contact_parameters": {
                "dhat": collision_config.get("dhat", 1e-3),
                "dmin": collision_config.get("dmin", 1e-4),
            },
            "friction_parameters": {
                "mu": collision_config.get("friction", {}).get("mu", 0.3),
                "tau": collision_config.get("friction", {}).get("tau", 1e-4),
            },
        }

    def load_solver_settings(self) -> Dict[str, Any]:
        """
        Load linear solver settings from configuration.

        Returns:
            Dict[str, Any]: Dictionary containing solver parameters
                - type: Solver type (ldlt, direct, iterative)
                - tolerance: Solver tolerance
                - max_iterations: Maximum iterations for iterative solvers
                - preconditioner: Preconditioner settings
                - direct_solver: Direct solver settings
        """
        solver_config = self.config.get("linear_solver", {})
        return {
            "type": solver_config.get("type", "ldlt"),
            "tolerance": solver_config.get("params", {}).get("tolerance", 1e-6),
            "max_iterations": solver_config.get("params", {}).get("max_iterations", 1000),
            "preconditioner": {
                "type": solver_config.get("preconditioner", {}).get("type", "diagonal"),
                "omega": solver_config.get("preconditioner", {}).get("omega", 0.5),
            },
            "direct_solver": {
                "type": solver_config.get("direct_solver", {}).get("type", "cholmod"),
                "ordering": solver_config.get("direct_solver", {}).get("ordering", "amd"),
            },
        }

    def load_output_settings(self) -> Dict[str, Any]:
        """
        Load output and visualization settings.

        Returns:
            Dict[str, Any]: Dictionary containing output parameters
                - format: Output format (vtk, obj, etc.)
                - directory: Output directory path
                - frequency: Output frequency
                - compression: Compression flag
                - visualization: Visualization settings
                - metrics: Performance metrics to track
        """
        output_config = self.config.get("output", {})
        return {
            "format": output_config.get("format", "vtk"),
            "directory": output_config.get("directory", "results"),
            "frequency": output_config.get("frequency", 60),
            "compression": output_config.get("compression", True),
            "visualization": {
                "enabled": output_config.get("visualization", {}).get("enabled", True),
                "backend": output_config.get("visualization", {}).get("backend", "opengl"),
                "window_size": output_config.get("visualization", {}).get("window_size", [1024, 768]),
                "camera": {
                    "position": output_config.get("visualization", {})
                    .get("camera", {})
                    .get("position", [0, 0, 5]),
                    "target": output_config.get("visualization", {})
                    .get("camera", {})
                    .get("target", [0, 0, 0]),
                    "up": output_config.get("visualization", {})
                    .get("camera", {})
                    .get("up", [0, 1, 0]),
                },
            },
            "metrics": {
                "enabled": output_config.get("metrics", {}).get("enabled", True),
                "types": output_config.get("metrics", {}).get("types", ["timing", "memory", "convergence"]),
                "log_file": output_config.get("metrics", {}).get("log_file", "performance.log"),
            },
        }

    def _generate_board_config(self) -> Dict[str, Any]:
        """
        Generates the default board (e.g., TensorBoard) configuration.

        Returns:
            Dict[str, Any]: Board configuration.
        """
        return {
            "type": "tensorboard",  # Default board type
            "enabled": True,
            "log_level": "INFO",
            "extra_config": {
                "log_dir": "./runs",
                "tensor_data_subdir": "tensor_data",
                "comment": "",
                "purge_step": None,
                "max_queue": 10,
                "flush_secs": 120,
                "filename_suffix": "",
            },
        }

    @property
    def board(self) -> DictConfig:
        """
        Access the board configuration.

        Returns:
            DictConfig: Board configuration section.
        """
        # If board configuration doesn't exist, create a default
        if "board" not in self.config:
            self.config["board"] = self._generate_board_config()
            logger.warning("No board configuration found. Using default board properties.")
        return self.config["board"]

    @classmethod
    def quick_init(cls, config_dict: Optional[Dict[str, Any]] = None) -> "ConfigManager":
        """
        Quick initialization method for testing purposes.

        Args:
            config_dict (Optional[Dict[str, Any]], optional): Custom configuration dictionary.
                                                            Defaults to None.

        Returns:
            ConfigManager: Initialized configuration manager
        """
        # Reset the singleton instance
        cls.reset()

        # Get the instance
        instance = cls.get_instance()

        # Use provided config or generate default
        if config_dict is not None:
            instance.config = OmegaConf.create(config_dict)
            logger.info("ConfigManager initialized with custom configuration for testing.")
        else:
            instance.config = OmegaConf.create(instance.generate_default_config())
            logger.info("ConfigManager initialized with default configuration for testing.")

        return instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.
        """
        if hasattr(cls, "_instance"):
            delattr(cls, "_instance")
            logger.info("ConfigManager singleton instance has been reset.")

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """
        Get or create the singleton instance.

        Returns:
            ConfigManager: The singleton instance.
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
            cls._instance.config = OmegaConf.create(cls._instance.generate_default_config())
            logger.info("ConfigManager singleton instance created.")
        return cls._instance
