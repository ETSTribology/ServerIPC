import json
import logging
import os
from dataclasses import field
from typing import Any, Dict, List, Tuple, Optional, Union

import yaml
from simulation.core.utils.singleton import SingletonMeta

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)



class ConfigManager(metaclass=SingletonMeta):
    """Singleton class for managing simulation configurations.
    Provides easy access, validation, and dynamic updates of configuration values.
    """
    config: DictConfig = field(default_factory=lambda: OmegaConf.create())

    def __init__(self):
        self.config = self.generate_default_config()


    def initialize(self, config_name: str = "config.yaml", config_path: str = "./configs") -> None:
        """Initializes the ConfigManager with Hydra."""
        # Initialize Hydra only if it hasn't been initialized yet
        if not hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.initialize(config_path=config_path, job_name="simulation_app")

        # Compose the configuration
        self.config = hydra.compose(config_name=config_name)
        logger.info(f"Configuration loaded:\n{OmegaConf.to_yaml(self.config)}")

        # Compose the configuration
        self.config = hydra.compose(config_name=config_name)
        logger.info(f"Configuration loaded:\n{OmegaConf.to_yaml(self.config)}")

    def generate_default_config(self) -> Dict[str, Any]:
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
            "board": {
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

    def load_material_properties(self, material_name: str = "default") -> Dict[str, Any]:
        """
        Load material properties for a given material name.

        Args:
            material_name (str, optional): Name of the material. Defaults to "default".

        Returns:
            Dict[str, Any]: Material properties dictionary.
        """
        default_material = {
            "name": "default",
            "density": 1000.0,  # kg/m³
            "young_modulus": 1e6,  # Pa
            "poisson_ratio": 0.3,
            "damping_coefficient": 0.1
        }
        
        # In a real-world scenario, this could load from a configuration file or database
        return default_material

    def load_transform_properties(self, transform_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load transformation properties.

        Args:
            transform_config (Optional[Dict[str, Any]], optional): Transformation configuration. Defaults to None.

        Returns:
            Dict[str, Any]: Transformation properties.
        """
        default_transform = {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0]
        }
        
        if transform_config:
            default_transform.update(transform_config)
        
        return default_transform

    def load_material_properties(self, material_name: str = "default") -> Dict[str, Any]:
        """
        Load material properties from a configuration or default settings.
        
        Args:
            material_name (str): Name of the material to load properties for.
        
        Returns:
            Dict[str, Any]: Material properties dictionary.
        """
        default_properties = {
            "default": {
                "young_modulus": 1e6,  # 1 MPa
                "poisson_ratio": 0.3,
                "density": 1000.0,  # kg/m³
            },
            "steel": {
                "young_modulus": 200e9,  # 200 GPa
                "poisson_ratio": 0.3,
                "density": 7850.0,  # kg/m³
            }
        }
        
        return default_properties.get(material_name, default_properties["default"])

    def load_material_properties(self, material_name: str) -> Dict[str, Any]:
        """Load material properties for a given material name.

        Args:
            material_name (str): Name of the material.

        Returns:
            Dict[str, Any]: Material properties dictionary.

        Raises:
            ValueError: If material is not found.
        """
        # Default material properties
        default_materials = {
            "default": {
                "density": 1000.0,  # kg/m³
                "young_modulus": 1e6,  # Pa
                "poisson_ratio": 0.4,
                "damping_coefficient": 0.1,
            }
        }

        # Try to find the material
        material = default_materials.get(material_name.lower())
        
        if material is None:
            logger.warning(f"Material '{material_name}' not found. Using default material.")
            material = default_materials["default"]

        return material


    def load_transform_properties(self, input_entry: Dict[str, Any]) -> Tuple[Union[float, List[float]], List[float], List[float]]:
        """Load transformation properties from input configuration.

        Args:
            input_entry (Dict[str, Any]): Input configuration dictionary.

        Returns:
            Tuple[Union[float, List[float]], List[float], List[float]]: 
                - Scale factor (scalar or list)
                - Rotation angles (degrees)
                - Translation vector
        """
        # Default transformation values
        scale = input_entry.get("transform", {}).get("scale", 1.0)
        rotation = input_entry.get("transform", {}).get("rotation", [0.0, 0.0, 0.0, 1.0])  # Updated to match test
        translation = input_entry.get("transform", {}).get("translation", [0.0, 0.0, 0.0])

        # Ensure rotation is a list of 4 elements (quaternion)
        if len(rotation) != 4:
            rotation = [0.0, 0.0, 0.0, 1.0]

        logger.debug(f"Transform properties - Scale: {scale}, Rotation: {rotation}, Translation: {translation}")

        return scale, rotation, translation


    def load_simulation_parameters(self) -> Dict[str, Any]:
        """Load simulation parameters from configuration.

        Returns:
            Dict[str, Any]: Dictionary containing simulation parameters
                - dt: Time step size
                - total_time: Total simulation time
                - output_frequency: Frequency of output saves
                - gravity: Gravity vector [x, y, z]
        """
        sim_config = self.config.get("simulation", {})
        return {
            "dt": sim_config.get("dt", 1/60),
            "total_time": sim_config.get("total_time", 10.0),
            "output_frequency": sim_config.get("output_frequency", 60),
            "gravity": sim_config.get("gravity", [0.0, -9.81, 0.0]),
            "dhat": sim_config.get("dhat", 1e-3),
            "dmin": sim_config.get("dmin", 1e-4),
            "damping": sim_config.get("damping", 0.1),
        }

    def load_optimizer_settings(self) -> Dict[str, Any]:
        """Load optimizer settings from configuration.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer parameters
                - type: Optimizer type (newton, gradient_descent, etc.)
                - max_iterations: Maximum number of iterations
                - tolerance: Convergence tolerance
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
        """Load collision detection and handling settings.

        Returns:
            Dict[str, Any]: Dictionary containing collision parameters
                - method: Collision detection method
                - friction_coefficient: Coefficient of friction
                - restitution: Coefficient of restitution
                - contact_stiffness: Contact stiffness parameter
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
        """Load linear solver settings from configuration.

        Returns:
            Dict[str, Any]: Dictionary containing solver parameters
                - type: Solver type (direct, iterative)
                - tolerance: Solver tolerance
                - max_iterations: Maximum iterations for iterative solvers
                - preconditioner: Preconditioner settings
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
        """Load output and visualization settings.

        Returns:
            Dict[str, Any]: Dictionary containing output parameters
                - format: Output format (vtk, obj, etc.)
                - directory: Output directory path
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
                    "position": output_config.get("visualization", {}).get("camera", {}).get("position", [0, 0, 5]),
                    "target": output_config.get("visualization", {}).get("camera", {}).get("target", [0, 0, 0]),
                    "up": output_config.get("visualization", {}).get("camera", {}).get("up", [0, 1, 0]),
                },
            },
            "metrics": {
                "enabled": output_config.get("metrics", {}).get("enabled", True),
                "types": output_config.get("metrics", {}).get("types", ["timing", "memory", "convergence"]),
                "log_file": output_config.get("metrics", {}).get("log_file", "performance.log"),
            },
        }

    @property
    def board(self):
        """
        Access the board configuration.

        Returns:
            DictConfig: Board configuration section.
        """
        # If board configuration doesn't exist, create a default
        if 'board' not in self.config:
            self.config['board'] = {
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
                }
            }
        return self.config['board']

    # Standalone functions for material and transform properties
    def load_material_properties(material_name: str) -> Dict[str, Any]:
        """Load material properties for a given material name.

        Args:
            material_name (str): Name of the material.

        Returns:
            Dict[str, Any]: Material properties dictionary.

        Raises:
            ValueError: If material is not found.
        """
        # Default material properties
        default_materials = {
            "default": {
                "density": 1000.0,  # kg/m³
                "young_modulus": 1e6,  # Pa
                "poisson_ratio": 0.4,
                "damping_coefficient": 0.1,
            }
        }

        # Try to find the material
        material = default_materials.get(material_name.lower())
        
        if material is None:
            logger.warning(f"Material '{material_name}' not found. Using default material.")
            material = default_materials["default"]

        return material


    def load_transform_properties(input_entry: Dict[str, Any]) -> Tuple[Union[float, List[float]], List[float], List[float]]:
        """Load transformation properties from input configuration.

        Args:
            input_entry (Dict[str, Any]): Input configuration dictionary.

        Returns:
            Tuple[Union[float, List[float]], List[float], List[float]]: 
                - Scale factor (scalar or list)
                - Rotation angles (degrees)
                - Translation vector
        """
        # Default transformation values
        scale = input_entry.get("transform", {}).get("scale", 1.0)
        rotation = input_entry.get("transform", {}).get("rotation", [0.0, 0.0, 0.0, 1.0])  
        translation = input_entry.get("transform", {}).get("translation", [0.0, 0.0, 0.0])

        # Ensure rotation is a list of 4 elements (quaternion)
        if len(rotation) != 4:
            rotation = [0.0, 0.0, 0.0, 1.0]

        logger.debug(f"Transform properties - Scale: {scale}, Rotation: {rotation}, Translation: {translation}")

        return scale, rotation, translation


    @classmethod
    def quick_init(cls, config_dict: Optional[Dict[str, Any]] = None) -> 'ConfigManager':
        """Quick initialization method for testing purposes.
        
        Args:
            config_dict (Optional[Dict[str, Any]], optional): Custom configuration dictionary. 
                                                              Defaults to None.
        
        Returns:
            ConfigManager: Initialized configuration manager
        """
        # Reset the singleton instance
        if hasattr(cls, '_instance'):
            delattr(cls, '_instance')
        
        # Create a new instance
        instance = cls()
        
        # Use provided config or generate default
        if config_dict is not None:
            instance.config = OmegaConf.create(config_dict)
        else:
            instance.config = OmegaConf.create(instance.generate_default_config())
        
        return instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance for testing purposes."""
        if hasattr(cls, '_instance'):
            delattr(cls, '_instance')

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Get or create the singleton instance."""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
            cls._instance.config = OmegaConf.create(cls._instance.generate_default_config())
        return cls._instance

    def load_transform_properties(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load transformation properties from a configuration file.

        Parameters
        ----------
        config_path : Optional[str], optional
            Path to the configuration file containing transform properties.
            If not provided, uses the default configuration.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing transformation properties.
        """
        config_manager = self.get_instance()
        
        # If a specific config path is provided, try to load from that path
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    transform_config = yaml.safe_load(f)
                    return transform_config.get('transform_properties', {})
            except (FileNotFoundError, IOError) as e:
                logger.warning(f"Could not load transform properties from {config_path}: {e}")
        
        # Fallback to default configuration
        transform_properties = config_manager.config.get('transform_properties', {})
        
        return transform_properties

def load_transform_properties(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load transformation properties from a configuration file.

    Parameters
    ----------
    config_path : Optional[str], optional
        Path to the configuration file containing transform properties.
        If not provided, uses the default configuration.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing transformation properties.
    """
    config_manager = ConfigManager.get_instance()
    return config_manager.load_transform_properties(config_path)
