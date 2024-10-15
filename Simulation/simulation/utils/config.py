import json
import yaml
import logging
import sys
import os

logger = logging.getLogger(__name__)


def generate_default_config(args):
    return {
            "name": "Default Configuration",
            "inputs": [
                {
                    "path": args.input or "meshes/rectangle.mesh",
                    "percent_fixed": args.percent_fixed or 0.0,
                    "material": {
                        "density": args.mass_density or 1000.0,
                        "young_modulus": args.young_modulus or 1e6,
                        "poisson_ratio": args.poisson_ratio or 0.45,
                        "color": [255, 255, 255, 1]
                    },
                    "transform": {
                        "scale": [1.0, 1.0, 1.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "translation": [0.0, 0.0, 0.0]
                    },
                    "force": {
                        "gravity": 9.81,
                        "top_force": 10,
                        "side_force": 0
                    }
                }
            ],
            'friction': {
                'friction_coefficient': 0.3,
                'damping_coefficient': 1e-4
            },
            'simulation': {
                'dhat': 1e-3,
                'dmin': 1e-4,
                'dt': 1/60
            },
            'server': {
                'redis_host': args.redis_host or 'localhost',
                'redis_port': args.redis_port or 6379,
                'redis_db': args.redis_db or 0
            },
            "initial_conditions": {
                "gravity": 9.81,
            }
        }



def load_config(config_path: str, default_dict: dict = None):
    if config_path is None or not os.path.exists(config_path):
        logger.warning(f"Config path not found or not provided. Using default config.")
        return default_dict
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as file:
                config = json.load(file)
            logger.info(f"JSON configuration loaded successfully from {config_path}.")
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"YAML configuration loaded successfully from {config_path}.")
        else:
            logger.error(f"Unsupported configuration file format: {config_path}")
            return default_dict
        return config
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        logger.error(f"Failed to decode configuration file: {e}")
        # Return default instead of exiting
        return default_dict

def get_config_value(config, key, default=None):
    try:
        keys = key.split(".")
        value = config
        for k in keys:
            value = value[k]
        return value
    except KeyError:
        return default