import logging
import csv
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)


@dataclass
class Material:
    name: str  # Material name (required)
    young_modulus: float  # Young's modulus in Pascals (Pa)
    poisson_ratio: float  # Poisson's ratio (dimensionless)
    density: float  # Density in kg/m^3
    color: Tuple[int, int, int]  # RGB color representation
    hardness: Optional[float] = None  # Optional hardness property, if relevant

    def to_dict(self) -> Dict:
        """Converts the Material instance to a dictionary for serialization."""
        return {
            "name": self.name,
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio,
            "density": self.density,
            "color": self.color,
            "hardness": self.hardness,
        }


# Predefined materials dictionary
materials: Dict[str, Material] = {}


def load_materials_from_csv(file_path: str):
    """
    Load materials from a CSV file into the materials dictionary.

    Args:
        file_path (str): The path to the CSV file.

    Raises:
        SimulationError: If there is an error reading the CSV file.
    """
    try:
        with open(file_path, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name'].upper()
                young_modulus = float(row['young_modulus'])
                poisson_ratio = float(row['poisson_ratio'])
                density = float(row['density'])
                color = (int(row['color_r']), int(row['color_g']), int(row['color_b']))
                hardness = float(row['hardness']) if row['hardness'] else None

                materials[name] = Material(
                    name=name,
                    young_modulus=young_modulus,
                    poisson_ratio=poisson_ratio,
                    density=density,
                    color=color,
                    hardness=hardness,
                )
        logger.info(SimulationLogMessageCode.CONFIGURATION_LOADED.details(f"Materials loaded from {file_path}"))
    except Exception as e:
        logger.error(SimulationLogMessageCode.CONFIGURATION_FAILED.details(f"Failed to load materials from {file_path}: {e}"))
        raise SimulationError(SimulationErrorCode.FILE_IO, f"Failed to load materials from {file_path}", details=str(e))


def add_custom_material(
    name: str,
    young_modulus: float,
    poisson_ratio: float,
    density: float,
    color: Tuple[int, int, int],
    hardness: Optional[float] = None,
    overwrite: bool = False,
):
    """
    Add a custom material to the materials dictionary.

    Args:
        name (str): The name of the material.
        young_modulus (float): Young's modulus in Pascals (Pa).
        poisson_ratio (float): Poisson's ratio (dimensionless).
        density (float): Density in kg/m^3.
        color (Tuple[int, int, int]): RGB color representation.
        hardness (Optional[float]): Optional hardness property, if relevant.
        overwrite (bool): Whether to overwrite an existing material with the same name.

    Raises:
        ValueError: If the color values are invalid or the material already exists and overwrite is False.
    """
    name_upper = name.upper()

    if any(c < 0 or c > 255 for c in color):
        logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Invalid color {color}. RGB values must be between 0 and 255."))
        raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, f"Invalid color {color}. RGB values must be between 0 and 255.")

    if name_upper in materials and not overwrite:
        logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Material '{name_upper}' already exists. Use overwrite=True to replace it."))
        raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, f"Material '{name_upper}' already exists. Use overwrite=True to replace it.")

    materials[name_upper] = Material(
        name=name_upper,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        color=color,
        hardness=hardness,
    )
    logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Material '{name_upper}' has been {'updated' if overwrite else 'added'}."))


def initial_material(
    name="default",
    young_modulus=1e6,
    poisson_ratio=0.45,
    density=1000.0,
    color=(128, 128, 128),
):
    """
    Initialize a material with the given properties.

    Args:
        name (str): The name of the material.
        young_modulus (float): Young's modulus in Pascals (Pa).
        poisson_ratio (float): Poisson's ratio (dimensionless).
        density (float): Density in kg/m^3.
        color (Tuple[int, int, int]): RGB color representation.

    Returns:
        Material: The initialized material.

    Raises:
        SimulationError: If the material cannot be initialized.
    """
    name = name.upper()
    try:
        if name in materials:
            material = materials[name]
        else:
            # Add custom material if it doesn't exist
            add_custom_material(name, young_modulus, poisson_ratio, density, color)
            material = materials[name]
        logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(
            f"Material '{name}' initialized with properties: Young's Modulus = {material.young_modulus}, Poisson's Ratio = {material.poisson_ratio}, Density = {material.density}, Color = {material.color}"
        ))
        return material
    except Exception as e:
        logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to initialize material '{name}': {e}"))
        raise SimulationError(SimulationErrorCode.COMMAND_PROCESSING, f"Failed to initialize material '{name}'", details=str(e))


# Load materials from CSV file
load_materials_from_csv('materials.csv')