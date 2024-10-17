from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class Material:
    name: str  # Material name (required)
    young_modulus: float  # Young's modulus in Pascals (Pa)
    poisson_ratio: float  # Poisson's ratio (dimensionless)
    density: float        # Density in kg/m^3
    color: Tuple[int, int, int]  # RGB color representation
    hardness: Optional[float] = None  # Optional hardness property, if relevant

    def to_dict(self) -> Dict:
        """
        Converts the Material instance to a dictionary for serialization.
        """
        return {
            'name': self.name,
            'young_modulus': self.young_modulus,
            'poisson_ratio': self.poisson_ratio,
            'density': self.density,
            'color': self.color,
            'hardness': self.hardness
        }

# Predefined materials dictionary
materials: Dict[str, Material] = {
    "WOOD": Material(name="WOOD", young_modulus=10e9, poisson_ratio=0.35, density=600, color=(139, 69, 19)),
    "STEEL": Material(name="STEEL", young_modulus=210e9, poisson_ratio=0.3, density=7850, color=(192, 192, 192)),
    "ALUMINUM": Material(name="ALUMINUM", young_modulus=69e9, poisson_ratio=0.33, density=2700, color=(211, 211, 211)),
    "CONCRETE": Material(name="CONCRETE", young_modulus=30e9, poisson_ratio=0.2, density=2400, color=(128, 128, 128)),
    "RUBBER": Material(name="RUBBER", young_modulus=0.01e9, poisson_ratio=0.48, density=1100, color=(0, 0, 0)),
    "COPPER": Material(name="COPPER", young_modulus=110e9, poisson_ratio=0.34, density=8960, color=(135, 206, 235)),
    "GLASS": Material(name="GLASS", young_modulus=50e9, poisson_ratio=0.22, density=2500, color=(135, 206, 235)),
    "TITANIUM": Material(name="TITANIUM", young_modulus=116e9, poisson_ratio=0.32, density=4500, color=(169, 169, 169)),
    "BRASS": Material(name="BRASS", young_modulus=100e9, poisson_ratio=0.34, density=8500, color=(218, 165, 32)),
    "PLA": Material(name="PLA", young_modulus=4.4e9, poisson_ratio=0.3, density=1250, color=(255, 228, 196)),
    "ABS": Material(name="ABS", young_modulus=2.3e9, poisson_ratio=0.35, density=1050, color=(0, 0, 0)),
    "PETG": Material(name="PETG", young_modulus=2.59e9, poisson_ratio=0.4, density=1300, color=(176, 196, 222)),
    "HYDROGEL": Material(name="HYDROGEL", young_modulus=1e6, poisson_ratio=0.35, density=1000, color=(173, 216, 230)),
    "POLYACRYLAMIDE": Material(name="POLYACRYLAMIDE", young_modulus=1e6, poisson_ratio=0.45, density=1050, color=(250, 250, 210)),
    "CAST_IRON": Material(name="CAST_IRON", young_modulus=170e9, poisson_ratio=0.26, density=7200, color=(128, 128, 128)),
    "CADMIUM": Material(name="CADMIUM", young_modulus=64e9, poisson_ratio=0.31, density=8650, color=(184, 184, 208)),
    "CHROMIUM": Material(name="CHROMIUM", young_modulus=248e9, poisson_ratio=0.31, density=7190, color=(180, 180, 180)),
    "GRAPHITE": Material(name="GRAPHITE", young_modulus=20e9, poisson_ratio=0.2, density=2050, color=(0, 0, 0)),
    "NICKEL": Material(name="NICKEL", young_modulus=170e9, poisson_ratio=0.31, density=8900, color=(175, 175, 175)),
    "NYLON": Material(name="NYLON", young_modulus=4e9, poisson_ratio=0.39, density=1130, color=(255, 255, 255)),
    "PLEXIGLASS": Material(name="PLEXIGLASS", young_modulus=3.3e9, poisson_ratio=0.37, density=1190, color=(255, 255, 255)),
    "POLYSTYRENE": Material(name="POLYSTYRENE", young_modulus=2.5e9, poisson_ratio=0.4, density=1040, color=(255, 255, 255)),
    "ASPHALT": Material(name="ASPHALT", young_modulus=3e9, poisson_ratio=0.35, density=2500, color=(0, 0, 0)),
    "TEFLON": Material(name="TEFLON", young_modulus=0.5e9, poisson_ratio=0.47, density=2200, color=(255, 255, 255)),
    "ZINC": Material(name="ZINC", young_modulus=82.7e9, poisson_ratio=0.25, density=7120, color=(192, 192, 192)),
    "PINE_WOOD": Material(name="PINE_WOOD", young_modulus=11e9, poisson_ratio=0.35, density=500, color=(139, 69, 19)),
    "STAINLESS_STEEL_17_4_PH": Material(name="STAINLESS_STEEL_17_4_PH", young_modulus=280e9, poisson_ratio=0.3, density=7800, color=(192, 192, 192)),
    "ABS_FDM": Material(name="ABS_FDM", young_modulus=1.65e9, poisson_ratio=0.35, density=1050, color=(0, 0, 0)),
    "FORMLABS_TOUGH_2000_RESIN_SLA": Material(name="FORMLABS_TOUGH_2000_RESIN_SLA", young_modulus=2.2e9, poisson_ratio=0.3, density=1200, color=(100, 100, 100)),
    "FORMLABS_NYLON_12_SLS": Material(name="FORMLABS_NYLON_12_SLS", young_modulus=1.85e9, poisson_ratio=0.4, density=970, color=(176, 196, 222)),
}

def add_custom_material(
    name: str,
    young_modulus: float,
    poisson_ratio: float,
    density: float,
    color: Tuple[int, int, int],
    hardness: Optional[float] = None,
    overwrite: bool = False
):
    name_upper = name.upper()

    if any(c < 0 or c > 255 for c in color):
        raise ValueError(f"Invalid color {color}. RGB values must be between 0 and 255.")

    if name_upper in materials and not overwrite:
        raise ValueError(f"Material '{name_upper}' already exists. Use overwrite=True to replace it.")

    materials[name_upper] = Material(
        name=name_upper,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        color=color,
        hardness=hardness
    )
    print(f"Material '{name_upper}' has been {'updated' if overwrite else 'added'}.")