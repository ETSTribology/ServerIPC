from enum import Enum

class Material(Enum):
    WOOD = {
        'young_modulus': 10e9,   # Pascals (Pa)
        'poisson_ratio': 0.35,   # Dimensionless
        'density': 600,          # kg/m^3
        'color': (139, 69, 19)  # RGB color
    }
    STEEL = {
        'young_modulus': 210e9,
        'poisson_ratio': 0.3,
        'density': 7850,
        'color': (192, 192, 192)
    }
    ALUMINUM = {
        'young_modulus': 69e9,
        'poisson_ratio': 0.33,
        'density': 2700,
        'color': (211, 211, 211)
    }
    CONCRETE = {
        'young_modulus': 30e9,
        'poisson_ratio': 0.2,
        'density': 2400,
        'color': (128, 128, 128)
    }
    RUBBER = {
        'young_modulus': 0.01e9,
        'poisson_ratio': 0.48,
        'density': 1100,
        'color': (0, 0, 0)
    }
    COPPER = {
        'young_modulus': 110e9,
        'poisson_ratio': 0.34,
        'density': 8960,
        'color': (135, 206, 235)
    }
    GLASS = {
        'young_modulus': 50e9,
        'poisson_ratio': 0.22,
        'density': 2500,
        'color': (135, 206, 235)
    }
    TITANIUM = {
        'young_modulus': 116e9,
        'poisson_ratio': 0.32,
        'density': 4500,
        'color': (169, 169, 169)
    }
    BRASS = {
        'young_modulus': 100e9,
        'poisson_ratio': 0.34,
        'density': 8500,
        'color': (218, 165, 32)
    }
    PLA = {
        'young_modulus': 4.4e9,
        'poisson_ratio': 0.3,
        'density': 1250,
        'color': (255, 228, 196)
    }
    ABS = {
        'young_modulus': 2.3e9,
        'poisson_ratio': 0.35,
        'density': 1050,
        'color': (0, 0, 0)
    }
    PETG = {
        'young_modulus': 2.59e9,
        'poisson_ratio': 0.4,
        'density': 1300,
        'color': (176, 196, 222)
    }
    HYDROGEL = {
        'young_modulus': 1e6,
        'poisson_ratio': 0.35,
        'density': 1000,
        'color': (173, 216, 230)
    }
    POLYACRYLAMIDE = {
        'young_modulus': 1e6,
        'poisson_ratio': 0.45,
        'density': 1050,
        'color': (250, 250, 210)
    }
    CAST_IRON = {
        'young_modulus': 170e9,
        'poisson_ratio': 0.26,
        'density': 7200,
        'color': (128, 128, 128)
    }
    CADMIUM = {
        'young_modulus': 64e9,
        'poisson_ratio': 0.31,
        'density': 8650,
        'color': (184, 184, 208)
    }
    CHROMIUM = {
        'young_modulus': 248e9,
        'poisson_ratio': 0.31,
        'density': 7190,
        'color': (180, 180, 180)
    }
    GRAPHITE = {
        'young_modulus': 20e9,
        'poisson_ratio': 0.2,
        'density': 2050,
        'color': (0, 0, 0)
    }
    NICKEL = {
        'young_modulus': 170e9,
        'poisson_ratio': 0.31,
        'density': 8900,
        'color': (175, 175, 175)
    }
    NYLON = {
        'young_modulus': 4e9,
        'poisson_ratio': 0.39,
        'density': 1130,
        'color': (255, 255, 255)
    }
    PLEXIGLASS = {
        'young_modulus': 3.3e9,
        'poisson_ratio': 0.37,
        'density': 1190,
        'color': (255, 255, 255)
    }
    POLYSTYRENE = {
        'young_modulus': 2.5e9,
        'poisson_ratio': 0.4,
        'density': 1040,
        'color': (255, 255, 255)
    }
    ASPHALT = {
        'young_modulus': 3e9,
        'poisson_ratio': 0.35,
        'density': 2500,
        'color': (0, 0, 0)
    }
    TEFLON = {
        'young_modulus': 0.5e9,
        'poisson_ratio': 0.47,
        'density': 2200,
        'color': (255, 255, 255)
    }
    ZINC = {
        'young_modulus': 82.7e9,
        'poisson_ratio': 0.25,
        'density': 7120,
        'color': (192, 192, 192)
    }
    DEFAULT = {
        'young_modulus': 1e6,
        'poisson_ratio': 0.35,
        'density': 1000,
        'color': (255, 255, 255)
    }

    @classmethod
    def add_custom_material(cls, name: str, young_modulus: float, poisson_ratio: float, density: float, color: tuple, hardness: float):
        name = name.upper()
        if name in cls.__members__:
            raise ValueError(f"Material '{name}' already exists.")
        new_material = {
            'young_modulus': young_modulus,
            'poisson_ratio': poisson_ratio,
            'density': density,
            'color': color,
            'hardness': hardness
        }
        cls._member_map_[name] = cls(name, new_material)
        cls._value2member_map_[new_material] = cls(name, new_material)
