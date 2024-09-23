from enum import Enum

class Material(Enum):
    WOOD = {
        'young_modulus': 10e9,   # Pascals (Pa)
        'poisson_ratio': 0.35,   # Dimensionless
        'density': 600           # kg/m^3
    }
    STEEL = {
        'young_modulus': 210e9,
        'poisson_ratio': 0.3,
        'density': 7850
    }
    ALUMINUM = {
        'young_modulus': 69e9,
        'poisson_ratio': 0.33,
        'density': 2700
    }
    CONCRETE = {
        'young_modulus': 30e9,
        'poisson_ratio': 0.2,
        'density': 2400
    }
    RUBBER = {
        'young_modulus': 0.01e9,
        'poisson_ratio': 0.48,
        'density': 1100
    }
    COPPER = {
        'young_modulus': 110e9,
        'poisson_ratio': 0.34,
        'density': 8960
    }
    GLASS = {
        'young_modulus': 50e9,
        'poisson_ratio': 0.22,
        'density': 2500
    }
    TITANIUM = {
        'young_modulus': 116e9,
        'poisson_ratio': 0.32,
        'density': 4500
    }
    BRASS = {
        'young_modulus': 100e9,
        'poisson_ratio': 0.34,
        'density': 8500
    }
    PLA = {
        'young_modulus': 4.4e9,
        'poisson_ratio': 0.3,
        'density': 1250
    }
    ABS = {
        'young_modulus': 2.3e9,
        'poisson_ratio': 0.35,
        'density': 1050
    }
    PETG = {
        'young_modulus': 2.59e9,
        'poisson_ratio': 0.4,
        'density': 1300
    }
    HYDROGEL = {
        'young_modulus': 1e6,
        'poisson_ratio': 0.35,
        'density': 1000
    }
    POLYACRYLAMIDE = {
        'young_modulus': 1e6,
        'poisson_ratio': 0.45,
        'density': 1050
    }
    DEFAULT = {
        'young_modulus': 1e6,
        'poisson_ratio': 0.35,
        'density': 1000
    }