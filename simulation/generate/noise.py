from abc import ABC, abstractmethod
import numpy as np
from noise import pnoise3

class NoiseFunction(ABC):
    @abstractmethod
    def generate(self, x, y):
        """Generate noise values for the given x and y coordinates."""
        pass
