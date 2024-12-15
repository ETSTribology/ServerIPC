from simulation.generate.noise import NoiseFunction
from noise import pnoise3
import numpy as np


from simulation.generate.noise import NoiseFunction
from noise import pnoise3
import numpy as np


class FractalBrownianMotion(NoiseFunction):
    def __init__(self, amplitude=1.0, frequency=1.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=42):
        self.amplitude = amplitude
        self.frequency = frequency
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate Fractal Brownian Motion noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of FBM noise values.
        """
        fbm = np.zeros_like(x, dtype=np.float64)
        freq = self.frequency
        amp = self.amplitude
        for _ in range(self.octaves):
            fbm += amp * np.vectorize(pnoise3)(x * freq, y * freq, 0, repeatx=1024, repeaty=1024, base=self.seed)
            amp *= self.persistence
            freq *= self.lacunarity
        return fbm


class PerlinNoise(NoiseFunction):
    def __init__(self, amplitude=1.0, frequency=1.0, seed=42):
        self.amplitude = amplitude
        self.frequency = frequency
        self.seed = seed

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate Perlin noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of Perlin noise values.
        """
        perlin = self.amplitude * np.vectorize(pnoise3)(x * self.frequency, y * self.frequency, 0, repeatx=1024, repeaty=1024, base=self.seed)
        return perlin


class SineWave(NoiseFunction):
    def __init__(self, amplitude=1.0, frequency=10.0):
        self.amplitude = amplitude
        self.frequency = frequency

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate a sine wave-based noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of sine wave values.
        """
        return self.amplitude * np.sin(self.frequency * x)


class SquareWave(NoiseFunction):
    def __init__(self, amplitude=1.0, frequency=10.0):
        self.amplitude = amplitude
        self.frequency = frequency

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate a square wave-based noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of square wave values.
        """
        return self.amplitude * np.sign(np.sin(self.frequency * x))


class BeckmannNoise(NoiseFunction):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate Beckmann microfacet distribution noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of Beckmann noise values.
        """
        tan_theta_h = np.sqrt(x**2 + y**2) / self.alpha
        cos_theta_h = 1 / np.sqrt(1 + tan_theta_h**2)
        D = np.exp(-tan_theta_h**2) / (np.pi * self.alpha**2 * cos_theta_h**4)
        return D


class GGXNoise(NoiseFunction):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate GGX (Trowbridge-Reitz) microfacet distribution noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of GGX noise values.
        """
        tan2_theta_h = (x**2 + y**2) / self.alpha**2
        cos_theta_h = 1 / np.sqrt(1 + tan2_theta_h)
        cos2_theta_h = cos_theta_h**2
        D = self.alpha**2 / (np.pi * ((cos2_theta_h * (self.alpha**2 - 1) + 1)**2))
        return D


class BlinnPhongNoise(NoiseFunction):
    def __init__(self, n=20):
        self.n = n

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate Blinn-Phong microfacet distribution noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of Blinn-Phong noise values.
        """
        tan_theta_h = np.sqrt(x**2 + y**2)
        cos_theta_h = 1 / np.sqrt(1 + tan_theta_h**2)
        D = (self.n + 2) / (2 * np.pi) * cos_theta_h**self.n
        return D


class MandelbrotNoise(NoiseFunction):
    def __init__(self, max_iter=100, amplitude=1.0):
        self.max_iter = max_iter
        self.amplitude = amplitude

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate Mandelbrot set-based noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of Mandelbrot noise values.
        """
        c = x + 1j * y
        z = np.zeros(c.shape, dtype=np.complex128)
        output = np.zeros(c.shape, dtype=np.float64)

        for i in range(self.max_iter):
            mask = np.abs(z) <= 2
            output[mask] += 1
            z[mask] = z[mask]**2 + c[mask]

        # Normalize the output
        output /= self.max_iter
        return self.amplitude * output


class GaussianNoise(NoiseFunction):
    def __init__(self, mean=0.0, std=1.0, seed=42):
        """
        Initialize Gaussian noise parameters.

        :param mean: Mean of the Gaussian distribution.
        :param std: Standard deviation of the Gaussian distribution.
        :param seed: Seed for random number generator.
        """
        self.mean = mean
        self.std = std
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of Gaussian noise values.
        """
        # Generate Gaussian noise with the same shape as x and y
        noise = self.rng.normal(loc=self.mean, scale=self.std, size=x.shape)
        return noise


class BrownianNoise(NoiseFunction):
    def __init__(self, step_size=1.0, scale=1.0, seed=42):
        """
        Initialize Brownian noise parameters.

        :param step_size: Step size for each iteration.
        :param scale: Scaling factor for the noise amplitude.
        :param seed: Seed for random number generator.
        """
        self.step_size = step_size
        self.scale = scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate Brownian noise for the given x and y coordinates.

        :param x: 2D numpy array of x coordinates.
        :param y: 2D numpy array of y coordinates.
        :return: 2D numpy array of Brownian noise values.
        """
        grid_shape = x.shape
        noise = np.zeros(grid_shape, dtype=np.float64)
        num_steps = int(self.step_size * 10)  # Example: 10 steps

        # Initialize the current position
        current_position = np.zeros(grid_shape, dtype=np.float64)

        for _ in range(num_steps):
            # Random angles for direction
            angles = self.rng.uniform(0, 2 * np.pi, size=grid_shape)
            dx = np.cos(angles) * self.step_size
            dy = np.sin(angles) * self.step_size

            # Update current position
            current_position += dx + dy

            # Accumulate noise
            noise += current_position

        # Scale the noise
        noise *= self.scale / num_steps

        return noise