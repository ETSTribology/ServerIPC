import json
from jsonschema import validate, ValidationError
from noise_functions import (
    FractalBrownianMotion,
    PerlinNoise,
    SineWave,
    SquareWave,
    BeckmannNoise,
    GGXNoise,
    BlinnPhongNoise,
    MandelbrotNoise,
    GaussianNoise,
    BrownianNoise
)
from heightmap_mesh_generator import HeightmapMeshGenerator

class HeightmapMeshGeneratorFactory:
    def __init__(self, config_path, schema_path):
        """
        Initialize the factory with paths to the configuration and schema files.
        
        :param config_path: Path to the JSON configuration file.
        :param schema_path: Path to the JSON schema file.
        """
        self.config_path = config_path
        self.schema_path = schema_path
        self.config = self._load_config()
        self.schema = self._load_schema()
        self._validate_config()

    def _load_config(self):
        """Load the JSON configuration file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _load_schema(self):
        """Load the JSON schema file."""
        with open(self.schema_path, 'r') as f:
            return json.load(f)

    def _validate_config(self):
        """Validate the configuration against the schema."""
        try:
            validate(instance=self.config, schema=self.schema)
        except ValidationError as ve:
            raise ValueError(f"Configuration validation error: {ve.message}")

    def create_noise_function(self):
        """Instantiate the appropriate NoiseFunction based on config."""
        noise_config = self.config.get("noise", {})
        noise_type = noise_config.get("type", "fractal_brownian_motion").lower()
        amplitude = noise_config.get("amplitude", 1.0)
        frequency = noise_config.get("frequency", 1.0)
        octaves = noise_config.get("octaves", 4)
        persistence = noise_config.get("persistence", 0.5)
        lacunarity = noise_config.get("lacunarity", 2.0)
        alpha = noise_config.get("alpha", 0.5)
        n = noise_config.get("n", 20)
        seed = noise_config.get("seed", 42)

        noise_functions = {
            "fractal_brownian_motion": FractalBrownianMotion,
            "perlin": PerlinNoise,
            "sine": SineWave,
            "square": SquareWave,
            "beckmann": BeckmannNoise,
            "ggx": GGXNoise,
            "blinn": BlinnPhongNoise,
            "mandelbrot": MandelbrotNoise,
            "gaussian": GaussianNoise,
            "brownian": BrownianNoise
        }

        noise_class = noise_functions.get(noise_type)
        if not noise_class:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        # Instantiate the noise function with relevant parameters
        if noise_type == "fractal_brownian_motion":
            return noise_class(amplitude=amplitude, frequency=frequency, octaves=octaves,
                            persistence=persistence, lacunarity=lacunarity, seed=seed)
        elif noise_type == "perlin":
            return noise_class(amplitude=amplitude, frequency=frequency, seed=seed)
        elif noise_type == "sine":
            return noise_class(amplitude=amplitude, frequency=frequency)
        elif noise_type == "square":
            return noise_class(amplitude=amplitude, frequency=frequency)
        elif noise_type == "beckmann":
            return noise_class(alpha=alpha)
        elif noise_type == "ggx":
            return noise_class(alpha=alpha)
        elif noise_type == "blinn":
            return noise_class(n=n)
        elif noise_type == "mandelbrot":
            return noise_class(amplitude=amplitude, frequency=frequency, seed=seed)
        elif noise_type == "gaussian":
            mean = noise_config.get("mean", 0.0)
            std = noise_config.get("std", 1.0)
            return noise_class(mean=mean, std=std, seed=seed)
        elif noise_type == "brownian":
            step_size = noise_config.get("step_size", 1.0)
            scale = noise_config.get("scale", 1.0)
            return noise_class(step_size=step_size, scale=scale, seed=seed)
        else:
            raise ValueError(f"Noise type '{noise_type}' not implemented.")

    def create_generator(self):
        """Create an instance of HeightmapMeshGenerator based on config."""
        enabled = self.config.get("enabled", False)
        if not enabled:
            raise ValueError("Mesh generation is disabled in the configuration.")

        method = self.config.get("method", "noise").lower()
        if method != "noise":
            raise ValueError(f"Unsupported method: {method}")

        noise_function = self.create_noise_function()

        output_config = self.config.get("output", {})
        output_format = output_config.get("format", "obj")
        resolution = output_config.get("resolution", [512, 512])

        # Extract additional parameters if needed
        generate_displacement_map = self.config["noise"].get("type", "").lower() in ["sine", "square"]

        return HeightmapMeshGenerator(
            noise_function=noise_function,
            size_x=resolution[0],
            size_y=resolution[1],
            amplitude=self.config["noise"].get("amplitude", 1.0),
            generate_displacement_map=generate_displacement_map,
            output_folder=self.config.get("output_folder", "output")
            # Add more parameters as needed
        )