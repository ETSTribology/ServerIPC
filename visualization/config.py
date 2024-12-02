"""Configuration management for visualization."""

from typing import Dict, Any, Optional
import json
import os
from dataclasses import dataclass, asdict, field

from simulation.core.config.config import ConfigManager


@dataclass
class VisualizationConfig:
    """
    Comprehensive configuration for visualization.
    
    Attributes:
        backend: Communication backend type (redis, grpc, websocket)
        backend_config: Backend-specific configuration
        rendering_mode: Visualization rendering mode
        color_scheme: Color palette for visualization
        interactive_mode: Enable/disable interactive features
        logging_level: Logging verbosity
        performance_tracking: Enable performance metrics
        extensions: Additional configuration options
    """
    backend: str = 'redis'
    backend_config: Dict[str, Any] = field(default_factory=lambda: {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    })
    rendering_mode: str = 'default'
    color_scheme: str = 'light'
    interactive_mode: bool = True
    logging_level: str = 'INFO'
    performance_tracking: bool = False
    extensions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, filepath: Optional[str] = None):
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
        
        Returns:
            Configured VisualizationConfig instance
        """
        if not filepath or not os.path.exists(filepath):
            return cls()

        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)

    def to_json(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert configuration to JSON.
        
        Args:
            filepath: Optional path to save configuration
        
        Returns:
            Configuration dictionary
        """
        config_dict = asdict(self)
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        return config_dict

    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            Boolean indicating configuration validity
        """
        # Add validation logic for configuration parameters
        valid_backends = ['redis', 'grpc', 'websocket']
        valid_rendering_modes = ['default', 'high_performance', 'detailed']
        valid_color_schemes = ['light', 'dark', 'custom']
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        return all([
            self.backend in valid_backends,
            self.rendering_mode in valid_rendering_modes,
            self.color_scheme in valid_color_schemes,
            self.logging_level in valid_log_levels
        ])


class VisualizationConfigManager(ConfigManager):
    """
    Specialized configuration manager for visualization.
    Extends the base ConfigManager with visualization-specific methods.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize visualization configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        super().__init__()
        self._config = VisualizationConfig.from_json(config_path)

    def get_backend_config(self) -> Dict[str, Any]:
        """
        Retrieve backend-specific configuration.
        
        Returns:
            Backend configuration dictionary
        """
        return self._config.backend_config

    def get_rendering_settings(self) -> Dict[str, Any]:
        """
        Retrieve rendering configuration.
        
        Returns:
            Rendering configuration dictionary
        """
        return {
            'mode': self._config.rendering_mode,
            'color_scheme': self._config.color_scheme,
            'interactive': self._config.interactive_mode
        }

    def update_config(self, **kwargs) -> None:
        """
        Update configuration dynamically.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        if not self._config.validate():
            raise ValueError("Invalid configuration parameters")

    def save_config(self, filepath: Optional[str] = None) -> None:
        """
        Save current configuration.
        
        Args:
            filepath: Optional path to save configuration
        """
        self._config.to_json(filepath)

    def load_config(self, filepath: str) -> None:
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
        """
        self._config = VisualizationConfig.from_json(filepath)
        if not self._config.validate():
            raise ValueError("Invalid configuration file")
