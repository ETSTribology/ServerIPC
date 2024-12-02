"""
Unit tests for visualization configuration management.
"""

import json
import os
import tempfile

import pytest

from visualization.config import VisualizationConfig, VisualizationConfigManager


class TestVisualizationConfig:
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = VisualizationConfig()

        assert config.backend == "redis"
        assert config.rendering_mode == "default"
        assert config.color_scheme == "light"
        assert config.interactive_mode is True
        assert config.logging_level == "INFO"

    def test_config_validation(self):
        """Test configuration validation."""
        config = VisualizationConfig()
        assert config.validate() is True

        # Test invalid backend
        config.backend = "invalid_backend"
        assert config.validate() is False

        # Test invalid rendering mode
        config.backend = "redis"
        config.rendering_mode = "non_existent_mode"
        assert config.validate() is False

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            try:
                # Create and save config
                original_config = VisualizationConfig(
                    backend="websocket", rendering_mode="high_performance", color_scheme="dark"
                )
                original_config.to_json(temp_file.name)

                # Load config from file
                loaded_config = VisualizationConfig.from_json(temp_file.name)

                # Verify loaded config matches original
                assert loaded_config.backend == "websocket"
                assert loaded_config.rendering_mode == "high_performance"
                assert loaded_config.color_scheme == "dark"
            finally:
                os.unlink(temp_file.name)

    def test_config_update(self):
        """Test dynamic configuration updates."""
        config_manager = VisualizationConfigManager()

        # Test initial state
        initial_settings = config_manager.get_rendering_settings()
        assert initial_settings["mode"] == "default"
        assert initial_settings["color_scheme"] == "light"

        # Update configuration
        config_manager.update_config(rendering_mode="high_performance", color_scheme="dark")

        # Verify updates
        updated_settings = config_manager.get_rendering_settings()
        assert updated_settings["mode"] == "high_performance"
        assert updated_settings["color_scheme"] == "dark"

    def test_invalid_config_update(self):
        """Test handling of invalid configuration updates."""
        config_manager = VisualizationConfigManager()

        # Test invalid configuration update
        with pytest.raises(ValueError, match="Invalid configuration parameters"):
            config_manager.update_config(backend="invalid_backend")


class TestBackendConfiguration:
    def test_backend_config_retrieval(self):
        """Test backend configuration retrieval."""
        config_manager = VisualizationConfigManager()

        backend_config = config_manager.get_backend_config()

        assert "host" in backend_config
        assert "port" in backend_config
        assert backend_config["host"] == "localhost"
        assert backend_config["port"] == 6379

    def test_custom_backend_config(self):
        """Test custom backend configuration."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            try:
                # Create a custom configuration
                custom_config = {
                    "backend": "websocket",
                    "backend_config": {"host": "custom.example.com", "port": 8080},
                }
                json.dump(custom_config, temp_file)
                temp_file.flush()

                # Load custom configuration
                config_manager = VisualizationConfigManager(temp_file.name)

                # Verify custom configuration
                backend_config = config_manager.get_backend_config()
                assert backend_config["host"] == "custom.example.com"
                assert backend_config["port"] == 8080
            finally:
                os.unlink(temp_file.name)
