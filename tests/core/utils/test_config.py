import os
import pytest
import tempfile
import yaml
from typing import Dict, Any

from simulation.core.config.config import ConfigManager

@pytest.fixture(autouse=True)
def initialize_config():
    ConfigManager.reset()
    ConfigManager.quick_init()

class TestConfigManager:
    def test_singleton_behavior(self):
        """Test that ConfigManager is a singleton."""
        config_manager1 = ConfigManager()
        config_manager2 = ConfigManager()
        
        assert config_manager1 is config_manager2

    def test_default_config_generation(self):
        """Test the generation of default configuration."""
        config_manager = ConfigManager()
        default_config = config_manager.generate_default_config()

        # Verify key sections exist
        required_sections = [
            "name", "inputs", "materials", "friction", "simulation", 
            "communication", "serialization", "optimizer", 
            "linear_solver", "ipc", "initial_conditions", "logging"
        ]
        
        for section in required_sections:
            assert section in default_config, f"Missing section: {section}"

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config_manager = ConfigManager()
        default_config = config_manager.generate_default_config()

        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_json:
            try:
                import json
                json.dump(default_config, temp_json)
                temp_json.close()

                with open(temp_json.name, 'r') as f:
                    loaded_config = json.load(f)

                assert loaded_config == default_config
            finally:
                os.unlink(temp_json.name)

        # Test YAML serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_yaml:
            try:
                yaml.safe_dump(default_config, temp_yaml)
                temp_yaml.close()

                with open(temp_yaml.name, 'r') as f:
                    loaded_config = yaml.safe_load(f)

                assert loaded_config == default_config
            finally:
                os.unlink(temp_yaml.name)

    def test_config_validation(self):
        """Test basic configuration validation."""
        config_manager = ConfigManager()
        default_config = config_manager.generate_default_config()

        # Validate specific nested configurations
        assert default_config['name'] == "Default Configuration"
        assert default_config['inputs'][0]['percent_fixed'] == 0.0
        assert default_config['simulation']['dt'] == 1/60

        # Validate material properties
        material = default_config['materials'][0]
        assert material['name'] == "default"
        assert material['density'] == 1000.0
        assert material['young_modulus'] == 1e6

    def test_config_immutability(self):
        """Test that the default configuration is not modified by multiple instantiations."""
        config_manager1 = ConfigManager()
        config_manager2 = ConfigManager()

        # Modify a nested configuration in one instance
        config_manager1.config['inputs'][0]['percent_fixed'] = 0.5

        # Verify that the other instance's configuration remains unchanged
        assert config_manager2.config['inputs'][0]['percent_fixed'] == 0.0
