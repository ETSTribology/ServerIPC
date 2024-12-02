from .config import ConfigManager

def get_config():
    """
    Get the global configuration instance.

    Returns:
        ConfigManager: The singleton configuration manager instance.
    """
    return ConfigManager.get_instance()