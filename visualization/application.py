"""
Visualization Application with Polyscope Integration

This module provides a comprehensive visualization framework 
with support for multiple communication backends and dynamic configuration.
"""

import logging
import polyscope as ps
import numpy as np
from typing import Any, Dict, Optional, Callable

from .config import VisualizationConfigManager
from .backends.registry import CommunicationBackendRegistry
from .backends.base import BaseCommunicationBackend


class VisualizationMenu:
    """
    Advanced menu system for Polyscope visualization.
    Provides dynamic, configurable menu options.
    """
    def __init__(self, config_manager: VisualizationConfigManager):
        """
        Initialize visualization menu.
        
        Args:
            config_manager: Configuration management instance
        """
        self._config_manager = config_manager
        self._menu_items = {}
        self._setup_default_menu()

    def _setup_default_menu(self):
        """Set up default menu items."""
        self.add_menu_item(
            "Configuration", 
            "Backend Settings", 
            self._show_backend_settings
        )
        self.add_menu_item(
            "Visualization", 
            "Rendering Mode", 
            self._toggle_rendering_mode
        )
        self.add_menu_item(
            "Performance", 
            "Toggle Tracking", 
            self._toggle_performance_tracking
        )

    def add_menu_item(
        self, 
        section: str, 
        label: str, 
        callback: Callable[[], None]
    ):
        """
        Add a custom menu item.
        
        Args:
            section: Menu section name
            label: Menu item label
            callback: Function to execute on selection
        """
        if section not in self._menu_items:
            self._menu_items[section] = {}
        
        self._menu_items[section][label] = callback

    def render_menu(self):
        """Render the dynamic menu in Polyscope."""
        for section, items in self._menu_items.items():
            if ps.ImGui.TreeNode(section):
                for label, callback in items.items():
                    if ps.ImGui.Button(label):
                        callback()
                ps.ImGui.TreePop()

    def _show_backend_settings(self):
        """Display and modify backend settings."""
        backend_config = self._config_manager.get_backend_config()
        # Implement dynamic configuration UI
        # Use ImGui for interactive configuration

    def _toggle_rendering_mode(self):
        """Toggle between rendering modes."""
        current_mode = self._config_manager.get_rendering_settings()['mode']
        modes = ['default', 'high_performance', 'detailed']
        next_mode = modes[(modes.index(current_mode) + 1) % len(modes)]
        self._config_manager.update_config(rendering_mode=next_mode)

    def _toggle_performance_tracking(self):
        """Toggle performance tracking."""
        current_tracking = self._config_manager.get_rendering_settings()['interactive']
        self._config_manager.update_config(performance_tracking=not current_tracking)


class ClientApplication:
    """
    Comprehensive visualization client with multi-backend support.
    
    Supports dynamic configuration, multiple communication backends,
    and advanced Polyscope visualization.
    """

    def __init__(
        self, 
        config_path: Optional[str] = None,
        backend: Optional[str] = None
    ):
        """
        Initialize visualization client.
        
        Args:
            config_path: Path to configuration file
            backend: Specific communication backend to use
        """
        # Initialize configuration
        self._config_manager = VisualizationConfigManager(config_path)
        
        # Determine backend (use config or override)
        backend = backend or self._config_manager.get_backend_config().get('backend', 'redis')
        
        # Initialize communication backend
        self._communication_backend: BaseCommunicationBackend = (
            CommunicationBackendRegistry.create(
                backend, 
                self._config_manager.get_backend_config()
            )
        )
        
        # Initialize visualization menu
        self._menu = VisualizationMenu(self._config_manager)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self._config_manager.get_rendering_settings().get('logging_level', 'INFO'))
        )
        self._logger = logging.getLogger(__name__)

    def initialize_visualization(self):
        """Initialize Polyscope visualization environment."""
        ps.init()
        
        # Configure Polyscope based on settings
        rendering_settings = self._config_manager.get_rendering_settings()
        
        # Set color scheme
        if rendering_settings['color_scheme'] == 'dark':
            ps.set_dark_mode()
        
        # Configure interactive mode
        ps.set_user_callback(self._update_visualization)

    def _update_visualization(self):
        """
        Periodic update method for visualization.
        Handles dynamic rendering and menu interactions.
        """
        # Render custom menu
        ps.ImGui.Begin("Simulation Controls")
        self._menu.render_menu()
        ps.ImGui.End()

        # Perform any periodic updates or data synchronization
        # This could involve fetching latest simulation data
        # and updating Polyscope visualization accordingly

    def subscribe_to_topic(self, topic: str, callback: Callable[[Any], None]):
        """
        Subscribe to a communication topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Function to handle received messages
        """
        self._communication_backend.subscribe(topic, callback)

    def send_message(self, topic: str, message: Any):
        """
        Send a message via the current communication backend.
        
        Args:
            topic: Message destination
            message: Payload to send
        """
        self._communication_backend.send_message(topic, message)

    def run(self):
        """
        Start the visualization application.
        Enters the main event loop.
        """
        try:
            self.initialize_visualization()
            ps.show()
        except Exception as e:
            self._logger.error(f"Visualization error: {e}")
        finally:
            self._communication_backend.disconnect()