"""
Unit tests for visualization application.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from visualization.application import ClientApplication, VisualizationMenu
from visualization.config import VisualizationConfigManager


class TestVisualizationMenu:
    @pytest.fixture
    def config_manager(self):
        """Create a config manager for testing."""
        return VisualizationConfigManager()

    def test_menu_initialization(self, config_manager):
        """Test menu initialization with default items."""
        menu = VisualizationMenu(config_manager)

        # Verify default sections exist
        assert "Configuration" in menu._menu_items
        assert "Visualization" in menu._menu_items
        assert "Performance" in menu._menu_items

    def test_custom_menu_item_addition(self, config_manager):
        """Test adding a custom menu item."""
        menu = VisualizationMenu(config_manager)

        # Create a mock callback
        mock_callback = Mock()

        # Add custom menu item
        menu.add_menu_item("Custom", "Test Action", mock_callback)

        # Verify item was added
        assert "Custom" in menu._menu_items
        assert "Test Action" in menu._menu_items["Custom"]

        # Simulate menu item selection
        menu._menu_items["Custom"]["Test Action"]()

        # Verify callback was called
        mock_callback.assert_called_once()

    def test_rendering_mode_toggle(self, config_manager):
        """Test rendering mode toggle functionality."""
        menu = VisualizationMenu(config_manager)

        # Initial state
        initial_mode = menu._current_rendering_mode

        # Toggle rendering mode
        menu.toggle_rendering_mode()

        # Verify mode changed
        assert menu._current_rendering_mode != initial_mode

    def test_performance_metrics_tracking(self, config_manager):
        """Test performance metrics tracking."""
        menu = VisualizationMenu(config_manager)

        # Simulate performance metric update
        test_metrics = {"fps": 60, "memory_usage": 500, "render_time": 0.016}

        menu.update_performance_metrics(test_metrics)

        # Verify metrics were stored
        assert menu._performance_metrics == test_metrics

    def test_configuration_update(self, config_manager):
        """Test configuration update through menu."""
        menu = VisualizationMenu(config_manager)

        # Prepare test configuration
        test_config = {"background_color": [0.2, 0.3, 0.4], "point_size": 5.0, "line_width": 2.0}

        menu.update_configuration(test_config)

        # Verify configuration was updated
        for key, value in test_config.items():
            assert config_manager.get(key) == value


class TestClientApplication:
    @pytest.fixture
    def client_app(self):
        """Create a client application for testing."""
        return ClientApplication()

    def test_application_initialization(self, client_app):
        """Test basic application initialization."""
        assert client_app is not None
        assert hasattr(client_app, "_menu")
        assert hasattr(client_app, "_visualizer")
        assert hasattr(client_app, "_communication_backend")

    @patch("polyscope.init")
    @patch("polyscope.set_user_callback")
    @patch("polyscope.set_dark_mode")
    def test_visualization_initialization(
        self, mock_set_dark_mode, mock_set_user_callback, mock_init, client_app
    ):
        """Test visualization environment initialization."""
        client_app.initialize_visualization()

        # Verify initialization calls
        mock_init.assert_called_once()
        mock_set_dark_mode.assert_called_once_with(True)
        mock_set_user_callback.assert_called_once()

    def test_message_communication(self, client_app):
        """Test message sending and subscription."""
        # Mock backend
        mock_backend = MagicMock()
        client_app._communication_backend = mock_backend

        # Test message sending
        test_topic = "test_channel"
        test_message = {"key": "value"}

        client_app.send_message(test_topic, test_message)
        mock_backend.send_message.assert_called_once_with(test_topic, test_message)

        # Test subscription
        mock_callback = Mock()
        client_app.subscribe(test_topic, mock_callback)
        mock_backend.subscribe.assert_called_once_with(test_topic, mock_callback)

    @patch("polyscope.show")
    def test_application_run(self, mock_show, client_app):
        """Test application run method."""
        client_app.run()

        # Verify visualization show method called
        mock_show.assert_called_once()

    def test_data_visualization(self, client_app):
        """Test data visualization methods."""
        # Prepare test data
        test_points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        test_mesh_vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        test_mesh_faces = [[0, 1, 2]]

        # Add point cloud
        client_app.add_point_cloud("test_points", test_points)

        # Add mesh
        client_app.add_mesh("test_mesh", test_mesh_vertices, test_mesh_faces)

        # Verify data added to visualizer
        assert "test_points" in client_app._visualizer._point_clouds
        assert "test_mesh" in client_app._visualizer._meshes

    def test_error_handling(self, client_app):
        """Test error handling in application."""
        # Simulate backend communication error
        mock_backend = MagicMock()
        mock_backend.send_message.side_effect = ConnectionError("Test error")
        client_app._communication_backend = mock_backend

        with pytest.raises(ConnectionError):
            client_app.send_message("error_topic", {"error": "test"})
