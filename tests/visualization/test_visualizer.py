"""
Unit tests for Polyscope Visualizer in visualization module.
"""

from unittest.mock import patch

import numpy as np
import pytest

from visualization.visualizer import PolyscopeVisualizer


class TestPolyscopeVisualizer:
    @pytest.fixture
    def visualizer(self):
        """Create a PolyscopeVisualizer instance for testing."""
        return PolyscopeVisualizer()

    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer is not None
        assert hasattr(visualizer, "_window")

    @patch("polyscope.init")
    def test_initialize_window(self, mock_init, visualizer):
        """Test window initialization."""
        visualizer.initialize_window()
        mock_init.assert_called_once()

    def test_add_point_cloud(self, visualizer):
        """Test adding a point cloud to visualization."""
        points = np.random.rand(100, 3)
        point_cloud_name = "test_points"

        visualizer.add_point_cloud(point_cloud_name, points)

        # Verify point cloud addition logic
        assert hasattr(visualizer, "_point_clouds")
        assert point_cloud_name in visualizer._point_clouds

    def test_add_mesh(self, visualizer):
        """Test adding a mesh to visualization."""
        vertices = np.random.rand(100, 3)
        faces = np.random.randint(0, 100, (50, 3))
        mesh_name = "test_mesh"

        visualizer.add_mesh(mesh_name, vertices, faces)

        # Verify mesh addition logic
        assert hasattr(visualizer, "_meshes")
        assert mesh_name in visualizer._meshes

    def test_update_point_cloud(self, visualizer):
        """Test updating an existing point cloud."""
        points = np.random.rand(100, 3)
        point_cloud_name = "test_points"

        # First add point cloud
        visualizer.add_point_cloud(point_cloud_name, points)

        # Update point cloud
        new_points = np.random.rand(150, 3)
        visualizer.update_point_cloud(point_cloud_name, new_points)

        # Verify update logic
        assert visualizer._point_clouds[point_cloud_name].shape == new_points.shape

    @patch("polyscope.show")
    def test_show(self, mock_show, visualizer):
        """Test show method."""
        visualizer.show()
        mock_show.assert_called_once()

    def test_clear(self, visualizer):
        """Test clearing visualization."""
        # Add some elements
        points = np.random.rand(100, 3)
        mesh_vertices = np.random.rand(100, 3)
        mesh_faces = np.random.randint(0, 100, (50, 3))

        visualizer.add_point_cloud("test_points", points)
        visualizer.add_mesh("test_mesh", mesh_vertices, mesh_faces)

        visualizer.clear()

        # Verify clear logic
        assert len(visualizer._point_clouds) == 0
        assert len(visualizer._meshes) == 0

    def test_invalid_input_handling(self, visualizer):
        """Test handling of invalid input data."""
        # Invalid point cloud (wrong dimensions)
        with pytest.raises(ValueError):
            visualizer.add_point_cloud("invalid_points", np.random.rand(10, 2))

        # Invalid mesh (mismatched vertices and faces)
        with pytest.raises(ValueError):
            vertices = np.random.rand(100, 3)
            faces = np.random.randint(0, 50, (50, 3))
            visualizer.add_mesh("invalid_mesh", vertices, faces)
