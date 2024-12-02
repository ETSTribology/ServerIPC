import numpy as np
import pytest

from simulation.core.utils.modifier.transformation import (
    apply_scaling, 
    apply_rotation, 
    apply_translation, 
    apply_transformations
)


class TestTransformationUtils:
    def test_apply_scaling(self):
        """Test scaling transformation."""
        vertices = np.array([[1, 2, 3], [4, 5, 6]])
        scale = [2, 3, 4]
        
        scaled_vertices = apply_scaling(vertices, scale)
        
        expected_vertices = np.array([[2, 6, 12], [8, 15, 24]])
        np.testing.assert_array_almost_equal(scaled_vertices, expected_vertices)

    def test_apply_scaling_invalid_input(self):
        """Test scaling with invalid input."""
        vertices = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="Scale must be a list of three elements"):
            apply_scaling(vertices, [1, 2])

    def test_apply_rotation(self):
        """Test rotation transformation."""
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # 90-degree rotation around Z-axis
        rotation = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]
        
        rotated_vertices = apply_rotation(vertices, rotation)
        
        # Expected vertices after 90-degree rotation around Z-axis
        expected_vertices = np.array([
            [0, -1, 0], 
            [1, 0, 0], 
            [0, 0, 1]
        ])
        
        np.testing.assert_array_almost_equal(rotated_vertices, expected_vertices, decimal=7)

    def test_apply_rotation_invalid_input(self):
        """Test rotation with invalid input."""
        vertices = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="Rotation must be a list of four elements"):
            apply_rotation(vertices, [1, 2, 3])

    def test_apply_translation(self):
        """Test translation transformation."""
        vertices = np.array([[1, 2, 3], [4, 5, 6]])
        translation = [10, -5, 2]
        
        translated_vertices = apply_translation(vertices, translation)
        
        expected_vertices = np.array([[11, -3, 5], [14, 0, 8]])
        np.testing.assert_array_almost_equal(translated_vertices, expected_vertices)

    def test_apply_translation_invalid_input(self):
        """Test translation with invalid input."""
        vertices = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="Translation must be a list of three elements"):
            apply_translation(vertices, [1, 2])

    def test_apply_transformations(self):
        """Test combined transformations."""
        vertices = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        
        scale = [2, 2, 2]
        rotation = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]  # 90-degree rotation around Z-axis
        translation = [5, 0, 0]
        
        transformed_vertices = apply_transformations(vertices, scale, rotation, translation)
        
        # Manually calculate expected result
        scaled_vertices = apply_scaling(vertices, scale)
        rotated_vertices = apply_rotation(scaled_vertices, rotation)
        expected_vertices = apply_translation(rotated_vertices, translation)
        
        np.testing.assert_array_almost_equal(transformed_vertices, expected_vertices, decimal=7)

    def test_apply_transformations_invalid_input(self):
        """Test combined transformations with invalid input."""
        vertices = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(ValueError):
            apply_transformations(
                vertices, 
                scale=[1, 2], 
                rotation=[0, 0, 0, 1], 
                translation=[0, 0, 0]
            )
        
        with pytest.raises(ValueError):
            apply_transformations(
                vertices, 
                scale=[1, 1, 1], 
                rotation=[0, 0], 
                translation=[0, 0, 0]
            )
        
        with pytest.raises(ValueError):
            apply_transformations(
                vertices, 
                scale=[1, 1, 1], 
                rotation=[0, 0, 0, 1], 
                translation=[0, 0]
            )

    def test_transformation_order(self):
        """Test that transformations are applied in the correct order."""
        vertices = np.array([[1, 0, 0]])
        
        # Scale, then rotate, then translate
        scale = [2, 2, 2]
        rotation = [0, 0, np.sin(np.pi/2), np.cos(np.pi/2)]  # 90-degree rotation around Z-axis
        translation = [5, 0, 0]
        
        transformed_vertices = apply_transformations(vertices, scale, rotation, translation)
        
        # Manually apply transformations
        scaled_vertices = apply_scaling(vertices, scale)
        rotated_vertices = apply_rotation(scaled_vertices, rotation)
        expected_vertices = apply_translation(rotated_vertices, translation)
        
        np.testing.assert_array_almost_equal(transformed_vertices, expected_vertices, decimal=7)
