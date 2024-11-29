import os
import pytest
import numpy as np
import tempfile
import meshio

from simulation.core.utils.io.io import load_mesh, load_individual_meshes
from simulation.core.utils.config.config import load_material_properties, load_transform_properties


class TestIOMeshLoading:
    @pytest.fixture
    def sample_mesh_file(self):
        """Create a temporary mesh file for testing."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.msh', delete=False) as temp_file:
            try:
                # Create a simple tetrahedral mesh
                points = np.array([
                    [0, 0, 0],   # Vertex 0
                    [1, 0, 0],   # Vertex 1
                    [0, 1, 0],   # Vertex 2
                    [0, 0, 1]    # Vertex 3
                ], dtype=np.float64)
                
                cells = [("tetra", np.array([[0, 1, 2, 3]]))]
                
                test_mesh = meshio.Mesh(points, cells)
                test_mesh.write(temp_file.name)
                
                yield temp_file.name
            finally:
                os.unlink(temp_file.name)

    def test_load_mesh_successful(self, sample_mesh_file):
        """Test successful mesh loading."""
        vertices, connectivity = load_mesh(sample_mesh_file)
        
        assert isinstance(vertices, np.ndarray)
        assert isinstance(connectivity, np.ndarray)
        
        assert vertices.shape == (4, 3)  # 4 vertices
        assert connectivity.shape == (1, 4)  # 1 tetrahedron with 4 vertices
        
        # Verify data types
        assert vertices.dtype == np.float64
        assert connectivity.dtype == np.int64

    def test_load_mesh_invalid_file(self):
        """Test loading an invalid mesh file."""
        with pytest.raises(Exception):
            load_mesh("/path/to/non_existent_file.msh")

    def test_load_mesh_no_tetra_cells(self):
        """Test loading a mesh without tetrahedral cells."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.msh', delete=False) as temp_file:
            try:
                # Create a triangle mesh (no tetrahedra)
                points = np.array([
                    [0, 0, 0],   # Vertex 0
                    [1, 0, 0],   # Vertex 1
                    [0, 1, 0]    # Vertex 2
                ], dtype=np.float64)
                
                cells = [("triangle", np.array([[0, 1, 2]]))]
                
                test_mesh = meshio.Mesh(points, cells)
                test_mesh.write(temp_file.name)
                
                with pytest.raises(ValueError, match="No tetrahedral cells"):
                    load_mesh(temp_file.name)
            finally:
                os.unlink(temp_file.name)

    def test_load_individual_meshes(self):
        """Test loading individual meshes with configurations."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.msh', delete=False) as temp_file:
            try:
                # Create a simple tetrahedral mesh
                points = np.array([
                    [0, 0, 0],   # Vertex 0
                    [1, 0, 0],   # Vertex 1
                    [0, 1, 0],   # Vertex 2
                    [0, 0, 1]    # Vertex 3
                ], dtype=np.float64)
                
                cells = [("tetra", np.array([[0, 1, 2, 3]]))]
                
                test_mesh = meshio.Mesh(points, cells)
                test_mesh.write(temp_file.name)
                
                # Create input configuration
                inputs = [{
                    "path": temp_file.name,
                    "percent_fixed": 0.0,
                    "material": "default",
                    "transform": {
                        "scale": [1.0, 1.0, 1.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "translation": [0.0, 0.0, 0.0]
                    }
                }]

                # Load meshes
                meshes, mesh_configs = load_individual_meshes(inputs)
                
                assert len(meshes) == 1
                assert len(mesh_configs) == 1
                
                vertices, connectivity = meshes[0]
                assert vertices.shape == (4, 3)
                assert connectivity.shape == (1, 4)
            finally:
                os.unlink(temp_file.name)
