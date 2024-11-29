import os
import csv
import logging
import tempfile
import numpy as np
import pytest
import ipctk

from simulation.core.contact.extract import CollisionInfoCollector


class MockCollisionConstraints:
    def __init__(self):
        self.vv_collisions = [MockCollision() for _ in range(2)]
        self.ev_collisions = [MockCollision() for _ in range(2)]
        self.ee_collisions = [MockCollision() for _ in range(2)]
        self.fv_collisions = [MockCollision() for _ in range(2)]


class MockCollision:
    def __init__(self):
        self.weight = 1.0
        self.mu = 0.5
        self.normal_force_magnitude = 10.0
        self._tangent_basis = np.eye(3)
        self._relative_velocity = np.array([0.1, 0.2, 0.3])

    def compute_tangent_basis(self, *args):
        return self._tangent_basis

    def relative_velocity(self, *args):
        return self._relative_velocity


class TestCollisionInfoCollector:
    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_filename = temp_file.name
        yield temp_filename
        # Clean up the file after the test
        os.unlink(temp_filename)

    @pytest.fixture
    def collector(self, temp_csv_file):
        """Create a CollisionInfoCollector with a temporary file."""
        return CollisionInfoCollector(filename=temp_csv_file)

    def test_initialization(self, collector):
        """Test initialization of CollisionInfoCollector."""
        assert collector.filename is not None
        assert collector.collision_data == []
        assert len(collector.header) == 8

    def test_collect_collision_info(self, collector, caplog):
        """Test collecting collision information."""
        caplog.set_level(logging.INFO)
        
        # Create mock objects
        cmesh = np.array([[0, 0, 0], [1, 1, 1]])
        BX = np.array([[0, 0, 0], [1, 1, 1]])
        BXdot = np.zeros_like(BX)
        epsv = 0.01
        EF = 1.0
        fconstraints = MockCollisionConstraints()

        # Collect collision info
        collector.collect_collision_info(fconstraints, cmesh, BX, BXdot, epsv, EF)

        # Check logging
        assert any('Collecting collision information' in record.message for record in caplog.records)
        assert any('Collision information collected' in record.message for record in caplog.records)

        # Check collected data
        assert len(collector.collision_data) > 0
        
        # Verify data structure
        for row in collector.collision_data:
            assert len(row) == 8

    def test_save_to_csv(self, collector, temp_csv_file, caplog):
        """Test saving collision data to CSV."""
        caplog.set_level(logging.INFO)
        
        # Create mock data
        collector.collision_data = [
            ["Vertex-Vertex", [0.1, 0.2], [0.3, 0.4], 1.0, 0.5, 10.0, 0.01, 1.0],
            ["Edge-Vertex", [0.5, 0.6], [0.7, 0.8], 1.0, 0.5, 10.0, 0.02, 1.0]
        ]

        # Save to CSV
        collector.save_to_csv()

        # Check logging
        assert any('Saving collision data' in record.message for record in caplog.records)
        assert any('Collision data saved successfully' in record.message for record in caplog.records)

        # Verify CSV file contents
        with open(temp_csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            assert header == collector.header
            
            data_rows = list(reader)
            assert len(data_rows) == len(collector.collision_data)

    def test_process_collision_with_different_inputs(self, collector):
        """Test processing collisions with various inputs."""
        # Prepare mock data
        tangent_basis = np.eye(3)
        relative_velocity = np.array([0.1, 0.2, 0.3])
        epsv = 0.01
        EF = 1.0

        test_cases = [
            ("Vertex-Vertex", 1.0, 0.5, 10.0),
            ("Edge-Vertex", 0.5, 0.3, 5.0),
            ("Face-Vertex", 2.0, 0.7, 15.0)
        ]

        for collision_type, weights, coefficients, normal_force_magnitude in test_cases:
            initial_data_length = len(collector.collision_data)
            
            collector._process_collision(
                collision_type=collision_type,
                collision=None,
                tangent_basis=tangent_basis,
                relative_velocity=relative_velocity,
                weights=weights,
                coefficients=coefficients,
                normal_force_magnitude=normal_force_magnitude,
                epsv=epsv,
                EF=EF
            )

            # Check that a new row was added
            assert len(collector.collision_data) == initial_data_length + 1
            
            # Verify row structure
            new_row = collector.collision_data[-1]
            assert new_row[0] == collision_type
            assert len(new_row) == 8

    def test_error_handling_csv_save(self, collector, monkeypatch):
        """Test error handling during CSV save."""
        # Simulate file write error
        def mock_open(*args, **kwargs):
            raise IOError("Simulated file write error")

        monkeypatch.setattr('builtins.open', mock_open)

        # Prepare some data
        collector.collision_data = [
            ["Vertex-Vertex", [0.1, 0.2], [0.3, 0.4], 1.0, 0.5, 10.0, 0.01, 1.0]
        ]

        # Attempt to save (should log an error)
        with pytest.raises(IOError):
            collector.save_to_csv()
