import logging
import pytest
import numpy as np
import ipctk

from simulation.core.contact.barrier_updater import (
    BarrierUpdaterBase, 
    BarrierUpdater, 
    BarrierUpdaterFactory, 
    RegistryContainer
)
from simulation.core.parameters import ParametersBase


class MockParameters(ParametersBase):
    def __init__(self):
        self.mesh = np.array([[0, 0, 0], [1, 1, 1]])
        self.cmesh = np.array([[0, 0, 0], [1, 1, 1]])
        self.dmin = 0.01
        self.dhat = 0.1
        self.broad_phase_method = 'default'
        self.cconstraints = ipctk.NormalCollisions()
        self.fconstraints = ipctk.TangentialCollisions()
        self.kB = 1.0
        self.maxkB = 10.0
        self.dprev = 0.1
        self.bboxdiag = 1.732  # sqrt(3)


class TestBarrierUpdaterBase:
    def test_barrier_updater_base_abstract(self):
        """Test that BarrierUpdaterBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BarrierUpdaterBase(MockParameters())  # type: ignore

    def test_base_class_interface(self):
        """Ensure the base class defines the required interface."""
        assert hasattr(BarrierUpdaterBase, '__call__')
        assert hasattr(BarrierUpdaterBase, '__init__')


class TestBarrierUpdater:
    @pytest.fixture
    def mock_params(self):
        """Create a mock parameters object."""
        return MockParameters()

    def test_barrier_updater_call(self, mock_params, caplog):
        """Test the barrier updater's __call__ method."""
        caplog.set_level(logging.DEBUG)
        
        # Prepare test data
        xk = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)

        # Create updater
        updater = BarrierUpdater(mock_params)
        updater(xk)

        # Check logging
        assert any('Updating barrier stiffness' in record.message for record in caplog.records)
        assert any('Barrier stiffness updated' in record.message for record in caplog.records)

        # Check that parameters were updated
        assert mock_params.kB is not None
        assert mock_params.dprev is not None
        assert mock_params.kB != 1.0  # Ensure kB was modified

    def test_barrier_updater_with_different_inputs(self, mock_params):
        """Test barrier updater with various input configurations."""
        updater = BarrierUpdater(mock_params)
        
        # Different input configurations
        test_cases = [
            np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
            np.array([[1, 1, 1], [2, 2, 2]]),
            np.random.rand(2, 3)
        ]

        for xk in test_cases:
            initial_kB = mock_params.kB
            updater(xk)
            assert mock_params.kB is not None
            assert mock_params.dprev is not None
            assert mock_params.kB != initial_kB


class TestBarrierUpdaterFactory:
    def test_barrier_updater_factory_singleton(self):
        """Test that BarrierUpdaterFactory is a singleton."""
        factory1 = BarrierUpdaterFactory()
        factory2 = BarrierUpdaterFactory()
        assert factory1 is factory2

    def test_barrier_updater_factory_create_default(self, caplog):
        """Test creating a default barrier updater through the factory."""
        caplog.set_level(logging.ERROR)
        
        # Create mock parameters
        mock_params = MockParameters()

        # Create through factory
        updater = BarrierUpdaterFactory.create('default', mock_params)
        
        assert isinstance(updater, BarrierUpdater)

    def test_barrier_updater_factory_invalid_type(self, caplog):
        """Test creating a barrier updater with an invalid type raises an exception."""
        caplog.set_level(logging.ERROR)
        
        mock_params = MockParameters()
        
        with pytest.raises(Exception):
            BarrierUpdaterFactory.create('non_existent_updater', mock_params)

    def test_barrier_updater_registry(self):
        """Test that the barrier updater is correctly registered in the registry."""
        registry = RegistryContainer()
        barrier_updater_registry = registry.barrier_updater

        # Check that 'default' is registered
        assert 'default' in barrier_updater_registry
        assert barrier_updater_registry['default'] is BarrierUpdater


class TestCustomBarrierUpdaterRegistration:
    def test_custom_barrier_updater_registration(self):
        """Test registering and using a custom barrier updater."""
        from simulation.core.registry.decorators import register

        @register(type="barrier_updater", name="custom_updater")
        class CustomBarrierUpdater(BarrierUpdaterBase):
            def __call__(self, xk: np.ndarray):
                # Custom implementation
                self.params.kB = 42.0

        # Verify registration
        registry = RegistryContainer()
        barrier_updater_registry = registry.barrier_updater

        assert 'custom_updater' in barrier_updater_registry
        assert barrier_updater_registry['custom_updater'] is CustomBarrierUpdater

        # Test creating through factory
        mock_params = MockParameters()
        factory = BarrierUpdaterFactory()
        custom_updater = factory.create('custom_updater', mock_params)
        assert isinstance(custom_updater, CustomBarrierUpdater)
        
        # Verify custom behavior
        custom_updater(np.zeros((2, 3)))
        assert mock_params.kB == 42.0
