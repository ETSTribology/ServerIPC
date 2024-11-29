import pytest
import numpy as np
import logging

from simulation.core.contact.barrier_initializer import (
    BarrierInitializerBase, 
    BarrierInitializer, 
    BarrierInitializerFactory
)
from simulation.core.parameters import ParametersBase


class MockParameters(ParametersBase):
    def __init__(self):
        self.mesh = np.array([[0, 0, 0], [1, 1, 1]])
        self.cmesh = np.array([[0, 0, 0], [1, 1, 1]])
        self.dhat = 0.01
        self.dmin = 0.001
        self.avgmass = 1.0
        self.bboxdiag = 1.0
        self.cconstraints = MockContactConstraints()
        self.barrier_potential = MockBarrierPotential()
        
        # Attributes to be set during initialization
        self.kB = None
        self.maxkB = None
        self.dprev = None


class MockContactConstraints:
    def compute_minimum_distance(self, cmesh, BX):
        return 0.1


class MockBarrierPotential:
    def __call__(self, cconstraints, cmesh, BX):
        pass

    def gradient(self, cconstraints, cmesh, BX):
        return np.zeros_like(BX)

    @property
    def barrier(self):
        return 1.0


class TestBarrierInitializerBase:
    def test_barrier_initializer_base_abstract(self):
        """Test that BarrierInitializerBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BarrierInitializerBase(MockParameters())  # type: ignore

    def test_base_class_interface(self):
        """Ensure the base class defines the required interface."""
        assert hasattr(BarrierInitializerBase, '__call__')
        assert hasattr(BarrierInitializerBase, '__init__')


class TestBarrierInitializer:
    @pytest.fixture
    def mock_params(self):
        """Create a mock parameters object."""
        return MockParameters()

    def test_barrier_initializer_call(self, mock_params, caplog):
        """Test the BarrierInitializer's __call__ method."""
        caplog.set_level(logging.INFO)
        
        # Prepare test data
        x = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        gU = np.zeros_like(x)
        gB = np.zeros_like(x)

        # Create BarrierInitializer
        barrier_initializer = BarrierInitializer(mock_params)
        barrier_initializer(x, gU, gB)

        # Check logging
        assert any('Initialized barrier stiffness' in record.message for record in caplog.records)

        # Check parameter updates
        assert mock_params.kB is not None
        assert mock_params.maxkB is not None
        assert mock_params.dprev is not None

    def test_barrier_initializer_with_different_inputs(self, mock_params):
        """Test BarrierInitializer with various input configurations."""
        barrier_initializer = BarrierInitializer(mock_params)
        
        # Different input configurations
        test_cases = [
            (np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), np.random.rand(2, 3), np.random.rand(2, 3)),
            (np.array([[1, 1, 1], [2, 2, 2]]), np.random.rand(2, 3), np.random.rand(2, 3)),
            (np.random.rand(2, 3), np.random.rand(2, 3), np.random.rand(2, 3))
        ]

        for x, gU, gB in test_cases:
            barrier_initializer(x, gU, gB)
            
            # Check parameter updates
            assert mock_params.kB is not None
            assert mock_params.maxkB is not None
            assert mock_params.dprev is not None


class TestBarrierInitializerFactory:
    def test_barrier_initializer_factory_create_default(self, caplog):
        """Test creating a default BarrierInitializer through the factory."""
        caplog.set_level(logging.ERROR)
        
        # Create mock parameters
        mock_params = MockParameters()

        # Create through factory
        barrier_initializer = BarrierInitializerFactory.create('default', mock_params)
        
        assert isinstance(barrier_initializer, BarrierInitializer)

    def test_barrier_initializer_factory_invalid_type(self, caplog):
        """Test creating a BarrierInitializer with an invalid type raises an exception."""
        caplog.set_level(logging.ERROR)
        
        mock_params = MockParameters()
        
        with pytest.raises(Exception):
            BarrierInitializerFactory.create('non_existent_barrier_initializer', mock_params)

    def test_barrier_initializer_registry(self):
        """Test that the BarrierInitializer is correctly registered in the registry."""
        registry_container = RegistryContainer()
        barrier_initializer_registry = registry_container.barrier_initializer

        # Check that 'default' is registered
        assert 'default' in barrier_initializer_registry
        assert barrier_initializer_registry['default'] is BarrierInitializer


class TestCustomBarrierInitializerRegistration:
    def test_custom_barrier_initializer_registration(self):
        """Test registering and using a custom BarrierInitializer."""
        from simulation.core.registry.decorators import register

        @register(type="barrier_initializer", name="custom_barrier_initializer")
        class CustomBarrierInitializer(BarrierInitializerBase):
            def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray) -> None:
                # Custom implementation
                pass

        # Verify registration
        registry_container = RegistryContainer()
        barrier_initializer_registry = registry_container.barrier_initializer

        assert 'custom_barrier_initializer' in barrier_initializer_registry
        assert barrier_initializer_registry['custom_barrier_initializer'] is CustomBarrierInitializer

        # Test creating through factory
        mock_params = MockParameters()
        factory = BarrierInitializerFactory()
        custom_barrier_initializer = factory.create('custom_barrier_initializer', mock_params)
        assert isinstance(custom_barrier_initializer, CustomBarrierInitializer)
