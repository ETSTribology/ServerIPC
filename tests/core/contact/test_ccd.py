import logging

import ipctk
import numpy as np
import pytest

from simulation.core.contact.ccd import CCD, CCDBase, CCDFactory, RegistryContainer
from simulation.core.parameters import ParametersBase


class MockParameters(ParametersBase):
    def __init__(self):
        self.mesh = np.array([[0, 0, 0], [1, 1, 1]])
        self.cmesh = np.array([[0, 0, 0], [1, 1, 1]])
        self.dmin = 0.01
        self.broad_phase_method = "default"
        self.cconstraints = ipctk.NormalCollisions()
        self.fconstraints = ipctk.TangentialCollisions()


class TestCCDBase:
    def test_ccd_base_abstract(self):
        """Test that CCDBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CCDBase(MockParameters())  # type: ignore

    def test_base_class_interface(self):
        """Ensure the base class defines the required interface."""
        assert callable(CCDBase)
        assert hasattr(CCDBase, "__init__")


class TestCCD:
    @pytest.fixture
    def mock_params(self):
        """Create a mock parameters object."""
        return MockParameters()

    def test_ccd_call(self, mock_params, caplog):
        """Test the CCD's __call__ method."""
        caplog.set_level(logging.DEBUG)

        # Prepare test data
        x = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        dx = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float64)

        # Create CCD
        ccd = CCD(mock_params)
        alpha = ccd(x, dx)

        # Check logging
        assert any("Computing CCD stepsize" in record.message for record in caplog.records)
        assert any("Computed CCD stepsize" in record.message for record in caplog.records)

        # Check return value and instance attribute
        assert 0 <= alpha <= 1.0
        assert ccd.alpha == alpha

    def test_ccd_with_different_inputs(self, mock_params):
        """Test CCD with various input configurations."""
        ccd = CCD(mock_params)

        # Different input configurations
        test_cases = [
            (np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), np.random.rand(2, 3)),
            (np.array([[1, 1, 1], [2, 2, 2]]), np.random.rand(2, 3)),
            (np.random.rand(2, 3), np.random.rand(2, 3)),
        ]

        for x, dx in test_cases:
            alpha = ccd(x, dx)
            assert 0 <= alpha <= 1.0
            assert ccd.alpha == alpha

    def test_ccd_with_different_broad_phase_methods(self, mock_params):
        """Test CCD with different broad phase methods."""
        broad_phase_methods = ["default", "brute_force", "spatial_hash"]

        for method in broad_phase_methods:
            mock_params.broad_phase_method = method
            ccd = CCD(mock_params)

            x = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
            dx = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float64)

            alpha = ccd(x, dx)
            assert 0 <= alpha <= 1.0


class TestCCDFactory:
    def test_ccd_factory_create_default(self, caplog):
        """Test creating a default CCD through the factory."""
        caplog.set_level(logging.ERROR)

        # Create mock parameters
        mock_params = MockParameters()

        # Create through factory
        ccd = CCDFactory.create("default", mock_params)

        assert isinstance(ccd, CCD)

    def test_ccd_factory_invalid_type(self, caplog):
        """Test creating a CCD with an invalid type raises an exception."""
        caplog.set_level(logging.ERROR)

        mock_params = MockParameters()

        with pytest.raises(Exception):
            CCDFactory.create("non_existent_ccd", mock_params)

    def test_ccd_registry(self):
        """Test that the CCD is correctly registered in the registry."""
        registry = RegistryContainer()
        ccd_registry = registry.ccd

        # Check that 'default' is registered
        assert "default" in ccd_registry
        assert ccd_registry["default"] is CCD


class TestCustomCCDRegistration:
    def test_custom_ccd_registration(self):
        """Test registering and using a custom CCD."""
        from simulation.core.registry.decorators import register

        @register(type="ccd", name="custom_ccd")
        class CustomCCD(CCDBase):
            def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
                # Custom implementation
                return 0.5

        # Verify registration
        registry = RegistryContainer()
        ccd_registry = registry.ccd

        assert "custom_ccd" in ccd_registry
        assert ccd_registry["custom_ccd"] is CustomCCD

        # Test creating through factory
        mock_params = MockParameters()
        factory = CCDFactory()
        custom_ccd = factory.create("custom_ccd", mock_params)
        assert isinstance(custom_ccd, CustomCCD)

        # Verify custom behavior
        x = np.zeros((2, 3))
        dx = np.ones((2, 3))
        assert custom_ccd(x, dx) == 0.5
