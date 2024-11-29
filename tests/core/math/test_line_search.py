import pytest
import numpy as np
import logging

from simulation.core.solvers.line_search import (
    LineSearchBase,
    BacktrackingLineSearch,
    WolfeLineSearch,
    StrongWolfeLineSearch,
    ParallelLineSearch,
    LineSearchFactory,
    RegistryContainer
)


def test_objective_function(x):
    """A simple quadratic objective function for testing."""
    return np.sum(x**2)


def test_gradient_function(x):
    """Gradient of the test objective function."""
    return 2 * x


class TestLineSearchBase:
    def test_line_search_base_abstract(self):
        """Test that LineSearchBase cannot be instantiated directly."""
        with pytest.raises(NotImplementedError):
            base_search = LineSearchBase(test_objective_function)
            base_search.search(1.0, np.zeros(3), np.ones(3), np.ones(3))


class TestBacktrackingLineSearch:
    @pytest.fixture
    def backtracking_search(self):
        """Create a BacktrackingLineSearch instance."""
        return BacktrackingLineSearch(
            f=test_objective_function,
            grad_f=test_gradient_function
        )

    def test_backtracking_line_search(self, backtracking_search, caplog):
        """Test Backtracking Line Search."""
        caplog.set_level(logging.DEBUG)
        
        x = np.array([2.0, 2.0, 2.0])
        dx = np.array([-1.0, -1.0, -1.0])
        g = test_gradient_function(x)
        
        alpha = backtracking_search.search(1.0, x, dx, g)
        
        # Check logging
        assert any('Running Backtracking Line Search' in record.message for record in caplog.records)
        
        # Validate alpha
        assert 0 < alpha <= 1.0

    def test_backtracking_near_zero_gradient(self, backtracking_search, caplog):
        """Test Backtracking Line Search with near-zero gradient."""
        caplog.set_level(logging.DEBUG)
        
        x = np.zeros(3)
        dx = np.zeros(3)
        g = np.zeros(3)
        
        alpha = backtracking_search.search(1.0, x, dx, g)
        
        # Check logging
        assert any('Directional derivative is near zero' in record.message for record in caplog.records)
        
        # Expect zero step size
        assert alpha == 0.0


class TestWolfeLineSearch:
    @pytest.fixture
    def wolfe_search(self):
        """Create a WolfeLineSearch instance."""
        return WolfeLineSearch(
            f=test_objective_function,
            grad_f=test_gradient_function
        )

    def test_wolfe_line_search(self, wolfe_search, caplog):
        """Test Wolfe Line Search."""
        caplog.set_level(logging.DEBUG)
        
        x = np.array([2.0, 2.0, 2.0])
        dx = np.array([-1.0, -1.0, -1.0])
        g = test_gradient_function(x)
        
        alpha = wolfe_search.search(1.0, x, dx, g)
        
        # Check logging
        assert any('Running Wolfe Line Search' in record.message for record in caplog.records)
        
        # Validate alpha
        assert 0 < alpha <= 1.0


class TestStrongWolfeLineSearch:
    @pytest.fixture
    def strong_wolfe_search(self):
        """Create a StrongWolfeLineSearch instance."""
        return StrongWolfeLineSearch(
            f=test_objective_function,
            grad_f=test_gradient_function
        )

    def test_strong_wolfe_line_search(self, strong_wolfe_search, caplog):
        """Test Strong Wolfe Line Search."""
        caplog.set_level(logging.DEBUG)
        
        x = np.array([2.0, 2.0, 2.0])
        dx = np.array([-1.0, -1.0, -1.0])
        g = test_gradient_function(x)
        
        alpha = strong_wolfe_search.search(1.0, x, dx, g)
        
        # Check logging
        assert any('Running Strong Wolfe Line Search' in record.message for record in caplog.records)
        
        # Validate alpha
        assert 0 < alpha <= 1.0


class TestParallelLineSearch:
    @pytest.fixture
    def parallel_search(self):
        """Create a ParallelLineSearch instance."""
        return ParallelLineSearch(
            f=test_objective_function,
            grad_f=test_gradient_function
        )

    def test_parallel_line_search(self, parallel_search, caplog):
        """Test Parallel Line Search."""
        caplog.set_level(logging.DEBUG)
        
        x = np.array([2.0, 2.0, 2.0])
        dx = np.array([-1.0, -1.0, -1.0])
        g = test_gradient_function(x)
        
        alpha = parallel_search.search(1.0, x, dx, g)
        
        # Check logging
        assert any('Running Parallel Line Search' in record.message for record in caplog.records)
        
        # Validate alpha
        assert 0 < alpha <= 1.0


class TestLineSearchFactory:
    def test_line_search_factory_create_default(self):
        """Test creating line search methods through the factory."""
        line_search_methods = [
            'backtracking', 
            'wolfe', 
            'strong_wolfe', 
            'parallel'
        ]
        
        for method in line_search_methods:
            line_search = LineSearchFactory.create(
                method, 
                f=test_objective_function, 
                grad_f=test_gradient_function
            )
            
            # Validate line search instance
            assert hasattr(line_search, 'search')
            assert hasattr(line_search, 'f')
            assert hasattr(line_search, 'grad_f')

    def test_line_search_factory_invalid_type(self):
        """Test creating a line search with an invalid type raises an exception."""
        with pytest.raises(Exception):
            LineSearchFactory.create(
                'non_existent_line_search', 
                f=test_objective_function
            )

    def test_line_search_registry(self):
        """Test that line search methods are correctly registered."""
        registry = RegistryContainer()
        line_search_registry = registry.line_search

        # Check that methods are registered
        expected_methods = [
            'backtracking', 
            'wolfe', 
            'strong_wolfe', 
            'parallel'
        ]
        
        for method in expected_methods:
            assert method in line_search_registry


class TestCustomLineSearchRegistration:
    def test_custom_line_search_registration(self):
        """Test registering and using a custom line search method."""
        from simulation.core.registry.decorators import register

        @register(type="line_search", name="custom_line_search")
        class CustomLineSearch(LineSearchBase):
            def search(self, alpha0, xk, dx, gk):
                # Custom implementation
                return 0.5

        # Verify registration
        registry = RegistryContainer()
        line_search_registry = registry.line_search

        assert 'custom_line_search' in line_search_registry
        assert line_search_registry['custom_line_search'] is CustomLineSearch

        # Test creating through factory
        custom_line_search = LineSearchFactory.create(
            'custom_line_search', 
            f=test_objective_function
        )
        assert isinstance(custom_line_search, CustomLineSearch)
        
        # Verify custom behavior
        x = np.zeros(3)
        dx = np.ones(3)
        g = np.ones(3)
        assert custom_line_search.search(1.0, x, dx, g) == 0.5
