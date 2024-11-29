import pytest
import numpy as np
import scipy.sparse as sp
import logging

from simulation.core.solvers.optimizer import (
    OptimizerBase,
    NewtonOptimizer,
    BFGSOptimizer,
    LBFGSOptimizer,
    OptimizerFactory,
    RegistryContainer
)
from simulation.core.solvers.line_search import LineSearchFactory
from simulation.core.solvers.linear import LinearSolverFactory


def test_objective_function(x):
    """A simple quadratic objective function for testing."""
    return np.sum(x**2)


def test_gradient_function(x):
    """Gradient of the test objective function."""
    return 2 * x


def test_hessian_function(x):
    """Hessian of the test objective function."""
    return 2 * sp.sparse.eye(len(x), format='csc')


def create_default_line_search():
    """Create a default line search for testing."""
    return LineSearchFactory.create('backtracking', 
                                    f=test_objective_function, 
                                    grad_f=test_gradient_function)


def create_default_linear_solver(dofs):
    """Create a default linear solver for testing."""
    return LinearSolverFactory.create('default', dofs)


def default_alpha0_func(x, dx):
    """Simple default step size function."""
    return 1.0


class TestOptimizerBase:
    def test_optimizer_base_abstract(self):
        """Test that OptimizerBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            base_optimizer = OptimizerBase()
            base_optimizer.optimize(
                x0=np.zeros(3), 
                f=test_objective_function, 
                grad=test_gradient_function, 
                hess=test_hessian_function
            )


class TestNewtonOptimizer:
    @pytest.fixture
    def newton_optimizer(self):
        """Create a default Newton Optimizer."""
        dofs = np.arange(3)
        linear_solver = create_default_linear_solver(dofs)
        line_search = create_default_line_search()
        
        return NewtonOptimizer(
            lsolver=linear_solver,
            line_searcher=line_search,
            alpha0_func=default_alpha0_func
        )

    def test_newton_optimizer_convergence(self, newton_optimizer, caplog):
        """Test Newton Optimizer convergence."""
        caplog.set_level(logging.INFO)
        
        x0 = np.array([10.0, 10.0, 10.0])
        result = newton_optimizer.optimize(
            x0=x0, 
            f=test_objective_function, 
            grad=test_gradient_function, 
            hess=test_hessian_function
        )
        
        # Check logging
        assert any('Converged' in record.message for record in caplog.records)
        
        # Check result is close to zero
        np.testing.assert_allclose(result, np.zeros_like(x0), rtol=1e-4)

    def test_newton_optimizer_callback(self, newton_optimizer):
        """Test Newton Optimizer with callback."""
        x0 = np.array([10.0, 10.0, 10.0])
        callback_results = []
        
        def callback(x):
            callback_results.append(x.copy())
        
        newton_optimizer.optimize(
            x0=x0, 
            f=test_objective_function, 
            grad=test_gradient_function, 
            hess=test_hessian_function,
            callback=callback
        )
        
        # Check callback was called
        assert len(callback_results) > 0
        assert all(np.linalg.norm(x) > 0 for x in callback_results)


class TestBFGSOptimizer:
    @pytest.fixture
    def bfgs_optimizer(self):
        """Create a default BFGS Optimizer."""
        line_search = create_default_line_search()
        
        return BFGSOptimizer(
            line_searcher=line_search,
            alpha0_func=default_alpha0_func
        )

    def test_bfgs_optimizer_convergence(self, bfgs_optimizer, caplog):
        """Test BFGS Optimizer convergence."""
        caplog.set_level(logging.INFO)
        
        x0 = np.array([10.0, 10.0, 10.0])
        result = bfgs_optimizer.optimize(
            x0=x0, 
            f=test_objective_function, 
            grad=test_gradient_function, 
            hess=test_hessian_function
        )
        
        # Check result is close to zero
        np.testing.assert_allclose(result, np.zeros_like(x0), rtol=1e-4)


class TestLBFGSOptimizer:
    @pytest.fixture
    def lbfgs_optimizer(self):
        """Create a default L-BFGS Optimizer."""
        return LBFGSOptimizer()

    def test_lbfgs_optimizer_convergence(self, lbfgs_optimizer, caplog):
        """Test L-BFGS Optimizer convergence."""
        caplog.set_level(logging.INFO)
        
        x0 = np.array([10.0, 10.0, 10.0])
        result = lbfgs_optimizer.optimize(
            x0=x0, 
            f=test_objective_function, 
            grad=test_gradient_function
        )
        
        # Check result is close to zero
        np.testing.assert_allclose(result, np.zeros_like(x0), rtol=1e-4)


class TestOptimizerFactory:
    def test_optimizer_factory_create_default(self):
        """Test creating optimizers through the factory."""
        optimizer_methods = [
            'default', 'newton', 
            'bfgs', 'lbfgs'
        ]
        
        x0 = np.array([10.0, 10.0, 10.0])
        
        for method in optimizer_methods:
            # Create optimizer
            optimizer = OptimizerFactory().create(method)
            
            # Validate optimizer instance
            assert hasattr(optimizer, 'optimize')
            
            # Solve the optimization problem
            result = optimizer.optimize(
                x0=x0, 
                f=test_objective_function, 
                grad=test_gradient_function, 
                hess=test_hessian_function
            )
            
            # Check result is close to zero
            np.testing.assert_allclose(result, np.zeros_like(x0), rtol=1e-4)

    def test_optimizer_factory_invalid_type(self):
        """Test creating an optimizer with an invalid type raises an exception."""
        with pytest.raises(Exception):
            OptimizerFactory().create('non_existent_optimizer')

    def test_optimizer_registry(self):
        """Test that optimizer methods are correctly registered."""
        registry = RegistryContainer()
        optimizer_registry = registry.optimizer

        # Check that methods are registered
        expected_methods = [
            'default', 'newton', 
            'bfgs', 'lbfgs'
        ]
        
        for method in expected_methods:
            assert method in optimizer_registry


class TestCustomOptimizerRegistration:
    def test_custom_optimizer_registration(self):
        """Test registering and using a custom optimizer method."""
        from simulation.core.registry.decorators import register

        @register(type="optimizer", name="custom_optimizer")
        class CustomOptimizer(OptimizerBase):
            def optimize(self, x0, f, grad, hess=None, callback=None, **kwargs):
                # Custom implementation
                return np.zeros_like(x0)

        # Verify registration
        registry = RegistryContainer()
        optimizer_registry = registry.optimizer

        assert 'custom_optimizer' in optimizer_registry
        assert optimizer_registry['custom_optimizer'] is CustomOptimizer

        # Test creating through factory
        custom_optimizer = OptimizerFactory().create('custom_optimizer')
        assert isinstance(custom_optimizer, CustomOptimizer)
        
        # Verify custom behavior
        x0 = np.array([10.0, 10.0, 10.0])
        result = custom_optimizer.optimize(
            x0=x0, 
            f=test_objective_function, 
            grad=test_gradient_function
        )
        np.testing.assert_array_equal(result, np.zeros_like(x0))
