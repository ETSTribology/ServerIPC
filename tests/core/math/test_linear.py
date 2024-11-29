import pytest
import numpy as np
import scipy.sparse as sp
import logging

from simulation.core.solvers.linear import (
    LinearSolverBase,
    LDLTSolver,
    CholeskySolver,
    CGSolver,
    LUSolver,
    DirectSolver,
    LinearSolverFactory,
    RegistryContainer
)


def create_sparse_matrix(n=10):
    """Create a symmetric positive definite sparse matrix."""
    # Create a diagonal matrix with random positive values
    diag = np.random.rand(n) + 1.0
    A = sp.diags(diag, format='csc')
    return A


class TestLinearSolverBase:
    def test_linear_solver_base_abstract(self):
        """Test that LinearSolverBase cannot be instantiated directly."""
        dofs = np.arange(10)
        with pytest.raises(NotImplementedError):
            base_solver = LinearSolverBase(dofs)
            base_solver.solve(sp.csc_matrix((10, 10)), np.zeros(10))


class TestLinearSolvers:
    @pytest.fixture
    def matrix_and_vector(self):
        """Create a test matrix and vector."""
        n = 10
        A = create_sparse_matrix(n)
        x_true = np.random.rand(n)
        b = A.dot(x_true)
        dofs = np.arange(n)
        return A, b, x_true, dofs

    @pytest.mark.parametrize("solver_class,solver_name", [
        (LDLTSolver, 'ldlt'),
        (CholeskySolver, 'chol'),
        (CGSolver, 'cg'),
        (LUSolver, 'lu'),
        (DirectSolver, 'direct')
    ])
    def test_linear_solvers(self, matrix_and_vector, solver_class, solver_name, caplog):
        """Test various linear solvers."""
        caplog.set_level(logging.DEBUG)
        
        A, b, x_true, dofs = matrix_and_vector
        
        # Create solver
        solver = solver_class(dofs)
        
        # Solve the system
        x_solved = solver.solve(A, b)
        
        # Check logging
        assert any(f'{solver_name.upper()}' in record.message for record in caplog.records)
        
        # Check solution
        np.testing.assert_allclose(x_solved[dofs], x_true, rtol=1e-5)

    def test_solver_with_different_matrix_sizes(self):
        """Test solvers with different matrix sizes."""
        solver_classes = [LDLTSolver, CholeskySolver, CGSolver, LUSolver, DirectSolver]
        
        sizes = [5, 10, 20]
        for size in sizes:
            for solver_class in solver_classes:
                # Create matrix and vector
                A = create_sparse_matrix(size)
                x_true = np.random.rand(size)
                b = A.dot(x_true)
                dofs = np.arange(size)
                
                # Create and solve
                solver = solver_class(dofs)
                x_solved = solver.solve(A, b)
                
                # Check solution
                np.testing.assert_allclose(x_solved[dofs], x_true, rtol=1e-5)


class TestLinearSolverFactory:
    def test_linear_solver_factory_create_default(self):
        """Test creating linear solvers through the factory."""
        linear_solver_methods = [
            'default', 'ldlt', 
            'chol', 'cg', 
            'lu', 'direct'
        ]
        
        dofs = np.arange(10)
        A = create_sparse_matrix(10)
        b = np.random.rand(10)
        
        for method in linear_solver_methods:
            linear_solver = LinearSolverFactory.create(method, dofs)
            
            # Validate linear solver instance
            assert hasattr(linear_solver, 'solve')
            
            # Solve the system
            x_solved = linear_solver.solve(A, b)
            assert x_solved.shape == b.shape

    def test_linear_solver_factory_invalid_type(self):
        """Test creating a linear solver with an invalid type raises an exception."""
        dofs = np.arange(10)
        with pytest.raises(Exception):
            LinearSolverFactory.create('non_existent_linear_solver', dofs)

    def test_linear_solver_registry(self):
        """Test that linear solver methods are correctly registered."""
        registry = RegistryContainer()
        linear_solver_registry = registry.linear_solver

        # Check that methods are registered
        expected_methods = [
            'default', 'ldlt', 
            'chol', 'cg', 
            'lu', 'direct'
        ]
        
        for method in expected_methods:
            assert method in linear_solver_registry


class TestCustomLinearSolverRegistration:
    def test_custom_linear_solver_registration(self):
        """Test registering and using a custom linear solver method."""
        from simulation.core.registry.decorators import register

        @register(type="linear_solver", name="custom_linear_solver")
        class CustomLinearSolver(LinearSolverBase):
            def solve(self, A, b, tol=1e-8, max_iter=1000):
                # Custom implementation
                return np.ones_like(b)

        # Verify registration
        registry = RegistryContainer()
        linear_solver_registry = registry.linear_solver

        assert 'custom_linear_solver' in linear_solver_registry
        assert linear_solver_registry['custom_linear_solver'] is CustomLinearSolver

        # Test creating through factory
        dofs = np.arange(10)
        custom_linear_solver = LinearSolverFactory.create('custom_linear_solver', dofs)
        assert isinstance(custom_linear_solver, CustomLinearSolver)
        
        # Verify custom behavior
        A = create_sparse_matrix(10)
        b = np.random.rand(10)
        result = custom_linear_solver.solve(A, b)
        np.testing.assert_array_equal(result, np.ones_like(b))
