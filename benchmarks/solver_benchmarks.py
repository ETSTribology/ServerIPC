import numpy as np

from simulation.core.solvers.line_search import LineSearch
from simulation.core.solvers.linear import LinearSolverBase
from simulation.core.solvers.optimizer import OptimizerBase


class LinearSolverBenchmarks:
    """Benchmarks for linear solvers."""

    def setup(self):
        # Generate test matrices of different sizes
        self.small_matrix = np.random.rand(100, 100)
        self.medium_matrix = np.random.rand(500, 500)
        self.large_matrix = np.random.rand(1000, 1000)

        self.small_vector = np.random.rand(100)
        self.medium_vector = np.random.rand(500)
        self.large_vector = np.random.rand(1000)

    def time_solve_small_matrix(self):
        """Benchmark solving a small linear system."""
        solver = LinearSolverBase()
        solver.solve(self.small_matrix, self.small_vector)

    def time_solve_medium_matrix(self):
        """Benchmark solving a medium-sized linear system."""
        solver = LinearSolverBase()
        solver.solve(self.medium_matrix, self.medium_vector)

    def time_solve_large_matrix(self):
        """Benchmark solving a large linear system."""
        solver = LinearSolverBase()
        solver.solve(self.large_matrix, self.large_vector)


class OptimizerBenchmarks:
    """Benchmarks for optimization algorithms."""

    def setup(self):
        # Create test optimization problems
        def quadratic_function(x):
            return np.sum(x**2)

        def rosenbrock_function(x):
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

        self.quadratic_problem = quadratic_function
        self.rosenbrock_problem = rosenbrock_function
        self.initial_guess = np.random.rand(10)

    def time_optimize_quadratic(self):
        """Benchmark quadratic function optimization."""
        optimizer = OptimizerBase()
        optimizer.optimize(self.quadratic_problem, self.initial_guess)

    def time_optimize_rosenbrock(self):
        """Benchmark Rosenbrock function optimization."""
        optimizer = OptimizerBase()
        optimizer.optimize(self.rosenbrock_problem, self.initial_guess)


class LineSearchBenchmarks:
    """Benchmarks for line search algorithms."""

    def setup(self):
        self.x0 = np.random.rand(10)
        self.direction = np.random.rand(10)
        self.alpha = 1.0

    def test_objective(self, x):
        """Test objective function."""
        return np.sum(x**2)

    def test_gradient(self, x):
        """Test gradient function."""
        return 2 * x

    def time_line_search(self):
        """Benchmark line search algorithm."""
        line_search = LineSearch()
        line_search.search(
            self.test_objective, self.test_gradient, self.x0, self.direction, self.alpha
        )


class MemorySolverBenchmarks:
    """Memory usage benchmarks for solvers."""

    def complex_function(self, x):
        """Complex objective function for memory testing."""
        return np.sum(np.sin(x) ** 2 + np.cos(x) ** 2)

    def mem_linear_solver(self):
        """Measure memory of linear solver with large matrix."""
        matrix = np.random.rand(2000, 2000)
        vector = np.random.rand(2000)
        solver = LinearSolverBase()
        solver.solve(matrix, vector)

    def mem_optimizer(self):
        """Measure memory of optimizer with complex problem."""
        x0 = np.random.rand(1000)
        optimizer = OptimizerBase()
        optimizer.optimize(self.complex_function, x0)
