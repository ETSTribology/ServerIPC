import numpy as np
import scipy.sparse as sp

from simulation.core.solvers.linear import LinearSolverFactory


class BenchmarkLinearSolvers:
    param_names = ["solver_type", "size", "density"]
    params = [
        ['default', 'chol', 'cg', 'lu', 'direct'],
        [1000, 5000],
        [0.001, 0.01, 0.1]
    ]

    def setup(self, solver_type, size, density):
        rng = np.random.default_rng()
        A = sp.random(size, size, density=density, format='csc', random_state=rng)
        A = A + A.T
        A += size * sp.eye(size)
        b = rng.random(size)
        dofs = np.arange(size)
        factory = LinearSolverFactory()
        self.solver = factory.create(type=solver_type, dofs=dofs)
        self.A = A
        self.b = b

    def time_solve(self, solver_type, size, density):
        self.solver.solve(self.A, self.b)
