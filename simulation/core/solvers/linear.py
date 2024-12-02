import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Type

import numpy as np
import pbatoolkit as pbat
import scipy as sp
from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register
from simulation.core.utils.singleton import SingletonMeta
from scipy.sparse.linalg import cg, splu, spsolve

# Initialize logging
logger = logging.getLogger(__name__)


class LinearSolverBase(ABC):
    def __init__(self, dofs: np.ndarray, reg_param: float = None):
        self.dofs = dofs
        self.reg_param = reg_param
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def solve(
        self,
        A: sp.sparse.csc_matrix,
        b: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """Solve the linear system A x = b.

        :param A: Sparse matrix A.
        :param b: Right-hand side vector b.
        :param tol: Tolerance for iterative solvers.
        :param max_iter: Maximum number of iterations for iterative solvers.
        :return: Solution vector x.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


registry_container = RegistryContainer()
registry_container.add_registry("linear_solver", "simulation.core.solvers.linear.LinearSolverBase")


@register(type="linear_solver", name="default")
class LDLTSolver(LinearSolverBase):
    def solve(
        self,
        A: sp.sparse.csc_matrix,
        b: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> np.ndarray:
        try:
            dofs = self.dofs

            Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
            bd = b[dofs]

            self.logger.debug(
                f"LDLTSolver: Add shape after slicing: {Add.shape}, bd shape: {bd.shape}"
            )

            # Perform LDLT decomposition using pbatoolkit
            Add_ldlt = pbat.math.linalg.ldlt(Add)
            Add_ldlt.compute(Add)
            self.logger.debug("LDLT decomposition used successfully.")

            # Assemble full solution
            x = np.zeros_like(b)
            x[dofs] = Add_ldlt.solve(bd).squeeze()
            return x
        except Exception as e:
            self.logger.error(f"LDLT solver failed: {e}")
            raise


@register(type="linear_solver", name="chol")
class CholeskySolver(LinearSolverBase):
    def solve(
        self,
        A: sp.sparse.csc_matrix,
        b: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> np.ndarray:
        try:
            dofs = self.dofs
            Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
            bd = b[dofs]

            # Perform Cholesky decomposition using pbatoolkit
            Add_chol = pbat.math.linalg.chol(
                Add, solver=pbat.math.linalg.SolverBackend.SuiteSparse
            )
            Add_chol.compute(
                sp.sparse.tril(Add),
                pbat.math.linalg.Cholmod.SparseStorage.SymmetricLowerTriangular,
            )
            x_dofs = Add_chol.solve(bd).squeeze()
            self.logger.debug("Cholesky decomposition used successfully.")

            # Assemble full solution
            x = np.zeros_like(b)
            x[dofs] = x_dofs
            return x
        except Exception as e:
            self.logger.error(f"Cholesky solver failed: {e}")
            raise


@register(type="linear_solver", name="cg")
class CGSolver(LinearSolverBase):
    def solve(
        self,
        A: sp.sparse.csc_matrix,
        b: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> np.ndarray:
        try:
            dofs = self.dofs
            Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
            bd = b[dofs]

            # Solve using Conjugate Gradient method
            x_dofs, info = cg(Add, bd, tol=tol, maxiter=max_iter)
            if info == 0:
                self.logger.debug("CG method converged successfully.")
            else:
                self.logger.warning(f"CG did not converge. Info: {info}.")
                raise sp.sparse.linalg.ConvergenceError(
                    f"CG did not converge. Info: {info}"
                )

            # Assemble full solution
            x = np.zeros_like(b)
            x[dofs] = x_dofs
            return x
        except sp.sparse.linalg.ConvergenceError as e:
            self.logger.error(f"CG solver failed due to convergence issues: {e}")
            raise
        except Exception as e:
            self.logger.error(f"CG solver failed: {e}")
            raise


@register(type="linear_solver", name="lu")
class LUSolver(LinearSolverBase):
    def solve(
        self,
        A: sp.sparse.csc_matrix,
        b: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> np.ndarray:
        try:
            dofs = self.dofs
            Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
            bd = b[dofs]

            # Perform LU decomposition using SciPy
            lu = splu(Add)
            x_dofs = lu.solve(bd).squeeze()
            self.logger.debug("LU decomposition used successfully.")

            # Assemble full solution
            x = np.zeros_like(b)
            x[dofs] = x_dofs
            return x
        except Exception as e:
            self.logger.error(f"LU solver failed: {e}")
            raise


@register(type="linear_solver", name="direct")
class DirectSolver(LinearSolverBase):
    def solve(
        self,
        A: sp.sparse.csc_matrix,
        b: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> np.ndarray:
        try:
            dofs = self.dofs

            Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
            bd = b[dofs]

            self.logger.debug(
                f"DirectSolver: Add shape after slicing: {Add.shape}, bd shape: {bd.shape}"
            )

            # Solve directly using SciPy's sparse solver
            x_dofs = spsolve(Add, bd)
            self.logger.debug("Direct sparse solve used successfully.")

            # Assemble full solution
            x = np.zeros_like(b)
            x[dofs] = x_dofs
            return x
        except Exception as e:
            self.logger.error(f"Direct solver failed: {e}")
            raise


class LinearSolverFactory(metaclass=SingletonMeta):
    @lru_cache(maxsize=None)
    def get_class(self, type_lower: str) -> Type[LinearSolverBase]:
        """Retrieve and cache the LinearSolver class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.linear_solver.get(type_lower)

    def create(self, type: str, dofs: np.ndarray, **kwargs) -> LinearSolverBase:
        """Factory method to create a LinearSolver instance.

        Parameters
        ----------
        - type (str): Type of the linear solver (e.g., 'ldlt', 'cg', 'lu', 'direct').
        - dofs (np.ndarray): Degrees of freedom for the solver.
        - kwargs: Additional keyword arguments for the solver.

        Returns
        -------
        - An instance of LinearSolverBase.

        """
        type_lower = type.lower()
        try:
            # Retrieve the solver class from the registry
            solver_cls = self.get_class(type_lower)

            # Retrieve the constructor parameters for the solver class
            required_params = solver_cls.__init__.__code__.co_varnames
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in required_params
            }

            # Create an instance of the solver class
            solver_instance = solver_cls(dofs=dofs, **filtered_kwargs)
            logger.info(
                f"Solver '{type_lower}' created successfully using class '{solver_cls.__name__}'."
            )
            return solver_instance
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create solver '{type_lower}': {e}")
            raise RuntimeError(
                f"Error during solver initialization for method '{type_lower}': {e}"
            )