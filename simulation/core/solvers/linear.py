import logging
from abc import ABC, abstractmethod
from typing import Type, Dict, Any

import numpy as np
import pbatoolkit as pbat
import scipy as sp
from scipy.sparse.linalg import cg, splu, spsolve

from simulation.core.utils.singleton import SingletonMeta

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
            Add_chol = pbat.math.linalg.chol(Add, solver=pbat.math.linalg.SolverBackend.SuiteSparse)
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
                raise sp.sparse.linalg.ConvergenceError(f"CG did not converge. Info: {info}")

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
    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a linear solver instance based on the configuration.

        Args:
            config: A dictionary containing the linear solver configuration.

        Returns:
            An instance of the linear solver class.

        Raises:
            ValueError: 
        """
        logger.info("Creating linear solver...")
        linear_solver_config = config.get("linear_solver", {})
        linear_solver_type = linear_solver_config.get("type", "default").lower()

        if linear_solver_type not in LinearSolverFactory._instances:
            if linear_solver_type == "default":
                linear_solver_instance = LinearSolver()
            elif linear_solver_type == "ldlt":
                linear_solver_instance = LDLTSolver()
            elif linear_solver_type == "cg":
                linear_solver_instance = CGSolver()
            elif linear_solver_type == "lu":
                linear_solver_instance = LUSolver()
            elif linear_solver_type == "direct":
                linear_solver_instance = DirectSolver()
            else:
                raise ValueError(f"Unknown linear solver type: {linear_solver_type}")

            LinearSolverFactory._instances[linear_solver_type] = linear_solver_instance

        return LinearSolverFactory._instances[linear_solver_type]