import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

import numpy as np
import pbatoolkit as pbat
import scipy as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg, splu, spsolve

from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

# Initialize logging
logger = logging.getLogger(__name__)


class SolverMethod(Enum):
    """Enumeration of supported linear solver methods."""

    DEFAULT = "default"
    LDLT = "ldlt"
    CHOLESKY = "cholesky"
    CG = "cg"
    LU = "lu"
    DIRECT = "direct"


@dataclass
class LinearSolverConfig:
    """Configuration for linear solvers."""

    type: SolverMethod
    reg_param: Optional[float] = None
    max_iter: int = 1000
    tol: float = 1e-8


class LinearSolverBase(ABC):
    """Abstract base class for linear solvers."""

    def __init__(self, dofs: np.ndarray, config: Optional[LinearSolverConfig] = None):
        """
        Initialize the linear solver.

        Args:
            dofs (np.ndarray): Degrees of freedom (indices) to consider in the solver.
            config (Optional[LinearSolverConfig]): Configuration parameters for the solver.
        """
        self.dofs = dofs
        self.config = config or LinearSolverConfig(type=SolverMethod.DEFAULT)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the linear system A x = b.

        Args:
            A (csc_matrix): Sparse matrix A.
            b (np.ndarray): Right-hand side vector b.

        Returns:
            np.ndarray: Solution vector x.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class LDLTSolver(LinearSolverBase):
    """LDLT decomposition-based linear solver."""

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the linear system using LDLT decomposition.

        Args:
            A (csc_matrix): Sparse matrix A.
            b (np.ndarray): Right-hand side vector b.

        Returns:
            np.ndarray: Solution vector x.
        """
        try:
            dofs = self.dofs

            # Slice the matrix and vector based on degrees of freedom
            Add = A[self.dofs, :][:, self.dofs].tocsc()
            bd = b[self.dofs]

            self.logger.debug(
                f"LDLTSolver: Add shape after slicing: {Add.shape}, bd shape: {bd.shape}"
            )

            # Perform LDLT decomposition using pbatoolkit
            Add_ldlt = pbat.math.linalg.ldlt(Add)
            Add_ldlt.compute(Add)
            self.logger.debug("LDLT decomposition used successfully.")

            # Assemble full solution
            x = np.zeros_like(b)
            x[self.dofs] = Add_ldlt.solve(bd).squeeze()
            return x
        except Exception as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(f"LDLT solver failed: {e}")
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER, "LDLT solver failed", details=str(e)
            )


class CholeskySolver(LinearSolverBase):
    """Cholesky decomposition-based linear solver."""

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the linear system using Cholesky decomposition.

        Args:
            A (csc_matrix): Sparse matrix A.
            b (np.ndarray): Right-hand side vector b.

        Returns:
            np.ndarray: Solution vector x.
        """
        try:
            dofs = self.dofs
            Add = A[self.dofs, :][:, self.dofs].tocsc()
            bd = b[self.dofs]

            self.logger.debug(
                f"CholeskySolver: Add shape after slicing: {Add.shape}, bd shape: {bd.shape}"
            )

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
            x[self.dofs] = x_dofs
            return x
        except Exception as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(f"Cholesky solver failed: {e}")
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER, "Cholesky solver failed", details=str(e)
            )


class CGSolver(LinearSolverBase):
    """Conjugate Gradient method-based linear solver."""

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the linear system using Conjugate Gradient method.

        Args:
            A (csc_matrix): Sparse matrix A.
            b (np.ndarray): Right-hand side vector b.

        Returns:
            np.ndarray: Solution vector x.
        """
        try:
            dofs = self.dofs
            Add = A[self.dofs, :][:, self.dofs].tocsc()
            bd = b[self.dofs]

            self.logger.debug(
                f"CGSolver: Add shape after slicing: {Add.shape}, bd shape: {bd.shape}"
            )

            # Solve using Conjugate Gradient method
            x_dofs, info = cg(Add, bd, tol=self.config.tol, maxiter=self.config.max_iter)
            if info == 0:
                self.logger.debug("CG method converged successfully.")
            else:
                self.logger.warning(
                    SimulationLogMessageCode.COMMAND_FAILED.details(
                        f"CG did not converge. Info: {info}."
                    )
                )
                raise SimulationError(
                    SimulationErrorCode.LINEAR_SOLVER, f"CG did not converge. Info: {info}"
                )

            # Assemble full solution
            x = np.zeros_like(b)
            x[self.dofs] = x_dofs
            return x
        except Exception as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(f"CG solver failed: {e}")
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER, "CG solver failed", details=str(e)
            )


class LUSolver(LinearSolverBase):
    """LU decomposition-based linear solver."""

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the linear system using LU decomposition.

        Args:
            A (csc_matrix): Sparse matrix A.
            b (np.ndarray): Right-hand side vector b.

        Returns:
            np.ndarray: Solution vector x.
        """
        try:
            dofs = self.dofs
            Add = A[self.dofs, :][:, self.dofs].tocsc()
            bd = b[self.dofs]

            self.logger.debug(
                f"LUSolver: Add shape after slicing: {Add.shape}, bd shape: {bd.shape}"
            )

            # Perform LU decomposition using SciPy
            lu = splu(Add)
            x_dofs = lu.solve(bd).squeeze()
            self.logger.debug("LU decomposition used successfully.")

            # Assemble full solution
            x = np.zeros_like(b)
            x[self.dofs] = x_dofs
            return x
        except Exception as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(f"LU solver failed: {e}")
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER, "LU solver failed", details=str(e)
            )


class DirectSolver(LinearSolverBase):
    """Direct sparse solver using SciPy."""

    def solve(
        self,
        A: csc_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the linear system using direct sparse solver.

        Args:
            A (csc_matrix): Sparse matrix A.
            b (np.ndarray): Right-hand side vector b.

        Returns:
            np.ndarray: Solution vector x.
        """
        try:
            dofs = self.dofs

            Add = A[self.dofs, :][:, self.dofs].tocsc()
            bd = b[self.dofs]

            self.logger.debug(
                f"DirectSolver: Add shape after slicing: {Add.shape}, bd shape: {bd.shape}"
            )

            # Solve directly using SciPy's sparse solver
            x_dofs = spsolve(Add, bd)
            self.logger.debug("Direct sparse solve used successfully.")

            # Assemble full solution
            x = np.zeros_like(b)
            x[self.dofs] = x_dofs
            return x
        except Exception as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(f"Direct solver failed: {e}")
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER, "Direct solver failed", details=str(e)
            )


class LinearSolverFactory(metaclass=SingletonMeta):
    """Factory for creating linear solver instances."""

    _solver_mapping: Dict[SolverMethod, Type[LinearSolverBase]] = {
        SolverMethod.LDLT: LDLTSolver,
        SolverMethod.CHOLESKY: CholeskySolver,
        SolverMethod.CG: CGSolver,
        SolverMethod.LU: LUSolver,
        SolverMethod.DIRECT: DirectSolver,
    }

    @staticmethod
    def create_config(config: Dict[str, Any]) -> LinearSolverConfig:
        """
        Create linear solver configuration from a dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            LinearSolverConfig: Parsed linear solver configuration.

        Raises:
            SimulationError: If the solver type is unknown.
        """
        try:
            linear_solver_config = config.get("linear", {})
            solver_type_str = linear_solver_config.get("solver", "direct").lower()
            solver_type = SolverMethod(solver_type_str)

            return LinearSolverConfig(
                type=solver_type,
                reg_param=linear_solver_config.get("regularization", 1e-6),
                max_iter=linear_solver_config.get("max_iterations", 1000),
                tol=linear_solver_config.get("tolerance", 1e-8),
            )
        except ValueError as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Unknown linear solver type: {solver_type_str}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER,
                f"Unknown linear solver type: {solver_type_str}",
                details=str(e),
            )

    @staticmethod
    def create(
        config: Dict[str, Any],
        dofs: np.ndarray,
    ) -> LinearSolverBase:
        """
        Create a linear solver instance based on the configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            dofs (np.ndarray): Degrees of freedom (indices) to consider in the solver.

        Returns:
            LinearSolverBase: An instance of a LinearSolverBase subclass.

        Raises:
            SimulationError: If the solver type is unknown.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating linear solver..."))
        ls_config = LinearSolverFactory.create_config(config)

        if ls_config.type in LinearSolverFactory._solver_mapping:
            solver_class = LinearSolverFactory._solver_mapping[ls_config.type]
        elif ls_config.type == SolverMethod.DEFAULT:
            # Define a default solver, e.g., DirectSolver
            solver_class = DirectSolver
            logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    "Using DirectSolver as the default linear solver."
                )
            )
        else:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Unknown linear solver type: {ls_config.type}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER, f"Unknown linear solver type: {ls_config.type}"
            )

        # Check if an instance already exists to enforce singleton behavior
        key = (ls_config.type, tuple(dofs), ls_config.reg_param)
        if not hasattr(LinearSolverFactory, "_instances"):
            LinearSolverFactory._instances = {}
        if key not in LinearSolverFactory._instances:
            solver_instance = solver_class(dofs=dofs, config=ls_config)
            LinearSolverFactory._instances[key] = solver_instance
            logger.debug(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Created new instance of {solver_class.__name__}."
                )
            )
        else:
            solver_instance = LinearSolverFactory._instances[key]
            logger.debug(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Retrieved existing instance of {solver_class.__name__}."
                )
            )

        return solver_instance
