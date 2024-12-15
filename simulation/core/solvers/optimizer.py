import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, Optional, Type

import numpy as np
import scipy.sparse as sp

from simulation.core.solvers.line_search import (
    ArmijoLineSearch,
    BacktrackingLineSearch,
    LineSearchBase,
    LineSearchFactory,
    LineSearchMethod,
    ParallelLineSearch,
    StrongWolfeLineSearch,
    WolfeLineSearch,
)
from simulation.core.solvers.linear import LinearSolverBase, LinearSolverFactory
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Enumeration of supported optimizer types."""

    NEWTON = "newton"
    BFGS = "bfgs"
    LBFGS = "lbfgs"


class OptimizerBase(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable[[np.ndarray], sp.csc_matrix]] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform optimization to find the optimal x that minimizes the function f.

        Args:
            x0 (np.ndarray): Initial guess.
            f (Callable[[np.ndarray], float]): Objective function.
            grad (Callable[[np.ndarray], np.ndarray]): Gradient of the objective function.
            hess (Optional[Callable[[np.ndarray], sp.csc_matrix]]): Hessian of the objective function.
            callback (Optional[Callable[[np.ndarray], None]]): Callback function called after each iteration.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Optimized variables.
        """
        pass


class NewtonOptimizer(OptimizerBase):
    """Newton's method optimizer."""

    def __init__(
        self,
        lsolver: LinearSolverBase,
        line_searcher: LineSearchBase,
        alpha0_func: Callable[[np.ndarray, np.ndarray], float],
        maxiters: int = 10,
        rtol: float = 1e-5,
        reg_param: float = 1e-4,
        n_threads: int = 1,
    ):
        """
        Initialize the Newton optimizer.

        Args:
            lsolver (LinearSolverBase): Linear solver instance.
            line_searcher (LineSearchBase): Line searcher instance.
            alpha0_func (Callable[[np.ndarray, np.ndarray], float]): Function to determine step size.
            maxiters (int, optional): Maximum number of iterations. Defaults to 10.
            rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-5.
            reg_param (float, optional): Regularization parameter. Defaults to 1e-4.
            n_threads (int, optional): Number of threads for parallel execution. Defaults to 1.
        """
        self.lsolver = lsolver
        self.line_searcher = line_searcher
        self.alpha0_func = alpha0_func
        self.maxiters = maxiters
        self.rtol = rtol
        self.reg_param = reg_param
        self.n_threads = n_threads

    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], sp.csc_matrix],
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform Newton optimization.

        Args:
            x0 (np.ndarray): Initial guess.
            f (Callable[[np.ndarray], float]): Objective function.
            grad (Callable[[np.ndarray], np.ndarray]): Gradient of the objective function.
            hess (Callable[[np.ndarray], sp.csc_matrix]): Hessian of the objective function.
            callback (Optional[Callable[[np.ndarray], None]]): Callback function called after each iteration.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Optimized variables.
        """
        maxiters = kwargs.get("maxiters", 10)
        rtol = kwargs.get("rtol", 1e-5)
        xk = x0

        try:
            if self.n_threads == 1:
                # Sequential Newton's method
                for k in range(maxiters):
                    try:
                        gk = grad(xk)
                    except Exception as e:
                        logger.error(f"Error computing gradient at iteration {k}: {e}")
                        raise

                    gnorm = np.linalg.norm(gk, 1)
                    if gnorm < rtol:
                        logger.info(f"Converged at iteration {k} with gradient norm {gnorm}")
                        break

                    try:
                        Hk = hess(xk)
                    except Exception as e:
                        logger.error(f"Error computing Hessian at iteration {k}: {e}")
                        raise

                    try:
                        dx = self.lsolver.solve(Hk, -gk)
                    except Exception as e:
                        logger.error(f"Error solving linear system at iteration {k}: {e}")
                        raise

                    try:
                        alpha = self.line_searcher.search(
                            alpha0=self.alpha0_func(xk, dx), x=xk, dx=dx, f=f, grad=gk
                        )
                        xk = xk + alpha * dx
                        gk = grad(xk)
                    except Exception as e:
                        logger.error(f"Error updating variables at iteration {k}: {e}")
                        raise

                    if callback is not None:
                        try:
                            callback(xk)
                        except Exception as e:
                            logger.error(f"Error in callback at iteration {k}: {e}")
                    return xk
            else:
                # Parallel Newton's method
                Hk_cache = None
                with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                    for k in range(maxiters):
                        try:
                            future_grad = executor.submit(grad, xk)
                            gk = future_grad.result()
                        except Exception as e:
                            logger.error(f"Error computing gradient at iteration {k}: {e}")
                            raise

                        gnorm = np.linalg.norm(gk, np.inf)
                        if gnorm < rtol:
                            logger.info(f"Converged at iteration {k} with gradient norm {gnorm}")
                            break

                        try:
                            if Hk_cache is None:
                                future_hess = executor.submit(hess, xk)
                                Hk = future_hess.result()
                                Hk_cache = Hk
                            else:
                                Hk = Hk_cache
                        except Exception as e:
                            logger.error(f"Error computing Hessian at iteration {k}: {e}")
                            raise

                        try:
                            # Regularize Hessian
                            Hk_reg = Hk + self.reg_param * sp.sparse.eye(Hk.shape[0])
                            dx = self.lsolver.solve(Hk_reg, -gk)
                        except Exception as e:
                            logger.error(f"Error solving linear system at iteration {k}: {e}")
                            raise

                        try:
                            alpha = self.alpha0_func(xk, dx)
                            xk = xk + alpha * dx
                        except Exception as e:
                            logger.error(f"Error updating variables at iteration {k}: {e}")
                            raise

                        if np.linalg.norm(alpha * dx) > 1e-5:
                            Hk_cache = None  # Invalidate cache

                        if callback is not None:
                            try:
                                callback(xk)
                            except Exception as e:
                                logger.error(f"Error in callback at iteration {k}: {e}")

        except Exception as e:
            logger.critical(f"Critical failure in optimization: {e}")
            raise

        return xk


class BFGSOptimizer(OptimizerBase):
    """BFGS optimization algorithm."""

    def __init__(
        self,
        line_searcher: LineSearchBase,
        alpha0_func: Callable[[np.ndarray, np.ndarray], float],
        maxiters: int = 100,
        rtol: float = 1e-5,
    ):
        """
        Initialize the BFGS optimizer.

        Args:
            line_searcher (LineSearchBase): Line searcher instance.
            alpha0_func (Callable[[np.ndarray, np.ndarray], float]): Function to determine step size.
            maxiters (int, optional): Maximum number of iterations. Defaults to 100.
            rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-5.
        """
        if line_searcher is None:
            raise ValueError("line_searcher cannot be None for Newton optimizer")
        self.alpha0_func = alpha0_func
        self.maxiters = maxiters
        self.rtol = rtol
        logger = logging.getLogger(self.__class__.__name__)

    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable[[np.ndarray], sp.csc_matrix]] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform BFGS optimization.

        Args:
            x0 (np.ndarray): Initial guess.
            f (Callable[[np.ndarray], float]): Objective function.
            grad (Callable[[np.ndarray], np.ndarray]): Gradient of the objective function.
            hess (Optional[Callable[[np.ndarray], sp.csc_matrix]]): Hessian of the objective function.
            callback (Optional[Callable[[np.ndarray], None]]): Callback function called after each iteration.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Optimized variables.
        """
        maxiters = kwargs.get("maxiters", self.maxiters)
        rtol = kwargs.get("rtol", self.rtol)

        xk = x0.copy()
        n = len(xk)
        Hk = np.eye(n)
        gk = grad(xk)

        for k in range(maxiters):
            gnorm = np.linalg.norm(gk, np.inf)
            if gnorm < rtol:
                logger.info(
                    SimulationLogMessageCode.COMMAND_SUCCESS.details(
                        f"BFGS converged at iteration {k} with gradient norm {gnorm}"
                    )
                )
                break

            pk = -Hk @ gk
            alpha = self.alpha0_func(xk, pk)
            xk_new = xk + alpha * pk
            gk_new = grad(xk_new)
            sk = xk_new - xk
            yk = gk_new - gk
            sy = sk @ yk

            if sy > 1e-10:
                rho_k = 1.0 / sy
                Vk = np.eye(n) - rho_k * np.outer(sk, yk)
                Hk = Vk @ Hk @ Vk.T + rho_k * np.outer(sk, sk)
            else:
                logger.warning(
                    SimulationLogMessageCode.COMMAND_FAILED.details(
                        "Skipping update due to small sy in BFGS."
                    )
                )

            xk, gk = xk_new, gk_new
            if callback:
                callback(xk)

        return xk


class LBFGSOptimizer(OptimizerBase):
    """L-BFGS optimization algorithm."""

    def __init__(self, maxiters: int = 100, rtol: float = 1e-5, m: int = 10):
        """
        Initialize the L-BFGS optimizer.

        Args:
            maxiters (int, optional): Maximum number of iterations. Defaults to 100.
            rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-5.
            m (int, optional): Maximum number of stored vector pairs. Defaults to 10.
        """
        self.maxiters = maxiters
        self.rtol = rtol
        self.m = m
        logger = logging.getLogger(self.__class__.__name__)

    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable[[np.ndarray], sp.csc_matrix]] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform L-BFGS optimization.

        Args:
            x0 (np.ndarray): Initial guess.
            f (Callable[[np.ndarray], float]): Objective function.
            grad (Callable[[np.ndarray], np.ndarray]): Gradient of the objective function.
            hess (Optional[Callable[[np.ndarray], sp.csc_matrix]]): Hessian of the objective function.
            callback (Optional[Callable[[np.ndarray], None]]): Callback function called after each iteration.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Optimized variables.
        """
        maxiters = kwargs.get("maxiters", self.maxiters)
        rtol = kwargs.get("rtol", self.rtol)

        xk = x0.copy()
        gk = grad(xk)
        s_list, y_list, rho_list = [], [], []

        for k in range(maxiters):
            gnorm = np.linalg.norm(gk, np.inf)
            if gnorm < rtol:
                logger.info(
                    SimulationLogMessageCode.COMMAND_SUCCESS.details(
                        f"L-BFGS converged at iteration {k} with gradient norm {gnorm}"
                    )
                )
                break

            # Two-loop recursion
            q = gk.copy()
            alpha_list = []
            for si, yi, rhoi in reversed(list(zip(s_list, y_list, rho_list))):
                alpha_i = rhoi * np.dot(si, q)
                alpha_list.append(alpha_i)
                q -= alpha_i * yi

            r = q
            for si, yi, rhoi in zip(s_list, y_list, rho_list):
                beta = rhoi * np.dot(yi, r)
                r += si * (alpha_list.pop() - beta)

            pk = -r
            alpha = 1.0
            xk_new = xk + alpha * pk
            gk_new = grad(xk_new)

            sk = xk_new - xk
            yk = gk_new - gk
            sy = np.dot(sk, yk)

            if sy > 1e-10:
                rho_k = 1.0 / sy
                if len(s_list) == self.m:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)
                s_list.append(sk)
                y_list.append(yk)
                rho_list.append(rho_k)
            else:
                logger.warning(
                    SimulationLogMessageCode.COMMAND_FAILED.details(
                        "Skipping update due to small sy in L-BFGS."
                    )
                )

            xk, gk = xk_new, gk_new
            if callback:
                callback(xk)

        return xk


class OptimizerFactory(metaclass=SingletonMeta):
    """Factory class for creating optimizer instances."""

    _optimizer_mapping: Dict[OptimizerType, Type[OptimizerBase]] = {
        OptimizerType.NEWTON: NewtonOptimizer,
        OptimizerType.BFGS: BFGSOptimizer,
        OptimizerType.LBFGS: LBFGSOptimizer,
    }

    _line_search_mapping: Dict[str, Type[LineSearchBase]] = {
        LineSearchMethod.ARMIJO: ArmijoLineSearch,
        LineSearchMethod.WOLFE: WolfeLineSearch,
        LineSearchMethod.STRONG_WOLFE: StrongWolfeLineSearch,
        LineSearchMethod.PARALLEL: ParallelLineSearch,
        LineSearchMethod.BACKTRACKING: BacktrackingLineSearch,
    }

    def __init__(self):
        """Initialize the OptimizerFactory with necessary factories."""
        self.line_search_factory = LineSearchFactory()
        self.linear_solver_factory = LinearSolverFactory()

    def create(
        self,
        config: Dict[str, Any],
        f: Callable[[np.ndarray], float],
        grad_f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        linear_solver_config: Optional[Dict[str, Any]] = None,
        dofs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> OptimizerBase:
        """
        Create optimizer instance based on the configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            f (Callable[[np.ndarray], float]): Objective function.
            grad_f (Optional[Callable[[np.ndarray], np.ndarray]]): Gradient function.
            linear_solver_config (Optional[Dict[str, Any]]): Linear solver configuration.
            dofs (Optional[np.ndarray]): Degrees of freedom (indices) to consider in the solver.
            **kwargs: Additional keyword arguments.

        Returns:
            OptimizerBase: An instance of an OptimizerBase subclass.

        Raises:
            SimulationError: If the optimizer type is unknown or required configurations are missing.
        """
        optimizer_type_str = config.get("optimization", {}).get("solver")
        if not optimizer_type_str:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    "Optimizer type must be specified in config"
                )
            )
            raise SimulationError(
                SimulationErrorCode.OPTIMIZER_SETUP, "Optimizer type must be specified in config"
            )

        try:
            optimizer_type = OptimizerType(optimizer_type_str)
        except ValueError:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Unsupported optimizer type: {optimizer_type_str}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.OPTIMIZER_SETUP,
                f"Unsupported optimizer type: {optimizer_type_str}",
            )

        optimizer_class = self._optimizer_mapping.get(optimizer_type)
        if not optimizer_class:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"No optimizer found for type: {optimizer_type_str}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.OPTIMIZER_SETUP,
                f"No optimizer found for type: {optimizer_type_str}",
            )

        maxiters = config.get("optimization", {}).get("max_iterations", 100)
        rtol = config.get("optimization", {}).get("convergence_tolerance", 1e-5)

        line_search_config = config.get("line_search", {})
        line_searcher = (
            self.line_search_factory.create(line_search_config, f=f, grad_f=grad_f)
            if line_search_config
            else None
        )

        if optimizer_type == OptimizerType.NEWTON:
            if not linear_solver_config:
                logger.error(
                    SimulationLogMessageCode.COMMAND_FAILED.details(
                        "Linear solver configuration is required for Newton optimizer."
                    )
                )
                raise SimulationError(
                    SimulationErrorCode.LINEAR_SOLVER,
                    "Linear solver configuration is required for Newton optimizer.",
                )
            if dofs is None:
                logger.error(
                    SimulationLogMessageCode.COMMAND_FAILED.details(
                        "Degrees of freedom (dofs) are required for Newton optimizer."
                    )
                )
                raise SimulationError(
                    SimulationErrorCode.LINEAR_SOLVER,
                    "Degrees of freedom (dofs) are required for Newton optimizer.",
                )

            linear_solver = self.linear_solver_factory.create(linear_solver_config, dofs=dofs)

            reg_param = config.get("reg_param", 1e-4)
            n_threads = config.get("n_threads", 1)

            return optimizer_class(
                lsolver=linear_solver,
                line_searcher=line_searcher,
                alpha0_func=lambda x, dx: 1.0,
                maxiters=maxiters,
                rtol=rtol,
                reg_param=reg_param,
                n_threads=n_threads,
            )

        elif optimizer_type == OptimizerType.BFGS:
            return optimizer_class(
                line_searcher=line_searcher,
                alpha0_func=lambda x, dx: 1.0,
                maxiters=maxiters,
                rtol=rtol,
            )

        elif optimizer_type == OptimizerType.LBFGS:
            m = config.get("m", 10)
            return optimizer_class(maxiters=maxiters, rtol=rtol, m=m)

        else:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Unsupported optimizer type: {optimizer_type_str}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER,
                f"Unsupported optimizer type: {optimizer_type_str}",
            )
