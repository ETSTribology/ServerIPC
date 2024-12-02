import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Optional, Type, Union

import numpy as np
import scipy as sp
import torch
from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register
from simulation.core.solvers.line_search import LineSearchFactory
from simulation.core.solvers.linear import LinearSolverFactory
from simulation.core.utils.singleton import SingletonMeta

logger = logging.getLogger(__name__)


class OptimizerBase(ABC):
    @abstractmethod
    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], sp.sparse.csc_matrix],
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform optimization to find the optimal x that minimizes the function f.

        Parameters
        ----------
        - x0: Initial guess for the solution.
        - f: Objective function to minimize.
        - grad: Gradient of the objective function.
        - hess: Hessian of the objective function.
        - callback: Optional callback function called after each iteration.
        - kwargs: Additional keyword arguments.

        Returns
        -------
        - The optimized solution vector.

        """
        pass


registry_container = RegistryContainer()
registry_container.add_registry("optimizer", "simulation.core.solvers.optimizer.OptimizerBase")


@register(type="optimizer", name="default")
class NewtonOptimizer(OptimizerBase):
    def __init__(
        self,
        lsolver: "LinearSolverBase",
        line_searcher: "LineSearchBase",
        alpha0_func: Callable[[np.ndarray, np.ndarray], float],
        maxiters: int = 10,
        rtol: float = 1e-5,
        reg_param: float = 1e-4,
        n_threads: int = 1,
    ):
        """Initialize the Newton Optimizer.

        Parameters
        ----------
        - lsolver: Linear solver instance.
        - line_searcher: Line search method instance.
        - alpha0_func: Function to compute initial step size.
        - maxiters: Maximum number of iterations.
        - rtol: Relative tolerance for convergence.
        - reg_param: Regularization parameter for Hessian.
        - n_threads: Number of threads for parallel execution.

        """
        self.lsolver = lsolver
        self.line_searcher = line_searcher
        self.alpha0_func = alpha0_func
        self.maxiters = maxiters
        self.rtol = rtol
        self.reg_param = reg_param
        self.n_threads = n_threads
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], sp.sparse.csc_matrix],
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform Newton optimization.

        Parameters
        ----------
        - x0: Initial guess.
        - f: Objective function.
        - grad: Gradient function.
        - hess: Hessian function.
        - callback: Optional callback after each iteration.
        - kwargs: Additional keyword arguments.

        Returns
        -------
        - Optimized solution vector.

        """
        maxiters = kwargs.get("maxiters", self.maxiters)
        rtol = kwargs.get("rtol", self.rtol)
        xk = x0.copy()

        if self.n_threads == 1:
            # Sequential Newton's method
            for k in range(maxiters):
                gk = grad(xk)
                gnorm = np.linalg.norm(gk, np.inf)
                if gnorm < rtol:
                    self.logger.info(
                        f"Converged at iteration {k} with gradient norm {gnorm}"
                    )
                    break

                Hk = hess(xk)
                # Regularize Hessian
                Hk_reg = Hk + self.reg_param * sp.sparse.eye(Hk.shape[0])
                dx = self.lsolver.solve(Hk_reg, -gk)
                alpha = self.alpha0_func(xk, dx)
                xk = xk + alpha * dx

                if callback is not None:
                    callback(xk)

        else:
            # Parallel Newton's method
            Hk_cache = None
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                for k in range(maxiters):
                    # Compute gradient and Hessian in parallel
                    future_grad = executor.submit(grad, xk)
                    gk = future_grad.result()
                    gnorm = np.linalg.norm(gk, np.inf)
                    if gnorm < rtol:
                        self.logger.info(
                            f"Converged at iteration {k} with gradient norm {gnorm}"
                        )
                        break

                    if Hk_cache is None:
                        future_hess = executor.submit(hess, xk)
                        Hk = future_hess.result()
                        Hk_cache = Hk
                    else:
                        Hk = Hk_cache

                    # Regularize Hessian
                    Hk_reg = Hk + self.reg_param * sp.sparse.eye(Hk.shape[0])
                    dx = self.lsolver.solve(Hk_reg, -gk)
                    alpha = self.alpha0_func(xk, dx)
                    xk = xk + alpha * dx

                    if np.linalg.norm(alpha * dx) > 1e-5:
                        Hk_cache = None  # Invalidate cache

                    if callback is not None:
                        callback(xk)

        return xk


@register(type="optimizer", name="bfgs")
class BFGSOptimizer(OptimizerBase):
    def __init__(
        self,
        line_searcher: "LineSearchBase",
        alpha0_func: Callable[[np.ndarray, np.ndarray], float],
        maxiters: int = 100,
        rtol: float = 1e-5,
    ):
        """Initialize the BFGS Optimizer.

        Parameters
        ----------
        - line_searcher: Line search method instance.
        - alpha0_func: Function to compute initial step size.
        - maxiters: Maximum number of iterations.
        - rtol: Relative tolerance for convergence.

        """
        self.line_searcher = line_searcher
        self.alpha0_func = alpha0_func
        self.maxiters = maxiters
        self.rtol = rtol
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable[[np.ndarray], sp.sparse.csc_matrix]] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform BFGS optimization.

        Parameters
        ----------
        - x0: Initial guess.
        - f: Objective function.
        - grad: Gradient function.
        - hess: Hessian function (optional).
        - callback: Optional callback after each iteration.
        - kwargs: Additional keyword arguments.

        Returns
        -------
        - Optimized solution vector.

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
                self.logger.info(
                    f"BFGS converged at iteration {k} with gradient norm {gnorm}"
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
                I = np.eye(n)
                Vk = I - rho_k * np.outer(sk, yk)
                Hk = Vk @ Hk @ Vk.T + rho_k * np.outer(sk, sk)
            else:
                self.logger.warning("Skipping update due to small sy in BFGS.")

            xk, gk = xk_new, gk_new
            if callback:
                callback(xk)

        return xk


@register(type="optimizer", name="lbfgs")
class LBFGSOptimizer(OptimizerBase):
    def __init__(self, maxiters: int = 100, rtol: float = 1e-5, m: int = 10):
        self.maxiters = maxiters
        self.rtol = rtol
        self.m = m  # Memory parameter
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable[[np.ndarray], sp.sparse.csc_matrix]] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform L-BFGS optimization.

        Parameters
        ----------
        - x0: Initial guess.
        - f: Objective function.
        - grad: Gradient function.
        - hess: Hessian function (optional).
        - callback: Optional callback after each iteration.
        - kwargs: Additional keyword arguments.

        Returns
        -------
        - Optimized solution vector.

        """
        maxiters = kwargs.get("maxiters", self.maxiters)
        rtol = kwargs.get("rtol", self.rtol)

        xk = x0.copy()
        n = len(xk)
        gk = grad(xk)
        s_list, y_list, rho_list = [], [], []

        for k in range(maxiters):
            gnorm = np.linalg.norm(gk, np.inf)
            if gnorm < rtol:
                self.logger.info(
                    f"L-BFGS converged at iteration {k} with gradient norm {gnorm}"
                )
                break

            # Two-loop recursion
            q = gk.copy()
            alpha_list = []
            for i in range(len(s_list) - 1, -1, -1):
                si, yi, rhoi = s_list[i], y_list[i], rho_list[i]
                alpha_i = rhoi * si @ q
                q -= alpha_i * yi
                alpha_list.append(alpha_i)

            r = q
            for i in range(len(s_list)):
                si, yi, rhoi = s_list[i], y_list[i], rho_list[i]
                beta = rhoi * yi @ r
                r += si * (alpha_list[-(i + 1)] - beta)

            pk = -r
            alpha = self.alpha0_func(xk, pk) if hasattr(self, "alpha0_func") else 1.0
            xk_new = xk + alpha * pk
            gk_new = grad(xk_new)

            sk = xk_new - xk
            yk = gk_new - gk
            sy = sk @ yk

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
                self.logger.warning("Skipping update due to small sy in L-BFGS.")

            xk, gk = xk_new, gk_new
            if callback:
                callback(xk)

        return xk


class OptimizerFactory(metaclass=SingletonMeta):
    """Factory class to create optimizer instances.
    Implemented as a Singleton to ensure only one instance exists.
    """

    def __init__(self):
        self.line_search_factory = LineSearchFactory()
        self.linear_solver_factory = LinearSolverFactory()

    @lru_cache(maxsize=None)
    def get_class(self, type_lower: str) -> Type[OptimizerBase]:
        registry_container = RegistryContainer()
        return registry_container.optimizer.get(type_lower)

    def create(self, type: str, **kwargs) -> OptimizerBase:
        type_lower = type.lower()
        try:
            optimizer_cls = self.get_class(type_lower)
        except ValueError as e:
            logger.error(str(e))
            raise

        logger.info(f"Creating optimizer '{type_lower}'.")

        # Extract line search parameters
        line_search_method = kwargs.pop("line_search_method", "backtracking").lower()
        line_search_kwargs = kwargs.pop("line_search_kwargs", {})

        # Create line_searcher using LineSearchFactory
        try:
            line_searcher = self.line_search_factory.create(
                type=line_search_method,
                f=kwargs.get("f"),
                grad_f=kwargs.get("grad_f"),
                **line_search_kwargs,
            )
            logger.info(f"Line searcher '{line_search_method}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create line searcher: {e}")
            raise

        # Create linear solver if optimizer requires it
        lsolver = None
        if type_lower == "default":  # Assuming 'default' maps to NewtonOptimizer
            linear_solver_method = kwargs.pop("linear_solver_method", "default").lower()
            linear_solver_kwargs = kwargs.pop("linear_solver_kwargs", {})
            try:
                lsolver = self.linear_solver_factory.create(
                    type=linear_solver_method,
                    dofs=kwargs.get("dofs"),
                    **linear_solver_kwargs,
                )
                logger.info(
                    f"Linear solver '{linear_solver_method}' created successfully."
                )
            except Exception as e:
                logger.error(f"Failed to create linear solver: {e}")
                raise

        # Extract alpha0_func
        alpha0_func = kwargs.pop("alpha0_func", None)
        if alpha0_func is None:
            raise ValueError("alpha0_func is required.")
        if not callable(alpha0_func):
            raise ValueError("alpha0_func must be a callable function.")

        # Prepare optimizer arguments
        optimizer_args = {
            "maxiters": kwargs.pop("maxiters", 100),
            "rtol": kwargs.pop("rtol", 1e-5),
            "line_searcher": line_searcher,
            "alpha0_func": alpha0_func,
        }

        if type_lower == "default":  # NewtonOptimizer
            if lsolver is None:
                raise ValueError(
                    "Linear solver 'lsolver' is required for Newton's method."
                )
            optimizer_args.update(
                {
                    "lsolver": lsolver,
                    "reg_param": kwargs.pop("reg_param", 1e-4),
                    "n_threads": kwargs.pop("n_threads", 1),
                }
            )
        elif type_lower == "lbfgs":
            optimizer_args["m"] = kwargs.pop("m", 10)

        # Create and return optimizer instance
        try:
            optimizer = optimizer_cls(**optimizer_args)
            logger.info(f"Optimizer '{type_lower}' created successfully.")
            return optimizer
        except TypeError as e:
            logger.error(f"Failed to initialize optimizer '{type_lower}': {e}")
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while creating optimizer '{type_lower}': {e}"
            )
            raise