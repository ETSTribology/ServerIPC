import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

import numpy as np
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)


class LineSearchMethod(Enum):
    ARMIJO = "armijo"
    BACKTRACKING = "backtracking"
    WOLFE = "wolfe"
    STRONG_WOLFE = "strong_wolfe"
    PARALLEL = "parallel"


@dataclass
class LineSearchConfig:
    """Configuration for line search methods"""

    type: LineSearchMethod
    max_iterations: int = 20
    convergence_tolerance: float = 1e-6
    c1: float = 1e-4  # Armijo condition parameter
    c2: float = 0.9  # Wolfe condition parameter
    tau: float = 0.5  # Step reduction factor
    n_jobs: int = 1  # For parallel line search


class ObjectiveFunction(Protocol):
    def __call__(self, x: np.ndarray) -> float: ...


class GradientFunction(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray: ...


class LineSearchBase:
    def __init__(
        self,
        f: ObjectiveFunction,
        grad_f: Optional[GradientFunction] = None,
        config: Optional[LineSearchConfig] = None,
    ):
        """
        Initialize the LineSearchBase with the objective function, gradient function, and configuration.

        Args:
            f (ObjectiveFunction): The objective function to minimize.
            grad_f (Optional[GradientFunction]): The gradient of the objective function.
            config (Optional[LineSearchConfig]): The configuration for the line search method.
        """
        self.f = f
        self.grad_f = grad_f
        self.config = config or LineSearchConfig(type=LineSearchMethod.ARMIJO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def search(self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray) -> float:
        """
        Perform the line search to find the optimal step size.

        Args:
            alpha0 (float): Initial step size.
            xk (np.ndarray): Current position.
            dx (np.ndarray): Search direction.
            gk (np.ndarray): Gradient at the current position.

        Returns:
            float: Optimal step size.
        """
        raise NotImplementedError


class ArmijoLineSearch(LineSearchBase):
    def search(self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray) -> float:
        """
        Perform the Armijo line search to find the optimal step size.

        Args:
            alpha0 (float): Initial step size.
            xk (np.ndarray): Current position.
            dx (np.ndarray): Search direction.
            gk (np.ndarray): Gradient at the current position.

        Returns:
            float: Optimal step size.
        """
        try:
            alphaj = alpha0
            fk = self.f(xk)
            Dfk = np.dot(gk, dx)

            if abs(Dfk) < self.config.convergence_tolerance:
                return 0.0

            for _ in range(self.config.max_iterations):
                xk_new = xk + alphaj * dx
                fk_new = self.f(xk_new)

                if fk_new <= fk + self.config.c1 * alphaj * Dfk:
                    return alphaj

                alphaj *= self.config.tau

                if alphaj < self.config.convergence_tolerance:
                    self.logger.warning(SimulationLogMessageCode.COMMAND_FAILED.details("Step size too small"))
                    break

            return alphaj
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed during Armijo line search: {e}"))
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, "Failed during Armijo line search", details=str(e))


class WolfeLineSearch(LineSearchBase):
    def search(self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray) -> float:
        """
        Perform the Wolfe line search to find the optimal step size.

        Args:
            alpha0 (float): Initial step size.
            xk (np.ndarray): Current position.
            dx (np.ndarray): Search direction.
            gk (np.ndarray): Gradient at the current position.

        Returns:
            float: Optimal step size.
        """
        if self.grad_f is None:
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, "Gradient function required for Wolfe line search")

        try:
            alphaj = alpha0
            fk = self.f(xk)
            Dfk = np.dot(gk, dx)

            for _ in range(self.config.max_iterations):
                xk_new = xk + alphaj * dx
                fk_new = self.f(xk_new)
                gk_new = self.grad_f(xk_new)

                # Armijo condition
                if fk_new > fk + self.config.c1 * alphaj * Dfk:
                    alphaj *= self.config.tau
                    continue

                # Wolfe condition
                if np.dot(gk_new, dx) >= self.config.c2 * Dfk:
                    return alphaj

                alphaj *= 2.0

            return alphaj
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed during Wolfe line search: {e}"))
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, "Failed during Wolfe line search", details=str(e))


class StrongWolfeLineSearch(LineSearchBase):
    def search(self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray) -> float:
        """
        Perform the Strong Wolfe line search to find the optimal step size.

        Args:
            alpha0 (float): Initial step size.
            xk (np.ndarray): Current position.
            dx (np.ndarray): Search direction.
            gk (np.ndarray): Gradient at the current position.

        Returns:
            float: Optimal step size.
        """
        if self.grad_f is None:
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, "Gradient function required for Strong Wolfe line search")

        try:
            alphaj = alpha0
            fk = self.f(xk)
            Dfk = np.dot(gk, dx)
            f_prev = fk

            for _ in range(self.config.max_iterations):
                xk_new = xk + alphaj * dx
                f_new = self.f(xk_new)

                # Check Armijo condition and monotonicity
                if f_new > fk + self.config.c1 * alphaj * Dfk or (f_new >= f_prev):
                    alphaj *= self.config.tau
                else:
                    g_new = self.grad_f(xk_new)
                    g_new_dx = np.dot(g_new, dx)

                    # Check strong Wolfe conditions
                    if abs(g_new_dx) <= -self.config.c2 * Dfk:
                        return alphaj

                    if g_new_dx >= 0:
                        alphaj *= self.config.tau
                    else:
                        alphaj *= 2.0

                f_prev = f_new

                if alphaj < self.config.convergence_tolerance:
                    self.logger.warning(SimulationLogMessageCode.COMMAND_FAILED.details("Step size too small"))
                    break

            return alphaj
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed during Strong Wolfe line search: {e}"))
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, "Failed during Strong Wolfe line search", details=str(e))


class ParallelLineSearch(LineSearchBase):
    def search(self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray) -> float:
        """
        Perform the Parallel line search to find the optimal step size.

        Args:
            alpha0 (float): Initial step size.
            xk (np.ndarray): Current position.
            dx (np.ndarray): Search direction.
            gk (np.ndarray): Gradient at the current position.

        Returns:
            float: Optimal step size.
        """
        try:
            alphaj = alpha0
            fk = self.f(xk)
            Dfk = np.dot(gk, dx)

            if abs(Dfk) < self.config.convergence_tolerance:
                return 0.0

            def evaluate(alpha: float) -> float:
                return self.f(xk + alpha * dx)

            # Generate candidate alphas
            alphas = [alphaj * (self.config.tau**i) for i in range(self.config.max_iterations)]

            if self.config.n_jobs > 1:
                with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                    futures = {executor.submit(evaluate, alpha): alpha for alpha in alphas}

                    for future in as_completed(futures):
                        fx = future.result()
                        alpha = futures[future]

                        if fx <= fk + self.config.c1 * alpha * Dfk:
                            return alpha

                        if alpha < self.config.convergence_tolerance:
                            self.logger.warning(SimulationLogMessageCode.COMMAND_FAILED.details("Step size too small"))
                            break
            else:
                for alpha in alphas:
                    fx = evaluate(alpha)

                    if fx <= fk + self.config.c1 * alpha * Dfk:
                        return alpha

                    if alpha < self.config.convergence_tolerance:
                        self.logger.warning(SimulationLogMessageCode.COMMAND_FAILED.details("Step size too small"))
                        break

            return alphaj
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed during Parallel line search: {e}"))
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, "Failed during Parallel line search", details=str(e))


class BacktrackingLineSearch(LineSearchBase):
    def search(self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray) -> float:
        """
        Perform the Backtracking line search to find the optimal step size.

        Args:
            alpha0 (float): Initial step size.
            xk (np.ndarray): Current position.
            dx (np.ndarray): Search direction.
            gk (np.ndarray): Gradient at the current position.

        Returns:
            float: Optimal step size.
        """
        try:
            alphaj = alpha0
            fk = self.f(xk)
            Dfk = np.dot(gk, dx)

            if abs(Dfk) < self.config.convergence_tolerance:
                return 0.0

            for _ in range(self.config.max_iterations):
                xk_new = xk + alphaj * dx
                fk_new = self.f(xk_new)

                if fk_new <= fk + self.config.c1 * alphaj * Dfk:
                    return alphaj

                alphaj *= self.config.tau

                if alphaj < self.config.convergence_tolerance:
                    self.logger.warning(SimulationLogMessageCode.COMMAND_FAILED.details("Step size too small"))
                    break

            return alphaj
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed during Backtracking line search: {e}"))
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, "Failed during Backtracking line search", details=str(e))


class LineSearchFactory:
    """Factory for creating line search instances"""

    @staticmethod
    def create_config(config: Dict[str, Any]) -> LineSearchConfig:
        """
        Create line search config from dict.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            LineSearchConfig: Line search configuration.
        """
        solver_config = config.get("solver", {}).get("optimization", {}).get("line_search", {})

        return LineSearchConfig(
            type=LineSearchMethod(solver_config.get("type", "armijo")),
            max_iterations=solver_config.get("max_iterations", 20),
            convergence_tolerance=solver_config.get("convergence_tolerance", 1e-6),
            c1=solver_config.get("c1", 1e-4),
            c2=solver_config.get("c2", 0.9),
            tau=solver_config.get("tau", 0.5),
            n_jobs=solver_config.get("n_jobs", 1),
        )

    @staticmethod
    def create(
        config: Dict[str, Any], f: ObjectiveFunction, grad_f: Optional[GradientFunction] = None
    ) -> LineSearchBase:
        """
        Create line search instance.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            f (ObjectiveFunction): Objective function.
            grad_f (Optional[GradientFunction]): Gradient function.

        Returns:
            LineSearchBase: Line search instance.
        """
        ls_config = LineSearchFactory.create_config(config)

        line_searches = {
            LineSearchMethod.ARMIJO: ArmijoLineSearch,
            LineSearchMethod.WOLFE: WolfeLineSearch,
            LineSearchMethod.STRONG_WOLFE: StrongWolfeLineSearch,
            LineSearchMethod.PARALLEL: ParallelLineSearch,
            LineSearchMethod.BACKTRACKING: BacktrackingLineSearch,
        }

        cls = line_searches.get(ls_config.type)
        if cls is None:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unknown line search type: {ls_config.type}"))
            raise SimulationError(SimulationErrorCode.LINE_SEARCH, f"Unknown line search type: {ls_config.type}")

        logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Line search '{ls_config.type}' created successfully."))
        return cls(f=f, grad_f=grad_f, config=ls_config)