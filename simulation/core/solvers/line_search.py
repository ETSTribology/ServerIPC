import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from core.registry.container import RegistryContainer
from core.registry.decorators import register

logger = logging.getLogger(__name__)


class LineSearchBase:
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        maxiters: int = 20,
        **kwargs,
    ):
        self.f = f
        self.grad_f = grad_f
        self.maxiters = maxiters

    def search(
        self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray
    ) -> float:
        raise NotImplementedError("This method should be overridden by subclasses")


registry_container = RegistryContainer()
registry_container.add_registry(
    "line_search", "core.solvers.line_search.LineSearchBase"
)


@register(type="line_search", name="backtracking")
class BacktrackingLineSearch(LineSearchBase):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        maxiters: int = 20,
        c: float = 1e-4,
        tau: float = 0.5,
        alpha_threshold: float = 1e-8,
        grad_threshold: float = 1e-12,
        **kwargs,
    ):
        super().__init__(f, grad_f, maxiters)
        self.c = c
        self.tau = tau
        self.alpha_threshold = alpha_threshold
        self.grad_threshold = grad_threshold

    def search(
        self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray
    ) -> float:
        logger.debug(f"Running Backtracking Line Search with alpha0={alpha0}")
        alphaj = alpha0
        Dfk = np.dot(gk, dx)
        fk = self.f(xk)

        # Early exit if directional derivative is near zero
        if np.abs(Dfk) < self.grad_threshold:
            logger.debug("Directional derivative is near zero. Exiting line search.")
            return 0.0

        for j in range(self.maxiters):
            flinear = fk + alphaj * self.c * Dfk
            xk_dx = xk + alphaj * dx
            fx = self.f(xk_dx)

            # Check the Armijo condition
            if fx <= flinear:
                return alphaj

            # Reduce step size
            alphaj *= self.tau
            if alphaj < self.alpha_threshold:
                logger.warning("Line search step size is too small.")
                break

        return alphaj


@register(type="line_search", name="wolfe")
class WolfeLineSearch(LineSearchBase):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        maxiters: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
    ):
        super().__init__(f, grad_f, maxiters)
        self.c1 = c1
        self.c2 = c2

    def search(
        self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray
    ) -> float:
        self.logger.debug(f"Running Wolfe Line Search with alpha0={alpha0}")
        alphaj = alpha0
        fk = self.f(xk)
        Dfk = np.dot(gk, dx)

        for j in range(self.maxiters):
            x_new = xk + alphaj * dx
            f_new = self.f(x_new)

            # Check Armijo condition
            if f_new > fk + self.c1 * alphaj * Dfk:
                alphaj *= 0.5
                continue

            # Check curvature condition
            g_new = self.grad_f(x_new)
            if np.dot(g_new, dx) < self.c2 * Dfk:
                alphaj *= 2.0
                continue

            return alphaj

        self.logger.warning("Wolfe line search did not converge.")
        return alphaj


@register(type="line_search", name="strong_wolfe")
class StrongWolfeLineSearch(LineSearchBase):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        maxiters: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
    ):
        super().__init__(f, grad_f, maxiters)
        self.c1 = c1
        self.c2 = c2

    def search(
        self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray
    ) -> float:
        self.logger.debug(f"Running Strong Wolfe Line Search with alpha0={alpha0}")
        alphaj = alpha0
        fk = self.f(xk)
        Dfk = np.dot(gk, dx)
        f_prev = fk

        for j in range(self.maxiters):
            x_new = xk + alphaj * dx
            f_new = self.f(x_new)

            # Check Armijo condition
            if f_new > fk + self.c1 * alphaj * Dfk or (j > 0 and f_new >= f_prev):
                alphaj *= 0.5
            else:
                g_new = self.grad_f(x_new)
                if abs(np.dot(g_new, dx)) <= -self.c2 * Dfk:
                    return alphaj
                if np.dot(g_new, dx) >= 0:
                    alphaj *= 0.5
                else:
                    alphaj *= 2.0
            f_prev = f_new

        self.logger.warning("Strong Wolfe line search did not converge.")
        return alphaj


@register(type="line_search", name="parallel")
class ParallelLineSearch(BacktrackingLineSearch):
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        maxiters: int = 20,
        c: float = 1e-4,
        tau: float = 0.5,
        alpha_threshold: float = 1e-8,
        grad_threshold: float = 1e-12,
        n_jobs: int = 2,
    ):
        super().__init__(f, grad_f, maxiters, c, tau, alpha_threshold, grad_threshold)
        self.n_jobs = n_jobs

    def search(
        self, alpha0: float, xk: np.ndarray, dx: np.ndarray, gk: np.ndarray
    ) -> float:
        self.logger.debug(f"Running Parallel Line Search with alpha0={alpha0}")
        alphaj = alpha0
        Dfk = np.dot(gk, dx)
        fk = self.f(xk)

        # Early exit if directional derivative is near zero
        if np.abs(Dfk) < self.grad_threshold:
            self.logger.debug(
                "Directional derivative is near zero. Exiting line search."
            )
            return 0.0

        def evaluate(alpha):
            return self.f(xk + alpha * dx)

        # Generate alpha values
        alpha_values = [alphaj * (self.tau**i) for i in range(self.maxiters)]

        # Parallel evaluation
        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(evaluate, alpha): alpha for alpha in alpha_values
                }

                for future in as_completed(futures):
                    fx = future.result()
                    alpha_value = futures[future]
                    flinear = fk + alpha_value * self.c * Dfk

                    if fx <= flinear:
                        return alpha_value

                    if alpha_value < self.alpha_threshold:
                        self.logger.warning("Line search step size is too small.")
                        break
        else:
            # Sequential evaluation
            for alpha_value in alpha_values:
                fx = evaluate(alpha_value)
                flinear = fk + alpha_value * self.c * Dfk

                if fx <= flinear:
                    return alpha_value

                if alpha_value < self.alpha_threshold:
                    self.logger.warning("Line search step size is too small.")
                    break

        return alphaj


class LineSearchFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the LineSearch class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.line_search.get(type_lower)

    @staticmethod
    def create(type: str, **kwargs) -> LineSearchBase:
        """Factory method to create a LineSearch instance based on the specified method."""
        type_lower = type.lower()
        try:
            # Retrieve the line search class from the cache or registry
            line_search_cls = LineSearchFactory.get_class(type_lower)

            # Filter constructor parameters
            required_params = line_search_cls.__init__.__code__.co_varnames
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in required_params
            }

            # Instantiate the class
            line_search_instance = line_search_cls(**filtered_kwargs)
            logger.info(
                f"Line search method '{type_lower}' created successfully using class '{line_search_cls.__name__}'."
            )
            return line_search_instance
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create line search method '{type}': {e}")
            raise RuntimeError(
                f"Error during line search initialization for method '{type}': {e}"
            ) from e
