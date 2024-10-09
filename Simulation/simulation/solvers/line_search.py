import numpy as np
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def line_search(alpha0: float,
                xk: np.ndarray,
                dx: np.ndarray,
                gk: np.ndarray,
                f: Callable[[np.ndarray], float],
                maxiters: int = 20,
                c: float = 1e-4,
                tau: float = 0.5,
                alpha_threshold: float = 1e-8,
                grad_threshold: float = 1e-12) -> float:
    """
    Optimized line search using backtracking and adaptive strategies.
    """
    alphaj = alpha0
    Dfk = np.dot(gk, dx)
    fk = f(xk)

    # Early exit if gradient is near zero (stationary point)
    if np.abs(Dfk) < grad_threshold:
        return 0.0

    flinear = fk + alphaj * c * Dfk

    # Function evaluation cache to reduce redundant evaluations
    xk_dx = xk + alphaj * dx
    fx = f(xk_dx)

    for j in range(maxiters):
        # Check the Armijo condition
        if fx <= flinear:
            return alphaj

        # Reduce step size and update function evaluation
        alphaj *= tau
        if alphaj < alpha_threshold:
            logger.warning("Line search step size is too small.")
            break

        # Update flinear and fx
        flinear = fk + alphaj * c * Dfk
        xk_dx = xk + alphaj * dx
        fx = f(xk_dx)

    return alphaj


def parallel_line_search(alpha0: float,
                         xk: np.ndarray,
                         dx: np.ndarray,
                         gk: np.ndarray,
                         f: Callable[[np.ndarray], float],
                         maxiters: int = 20,
                         c: float = 1e-4,
                         tau: float = 0.5,
                         alpha_threshold: float = 1e-8,
                         n_jobs: int = 2,
                         grad_threshold: float = 1e-12) -> float:
    """
    Optimized line search using parallel function evaluations.
    """
    alphaj = alpha0
    Dfk = np.dot(gk, dx)
    fk = f(xk)

    # Early exit if gradient is near zero (stationary point)
    if np.abs(Dfk) < grad_threshold:
        return 0.0

    flinear = fk + alphaj * c * Dfk

    def evaluate(alpha):
        return f(xk + alpha * dx)

    # Ensure n_jobs is appropriate for the computation cost of f
    if n_jobs > 1 and callable(f):
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(evaluate, alphaj): alphaj for _ in range(maxiters)}
            for future in as_completed(futures):
                fx = future.result()
                alphaj = futures[future]

                # Check the Armijo condition
                if fx <= flinear:
                    return alphaj

                # Reduce step size
                alphaj *= tau
                if alphaj < alpha_threshold:
                    logger.warning("Line search step size is too small.")
                    break

                # Update flinear
                flinear = fk + alphaj * c * Dfk
    else:
        # Fallback to sequential evaluation if parallelism is not beneficial
        for j in range(maxiters):
            fx = evaluate(alphaj)

            # Check the Armijo condition
            if fx <= flinear:
                return alphaj

            # Reduce step size
            alphaj *= tau
            if alphaj < alpha_threshold:
                logger.warning("Line search step size is too small.")
                break

            # Update flinear
            flinear = fk + alphaj * c * Dfk

    return alphaj
