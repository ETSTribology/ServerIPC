import numpy as np
import scipy as sp
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from solvers.line_search import line_search
import torch
from torch import autograd

logger = logging.getLogger(__name__)


def newton(x0: np.ndarray,
           f: Callable[[np.ndarray], float],
           grad: Callable[[np.ndarray], np.ndarray],
           hess: Callable[[np.ndarray], sp.sparse.csc_matrix],
           lsolver: Callable[[sp.sparse.csc_matrix, np.ndarray], np.ndarray],
           alpha0: Callable[[np.ndarray, np.ndarray], float],
           maxiters: int = 10,
           rtol: float = 1e-5,
           callback: Callable[[np.ndarray], None] = None) -> np.ndarray:
    logger = logging.getLogger('Newton')
    xk = x0.copy()
    gk = grad(xk)
    for k in range(maxiters):
        gnorm = np.linalg.norm(gk, 1)
        if gnorm < rtol:
            logger.info(f"Converged at iteration {k} with gradient norm {gnorm}.")
            break
        Hk = hess(xk)
        dx = lsolver(Hk, -gk)
        alpha = line_search(alpha0(xk, dx), xk, dx, gk, f)
        xk = xk + alpha * dx
        gk = grad(xk)
        if callback is not None:
            callback(xk)
        logger.debug(f"Iteration {k}: alpha={alpha}, gradient norm={gnorm}")
    return xk

def parallel_newton(x0: np.ndarray,
                    f: Callable[[np.ndarray], float],
                    grad: Callable[[np.ndarray], np.ndarray],
                    hess: Callable[[np.ndarray], sp.sparse.csc_matrix],
                    lsolver: Callable[[sp.sparse.csc_matrix, np.ndarray], np.ndarray],
                    alpha0: Callable[[np.ndarray, np.ndarray], float],
                    maxiters: int = 10,
                    rtol: float = 1e-5,
                    callback: Callable[[np.ndarray], None] = None,
                    n_threads: int = 8) -> np.ndarray:
    logger = logging.getLogger('ParallelNewton')
    xk = x0.copy()
    Hk_cache = None

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for k in range(maxiters):
            # Compute the gradient in parallel
            future_grad = executor.submit(grad, xk)
            gk = future_grad.result()

            gnorm = np.linalg.norm(gk, 1)
            if gnorm < rtol:
                logger.info(f"Converged at iteration {k}")
                break

            # Compute the Hessian in parallel
            if Hk_cache is None:
                future_hess = executor.submit(hess, xk)
                Hk = future_hess.result()
                Hk_cache = Hk
            else:
                Hk = Hk_cache

            # Solve the linear system
            dx = lsolver(Hk, -gk)

            # Perform line search to find optimal step size
            alpha = line_search(alpha0(xk, dx), xk, dx, gk, f, maxiters=10, c=1e-4, tau=0.8)

            # Update the current solution
            xk = xk + alpha * dx

            # Recompute gradient for next iteration
            future_grad = executor.submit(grad, xk)
            gk = future_grad.result()

            # Reset Hessian cache if step size is significant
            if np.linalg.norm(alpha * dx) > 1e-5:
                Hk_cache = None  # Invalidate the cache

            if callback is not None:
                callback(xk)

            logger.debug(f"Iteration {k}: alpha={alpha}, gradient norm={gnorm}")

    return xk
