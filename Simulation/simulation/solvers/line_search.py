import numpy as np
import scipy as sp
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging



def line_search(alpha0: float,
                xk: np.ndarray,
                dx: np.ndarray,
                gk: np.ndarray,
                f: Callable[[np.ndarray], float],
                maxiters: int = 20,
                c: float = 1e-4,
                tau: float = 0.5,
                alpha_threshold: float = 1e-8) -> float:
    alphaj = alpha0
    Dfk = gk.dot(dx)
    fk = f(xk)
    
    # Avoid unnecessary iterations if Dfk is too small (indicating a stationary point)
    if np.abs(Dfk) < 1e-12:
        return 0.0
    
    flinear = fk + alphaj * c * Dfk

    for j in range(maxiters):
        # Evaluate the function at the new point
        fx = f(xk + alphaj * dx)
        
        # Check the Armijo condition
        if fx <= flinear:
            return alphaj
        
        # Reduce step size
        alphaj *= tau
        
        # Update the linear approximation value to prevent recomputation
        flinear = fk + alphaj * c * Dfk
        
        # Stop if the step size becomes too small
        if alphaj < alpha_threshold:
            print("Warning: Line search step size is too small.")
            break
            
    return alphaj