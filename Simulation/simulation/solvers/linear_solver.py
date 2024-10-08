import scipy as sp
import logging
import numpy as np

logger = logging.getLogger(__name__)

class LinearSolver:
    def __init__(self, dofs: np.ndarray):
        self.dofs = dofs

    def __call__(self, A: sp.sparse.csc_matrix, b: np.ndarray, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
        dofs = self.dofs
        try:
            # Select the submatrix corresponding to the free degrees of freedom
            Add = A[dofs, :][:, dofs]
            bd = b[dofs]

            # Use the Conjugate Gradient method to solve the linear system Add * x_dofs = bd
            x_dofs, info = sp.sparse.linalg.cg(Add, bd, maxiter=max_iter)

            if info != 0:
                logger.warning(f"CG did not converge. Info: {info}")
                Addinv = sp.sparse.linalg.splu(Add)
                x_dofs = Addinv.solve(bd).squeeze()

            # Initialize the full solution vector and set the computed values
            x = np.zeros_like(b)
            x[dofs] = x_dofs

            logger.debug("Linear system solved successfully using CG method.")
            return x
        except Exception as e:
            logger.error(f"Failed to solve linear system using CG method: {e}")
            raise
