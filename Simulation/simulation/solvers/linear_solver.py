import scipy as sp
import logging
import numpy as np

logger = logging.getLogger(__name__)

class LinearSolver:
    def __init__(self, dofs: np.ndarray):
        self.dofs = dofs

    def __call__(self, A: sp.sparse.csc_matrix, b: np.ndarray) -> np.ndarray:
        dofs = self.dofs
        try:
            # Select the submatrix corresponding to the free degrees of freedom
            Add = A[dofs, :][:, dofs]
            bd = b[dofs]

            # Compute the LU decomposition of the sparse matrix
            Addinv = sp.sparse.linalg.splu(Add)

            # Solve the linear system Add * x_dofs = bd
            x_dofs = Addinv.solve(bd).squeeze()

            # Initialize the full solution vector and set the computed values
            x = np.zeros_like(b)
            x[dofs] = x_dofs

            logger.debug("Linear system solved successfully.")
            return x
        except Exception as e:
            logger.error(f"Failed to solve linear system: {e}")
            raise
