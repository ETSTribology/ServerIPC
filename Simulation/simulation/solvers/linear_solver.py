import scipy as sp
import logging
import numpy as np
from scipy.sparse.linalg import cg, splu, spsolve
import pbat.math.linalg

logger = logging.getLogger(__name__)

class LinearSolver:
    def __init__(self, dofs: np.ndarray, solver_type: str = "ldlt"):
        self.dofs = dofs
        self.solver_type = solver_type.lower()

    def __call__(self, A: sp.sparse.csc_matrix, b: np.ndarray, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
        dofs = self.dofs
        try:
            # Select the submatrix corresponding to the free degrees of freedom
            Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
            bd = b[dofs]

            # Initialize the full solution vector
            x = np.zeros_like(b)

            # Use the specified solver to solve the linear system Add * x_dofs = bd
            if self.solver_type == "ldlt":
                Addinv = pbat.math.linalg.ldlt(Add)
                Addinv.compute(Add)
                x_dofs = Addinv.solve(bd).squeeze()
                logger.debug("LDLT decomposition used successfully.")

            elif self.solver_type == "cg":
                x_dofs, info = cg(Add, bd, tol=tol, maxiter=max_iter)

                if info == 0:
                    logger.debug("CG method converged successfully.")
                else:
                    logger.warning(f"CG did not converge. Info: {info}. Switching to LDLT decomposition.")
                    Addinv = pbat.math.linalg.ldlt(Add)
                    Addinv.compute(Add)
                    x_dofs = Addinv.solve(bd).squeeze()

            elif self.solver_type == "lu":
                Addinv = splu(Add)
                x_dofs = Addinv.solve(bd).squeeze()
                logger.debug("LU decomposition used successfully.")

            elif self.solver_type == "direct":
                x_dofs = spsolve(Add, bd)
                logger.debug("Direct sparse solve used successfully.")

            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}. Use 'ldlt', 'cg', 'lu', or 'direct'.")

            # Set the computed values in the full solution vector
            x[dofs] = x_dofs

            logger.debug("Linear system solved successfully.")
            return x

        except sp.sparse.linalg.ArpackNoConvergence as e:
            logger.error(f"Failed to solve linear system using CG or LU method due to convergence issues: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to solve linear system: {e}")
            raise
