import numpy as np
import scipy as sp
import ipctk
from core.parameters import Parameters
from utils.mesh_utils import to_surface
import logging

logger = logging.getLogger(__name__)

class Potential:
    def __init__(self, params: Parameters):
        self.params = params
        self.cached_x = None
        self.cached_U = None

    def __call__(self, x: np.ndarray) -> float:
        params = self.params
        dt = params.dt
        xt = params.xt
        xtilde = params.xtilde
        M = params.M
        hep = params.hep
        mesh = params.mesh
        cmesh = params.cmesh
        cconstraints = params.cconstraints
        fconstraints = params.fconstraints
        dhat = params.dhat
        dmin = params.dmin
        mu = params.mu
        epsv = params.epsv
        kB = params.kB
        B = params.barrier_potential
        D = params.friction_potential

        if np.array_equal(x, self.cached_x):
            U = self.cached_U
        else:
            hep.compute_element_elasticity(x, grad=False, hessian=False)
            U = hep.eval()
            self.cached_U = U
            self.cached_x = x.copy()

        v = (x - xt) / dt

        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)

        # Compute the barrier potential
        cconstraints.use_area_weighting = True
        cconstraints.use_improved_max_approximator = True
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)

        # Build friction constraints
        fconstraints.build(cmesh, BX, cconstraints, B, kB, mu)

        EB = B(cconstraints, cmesh, BX)
        EF = D(fconstraints, cmesh, BXdot)

        potential_energy = (
            0.5 * (x - xtilde).T @ M @ (x - xtilde) + dt**2 * U + kB * EB + dt**2 * EF
        )

        # Add the check for finite potential energy here
        if not np.isfinite(potential_energy):
            logger.error(f"Potential energy is not finite: {potential_energy}")
            raise ValueError("Potential energy is NaN or Inf.")

        return potential_energy


