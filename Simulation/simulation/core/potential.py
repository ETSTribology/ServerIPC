import numpy as np
import scipy as sp
import ipctk
from core.parameters import Parameters
from utils.mesh_utils import to_surface
import logging

logger = logging.getLogger(__name__)

class Potential():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> float:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        xtilde = self.params.xtilde
        M = self.params.M
        hep = self.params.hep
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        cconstraints = self.params.cconstraints
        fconstraints = self.params.fconstraints
        dhat = self.params.dhat
        dmin = self.params.dmin
        mu = self.params.mu
        epsv = self.params.epsv
        kB = self.params.kB
        B = self.params.barrier_potential
        D = self.params.friction_potential

        hep.compute_element_elasticity(x, grad=False, hessian=False)
        U = hep.eval()
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
        return potential_energy



