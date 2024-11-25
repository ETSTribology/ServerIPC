import numpy as np
import scipy as sp
import ipctk
from contact.barrier_initializer import BarrierInitializer
from core.parameters import Parameters
from utils.mesh_utils import to_surface
import logging

logger = logging.getLogger(__name__)

class Gradient():
    def __init__(self, params: Parameters):
        self.params = params
        self.gradU = None
        self.gradB = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
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

        hep.compute_element_elasticity(x, grad=True, hessian=False)
        gU = hep.gradient()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)
        gB = B.gradient(cconstraints, cmesh, BX)
        gB = cmesh.to_full_dof(gB)

        # Cannot compute gradient without barrier stiffness
        if self.params.kB is None:
            binit = BarrierInitializer(self.params)
            binit(x, gU, gB)
            kB = self.params.kB

        kB = self.params.kB
        BXdot = to_surface(v, mesh, cmesh)

        # Use the BarrierPotential in the build method
        fconstraints.build(cmesh, BX, cconstraints, B, kB, mu)

        friction_potential = D(fconstraints, cmesh, BXdot)
        gF = D.gradient(fconstraints, cmesh, BXdot)
        gF = cmesh.to_full_dof(gF)
        g = M @ (x - xtilde) + dt2 * gU + kB * gB + dt * gF
        return g