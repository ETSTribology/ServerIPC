import numpy as np
import scipy as sp
import ipctk
from core.parameters import Parameters
from utils.mesh_utils import to_surface

class Gradient:
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> np.ndarray:
        params = self.params
        dt = params.dt
        dt2 = params.dt2
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

        params.hep.compute_element_elasticity(x, grad=True, hessian=False)
        gU = hep.gradient()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)

        barrier_potential = B(cconstraints, cmesh, BX)
        gB = B.gradient(cconstraints, cmesh, BX)
        gB = cmesh.to_full_dof(gB)

        # Cannot compute gradient without barrier stiffness
        if kB is None:
            from contact.barrier_initializer import BarrierInitializer
            binit = BarrierInitializer(params)
            binit(x, gU, gB)

        kB = params.kB
        BXdot = to_surface(v, mesh, cmesh)

        # Use the BarrierPotential in the build method
        fconstraints.build(cmesh, BX, cconstraints, B, kB, mu)

        friction_potential = D(fconstraints, cmesh, BXdot)
        gF = D.gradient(fconstraints, cmesh, BXdot)
        gF = cmesh.to_full_dof(gF)
        g = M @ (x - xtilde) + dt2 * gU + kB * gB + dt * gF
        return g
