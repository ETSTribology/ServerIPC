import numpy as np
import scipy as sp
import ipctk
from core.parameters import Parameters
from utils.mesh_utils import to_surface

class Potential:
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> float:
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

        hep.compute_element_elasticity(x, grad=False, hessian=False)
        U = hep.eval()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)
        cconstraints.use_area_weighting = True
        cconstraints.use_improved_max_approximator = True
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)

        # Create a BarrierPotential object and use it in the build method
        ipctk.BarrierPotential.use_physical_barrier = True
        B = ipctk.BarrierPotential(dhat)
        fconstraints.build(cmesh, BX, cconstraints, B, kB, mu)

        EB = B(cconstraints, cmesh, BX)
        D = ipctk.FrictionPotential(epsv)
        EF = D(fconstraints, cmesh, BXdot)
        return 0.5 * (x - xtilde).T @ M @ (x - xtilde) + dt2*U + kB * EB + dt2*EF
