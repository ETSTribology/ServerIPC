import scipy as sp
import ipctk
from core.parameters import Parameters
from utils.mesh_utils import to_surface
import numpy as np


class Hessian:
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        params = self.params
        dt = params.dt
        dt2 = params.dt2
        xt = params.xt
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

        # Compute the Hessian of the elastic potential
        hep.compute_element_elasticity(x, grad=False, hessian=True)
        HU = hep.hessian()

        # Velocity
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)

        # Compute the Hessian of the barrier potential using the correct signature
        B = ipctk.BarrierPotential(dhat)
        HB = B.hessian(cconstraints, cmesh, BX, project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS)
        HB = cmesh.to_full_dof(HB)

        # Compute the Hessian of the friction dissipative potential
        D = ipctk.FrictionPotential(epsv)
        HF = D.hessian(fconstraints, cmesh, BXdot, project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS)
        HF = cmesh.to_full_dof(HF)

        # Combine Hessians
        H = M + dt2 * HU + kB * HB + dt * HF
        return H
