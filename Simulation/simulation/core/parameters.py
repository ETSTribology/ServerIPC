import numpy as np
import scipy as sp
import ipctk
from utils.mesh_utils import to_surface
from materials import Material
import pbatoolkit as pbat
import numpy as np

class Parameters:
    def __init__(self,
                 mesh: pbat.fem.Mesh,
                 xt: np.ndarray,
                 vt: np.ndarray,
                 a: np.ndarray,
                 M: sp.sparse.dia_array,
                 hep: pbat.fem.HyperElasticPotential,
                 dt: float,
                 cmesh: ipctk.CollisionMesh,
                 cconstraints: ipctk.Collisions,
                 fconstraints: ipctk.FrictionCollisions,
                 material: Material,
                 dhat: float = 1e-3,
                 dmin: float = 1e-4,
                 mu: float = 0.3,
                 epsv: float = 1e-4):
        self.mesh = mesh
        self.xt = xt
        self.vt = vt
        self.a = a
        self.M = M
        self.hep = hep
        self.dt = dt
        self.cmesh = cmesh
        self.cconstraints = cconstraints
        self.fconstraints = fconstraints
        self.dhat = dhat
        self.dmin = dmin
        self.mu = mu
        self.epsv = epsv

        self.dt2 = dt**2
        self.xtilde = xt + dt*vt + self.dt2 * a
        self.avgmass = M.diagonal().mean()
        self.kB = None
        self.maxkB = None
        self.dprev = None
        self.dcurrent = None
        BX = to_surface(xt, mesh, cmesh)
        self.bboxdiag = ipctk.world_bbox_diagonal_length(BX)
        self.gU = None
        self.gB = None