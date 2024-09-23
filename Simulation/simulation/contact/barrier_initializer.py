import ipctk
from core.parameters import Parameters
from utils.mesh_utils import to_surface
import logging
import numpy as np

class BarrierInitializer:
    def __init__(self, params: Parameters):
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray):
        params = self.params
        mesh = params.mesh
        cmesh = params.cmesh
        dhat = params.dhat
        dmin = params.dmin
        avgmass = params.avgmass
        bboxdiag = params.bboxdiag
        cconstraints = params.cconstraints

        # Compute adaptive barrier stiffness
        BX = to_surface(x, mesh, cmesh)
        B = ipctk.BarrierPotential(dhat)
        barrier_potential = B(cconstraints, cmesh, BX)
        gB = B.gradient(cconstraints, cmesh, BX)
        kB, maxkB = ipctk.initial_barrier_stiffness(
            bboxdiag, B.barrier, dhat, avgmass, gU, gB, dmin=dmin
        )
        dprev = cconstraints.compute_minimum_distance(cmesh, BX)
        params.kB = kB
        params.maxkB = maxkB
        params.dprev = dprev
        self.logger.info(f"Initialized barrier stiffness: kB={kB}, maxkB={maxkB}")
