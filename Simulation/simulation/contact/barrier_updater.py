import logging
import ipctk
import numpy as np
from core.parameters import Parameters
from utils.mesh_utils import to_surface

logger = logging.getLogger(__name__)

class BarrierUpdater:
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, xk: np.ndarray):
        logger.debug("Updating barrier stiffness")
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        kB = self.params.kB
        maxkB = self.params.maxkB
        dprev = self.params.dprev
        bboxdiag = self.params.bboxdiag
        dhat = self.params.dhat
        dmin = self.params.dmin
        cconstraints = self.params.cconstraints

        BX = to_surface(xk, mesh, cmesh)
        dcurrent = cconstraints.compute_minimum_distance(cmesh, BX)
        kB_new = ipctk.update_barrier_stiffness(
            dprev, dcurrent, maxkB, kB, bboxdiag, dmin=dmin
        )
        self.params.kB = kB_new
        self.params.dprev = dcurrent

        logger.debug(f"Barrier stiffness updated: kB={kB_new}, dprev={dcurrent}")
