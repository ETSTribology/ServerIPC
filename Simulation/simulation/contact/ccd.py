import logging
import ipctk
import numpy as np
from core.parameters import Parameters
from utils.mesh_utils import to_surface

logger = logging.getLogger(__name__)

class CCD:
    def __init__(self, params: Parameters, broad_phase_method: ipctk.BroadPhaseMethod = ipctk.BroadPhaseMethod.SWEEP_AND_PRUNE):
        self.params = params
        self.broad_phase_method = broad_phase_method

    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        logger.debug("Computing CCD stepsize")
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        dmin = self.params.dmin
        broad_phase_method = self.broad_phase_method

        logger.debug(f"Computing CCD stepsize with broad_phase_method={broad_phase_method}")

        BXt0 = to_surface(x, mesh, cmesh)
        BXt1 = to_surface(x + dx, mesh, cmesh)
        max_alpha = ipctk.compute_collision_free_stepsize(
            cmesh,
            BXt0,
            BXt1,
            broad_phase_method=broad_phase_method,
            min_distance=dmin
        )
        self.alpha = max_alpha
        return max_alpha
