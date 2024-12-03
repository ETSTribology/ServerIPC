import logging
from abc import ABC, abstractmethod

import ipctk
import numpy as np
import scipy as sp

from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.utils.modifier.mesh import to_surface

logger = logging.getLogger(__name__)


class HessianBase(ABC):
    def __init__(self, params: Parameters):
        self.params = params

    @abstractmethod
    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        pass

class Hessian(HessianBase):
    def __init__(self, params: ParametersBase):
        self.params = params

    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
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

        hep.compute_element_elasticity(x, grad=False, hessian=True)
        HU = hep.hessian()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)
        # Compute the Hessian of the barrier potential using the correct signature
        HB = B.hessian(
            cconstraints,
            cmesh,
            BX,
            project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS,
        )
        HB = cmesh.to_full_dof(HB)

        # Compute the Hessian of the friction dissipative potential
        HF = D.hessian(
            fconstraints,
            cmesh,
            BXdot,
            project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS,
        )
        HF = cmesh.to_full_dof(HF)
        H = M + dt2 * HU + kB * HB + HF
        return H


class HessianFactory(meta=SingletonMeta):
    """
    Factory class for creating hessian instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a hessian instance based on the configuration.

        Args:
            config: A dictionary containing the hessian configuration.

        Returns:
            An instance of the hessian class.

        Raises:
            ValueError: 
        """
        logger.info("Creating hessian...")
        hessian_config = config.get("hessian", {})
        hessian_type = hessian_config.get("type", "default").lower()

        if hessian_type not in HessianFactory._instances:
            if hessian_type == "default":
                hessian_instance = Hessian(config)
            else:
                raise ValueError(f"Unknown hessian type: {hessian_type}")

            HessianFactory._instances[hessian_type] = hessian_instance

        return HessianFactory._instances[hessian_type]

