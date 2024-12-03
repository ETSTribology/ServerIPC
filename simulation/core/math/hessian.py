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


class HessianFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Hessian class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.hessian.get(type_lower)

    @staticmethod
    def create(type: str, params: ParametersBase) -> HessianBase:
        """Factory method to create a Hessian computation method based on the provided type.

        :param type: The type of Hessian computation method to create.
        :param params: The simulation parameters.
        :return: An instance of the specified Hessian computation method.
        """
        type_lower = type.lower()
        logger.debug(f"Creating Hessian computation method of type '{type_lower}'.")
        try:
            # Retrieve the Hessian class from the generalized registry
            registry_container = RegistryContainer()
            hessian_cls = registry_container.hessian.get(type_lower)

            # Instantiate the Hessian class
            hessian_instance = hessian_cls(params)
            logger.info(
                f"Hessian computation method '{type_lower}' created successfully using class '{hessian_cls.__name__}'."
            )
            return hessian_instance
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create Hessian computation method '{type_lower}': {e}")
            raise RuntimeError(
                f"Error during Hessian computation initialization for method '{type_lower}': {e}"
            ) from e
