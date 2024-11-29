import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import ipctk
import numpy as np
from simulation.core.parameters import ParametersBase
from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register
from simulation.core.utils.modifier.mesh import to_surface

logger = logging.getLogger(__name__)


# Base Class for CCD Implementations
class CCDBase(ABC):
    def __init__(self, params: ParametersBase):
        self.params = params
        self.broad_phase_method = params.broad_phase_method
        self.alpha = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        """Compute the collision-free stepsize using CCD.

        Args:
            x (np.ndarray): Current positions.
            dx (np.ndarray): Position increments.

        Returns:
            float: Maximum collision-free stepsize.

        """
        raise NotImplementedError("Method not implemented.")


registry_container = RegistryContainer()
registry_container.add_registry("ccd", "simulation.core.contact.ccd.CCDBase")


@register(type="ccd", name="default")
class CCD(CCDBase):
    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        self.logger.debug("Computing CCD stepsize")
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        dmin = self.params.dmin
        broad_phase_method = self.params.broad_phase_method

        self.logger.debug(
            f"Computing CCD stepsize with broad_phase_method={broad_phase_method}"
        )

        BXt0 = to_surface(x, mesh, cmesh)
        BXt1 = to_surface(x + dx, mesh, cmesh)
        max_alpha = ipctk.compute_collision_free_stepsize(
            cmesh, BXt0, BXt1, broad_phase_method=broad_phase_method, min_distance=dmin
        )
        self.alpha = max_alpha
        self.logger.info(f"Computed CCD stepsize: alpha={max_alpha}")
        return max_alpha


class CCDFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the CCD class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.ccd.get(type_lower)

    @staticmethod
    def create(type: str, params: ParametersBase) -> CCDBase:
        """Create a CCD implementation.

        Args:
            type (str): Name of the CCD implementation.
            params (ParametersBase): Simulation parameters.

        Returns:
            CCDBase: CCD implementation.

        """
        type_lower = type.lower()
        try:
            ccd_cls = CCDFactory.get_class(type_lower)
            return ccd_cls(params)
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create CCD implementation '{type}': {e}")
            raise RuntimeError(
                f"Error during CCD initialization for method '{type}': {e}"
            ) from e
