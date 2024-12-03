import logging
from abc import ABC, abstractmethod

import ipctk
import numpy as np

from simulation.core.parameters import ParametersBase
from simulation.core.utils.modifier.mesh import to_surface
from simulation.core.utils.singleton import SingletonMeta

logger = logging.getLogger(__name__)


# Base Class for Barrier Updaters
class BarrierUpdaterBase(ABC):
    def __init__(self, params: ParametersBase):
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, xk: np.ndarray):
        """Update the barrier stiffness based on the current positions.

        Args:
            xk (np.ndarray): Current positions.

        """
        raise NotImplementedError("Method not implemented.")


class BarrierUpdater(BarrierUpdaterBase):
    def __call__(self, xk: np.ndarray):
        self.logger.debug("Updating barrier stiffness")
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
        kB_new = ipctk.update_barrier_stiffness(dprev, dcurrent, maxkB, kB, bboxdiag, dmin=dmin)
        self.params.kB = kB_new
        self.params.dprev = dcurrent

        self.logger.debug(f"Barrier stiffness updated: kB={kB_new}, dprev={dcurrent}")


class BarrierUpdaterFactory(metaclass=SingletonMeta):
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Barrier Updater class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.barrier_updater.get(type_lower)

    @staticmethod
    def create(type: str, params: ParametersBase) -> BarrierUpdaterBase:
        """Create a Barrier Updater instance based on the given type.

        Args:
            type (str): Name of the Barrier Updater.
            params (ParametersBase): Simulation parameters.

        Returns:
            BarrierUpdaterBase: Instance of the Barrier Updater.

        """
        type_lower = type.lower()
        try:
            barrier_updater_cls = BarrierUpdaterFactory.get_class(type_lower)
            return barrier_updater_cls(params)
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create Barrier Updater '{type}': {e}")
            raise RuntimeError(
                f"Error during Barrier Updater initialization for type '{type}': {e}"
            ) from e
