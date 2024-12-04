import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import ipctk
import numpy as np

from simulation.core.parameters import ParametersBase
from simulation.core.modifier.mesh import to_surface
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
    """
    Factory class for creating barrier updater instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a barrier updater instance based on the configuration.

        Args:
            config: A dictionary containing the barrier updater configuration.

        Returns:
            An instance of the barrier updater class.

        Raises:
            ValueError: 
        """
        logger.info("Creating barrier updater...")
        barrier_updater_config = config.get("barrier_updater", {})
        barrier_updater_type = barrier_updater_config.get("type", "default").lower()

        if barrier_updater_type not in BarrierUpdaterFactory._instances:
            if barrier_updater_type == "default":
                barrier_updater_instance = BarrierUpdater()
            else:
                raise ValueError(f"Unknown barrier updater type: {barrier_updater_type}")

            BarrierUpdaterFactory._instances[barrier_updater_type] = barrier_updater_instance

        return BarrierUpdaterFactory._instances[barrier_updater_type]