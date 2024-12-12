import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import ipctk
import numpy as np

from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import ParametersBase
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)


# Base Class for Barrier Updaters
class BarrierUpdaterBase(ABC):
    def __init__(self, params: ParametersBase):
        """
        Initialize the BarrierUpdaterBase with parameters.

        Args:
            params (ParametersBase): The parameters for the barrier updater.
        """
        self.params = params
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, xk: np.ndarray):
        """
        Update the barrier stiffness based on the current positions.

        Args:
            xk (np.ndarray): Current positions.
        """
        raise NotImplementedError("Method not implemented.")


class BarrierUpdater(BarrierUpdaterBase):
    def __call__(self, xk: np.ndarray):
        """
        Update the barrier stiffness based on the current positions.

        Args:
            xk (np.ndarray): Current positions.
        """
        try:
            self.logger.debug(SimulationLogMessageCode.COMMAND_STARTED.details("Updating barrier stiffness"))
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

            self.logger.debug(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Barrier stiffness updated: kB={kB_new}, dprev={dcurrent}"))
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to update barrier stiffness: {e}"))
            raise SimulationError(SimulationErrorCode.BARRIER_UPDATER_SETUP, "Failed to update barrier stiffness", details=str(e))


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
            SimulationError: If the barrier updater type is unknown.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating barrier updater..."))
        barrier_updater_config = config.get("barrier_updater", {})
        barrier_updater_type = barrier_updater_config.get("type", "default").lower()

        if barrier_updater_type not in BarrierUpdaterFactory._instances:
            try:
                if barrier_updater_type == "default":
                    barrier_updater_instance = BarrierUpdater()
                else:
                    raise ValueError(f"Unknown barrier updater type: {barrier_updater_type}")

                BarrierUpdaterFactory._instances[barrier_updater_type] = barrier_updater_instance
                logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Barrier updater '{barrier_updater_type}' created successfully."))
            except ValueError as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unknown barrier updater type: {barrier_updater_type}"))
                raise SimulationError(SimulationErrorCode.BARRIER_UPDATER_SETUP, f"Unknown barrier updater type: {barrier_updater_type}", details=str(e))
            except Exception as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unexpected error when creating barrier updater '{barrier_updater_type}': {e}"))
                raise SimulationError(SimulationErrorCode.BARRIER_UPDATER_SETUP, f"Unexpected error when creating barrier updater '{barrier_updater_type}'", details=str(e))

        return BarrierUpdaterFactory._instances[barrier_updater_type]