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


# Base Class for CCD Implementations
class CCDBase(ABC):
    def __init__(self, params: ParametersBase):
        """
        Initialize the CCDBase with parameters.

        Args:
            params (ParametersBase): The parameters for the CCD.
        """
        self.params = params
        self.broad_phase_method = params.broad_phase_method
        self.alpha = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        """
        Compute the collision-free stepsize using CCD.

        Args:
            x (np.ndarray): Current positions.
            dx (np.ndarray): Position increments.

        Returns:
            float: Maximum collision-free stepsize.
        """
        raise NotImplementedError("Method not implemented.")


class CCD(CCDBase):
    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        """
        Compute the collision-free stepsize using CCD.

        Args:
            x (np.ndarray): Current positions.
            dx (np.ndarray): Position increments.

        Returns:
            float: Maximum collision-free stepsize.
        """
        try:
            self.logger.debug(SimulationLogMessageCode.COMMAND_STARTED.details("Computing CCD stepsize"))
            mesh = self.params.mesh
            cmesh = self.params.cmesh
            dmin = self.params.dmin
            broad_phase_method = self.params.broad_phase_method

            self.logger.debug(SimulationLogMessageCode.COMMAND_STARTED.details(f"Computing CCD stepsize with broad_phase_method={broad_phase_method}"))

            BXt0 = to_surface(x, mesh, cmesh)
            BXt1 = to_surface(x + dx, mesh, cmesh)
            max_alpha = ipctk.compute_collision_free_stepsize(
                cmesh, BXt0, BXt1, broad_phase_method=broad_phase_method, min_distance=dmin
            )
            self.alpha = max_alpha
            self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Computed CCD stepsize: alpha={max_alpha}"))
            return max_alpha
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to compute CCD stepsize: {e}"))
            raise SimulationError(SimulationErrorCode.CCD_SETUP, "Failed to compute CCD stepsize", details=str(e))


class CCDFactory(metaclass=SingletonMeta):
    """
    Factory class for creating CCD instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a CCD instance based on the configuration.

        Args:
            config: A dictionary containing the CCD configuration.

        Returns:
            An instance of the CCD class.

        Raises:
            SimulationError: If the CCD type is unknown.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating CCD..."))
        ccd_config = config.get("ccd", {})
        ccd_type = ccd_config.get("type", "default").lower()

        if ccd_type not in CCDFactory._instances:
            try:
                if ccd_type == "default":
                    ccd_instance = CCD(config)
                else:
                    raise ValueError(f"Unknown CCD type: {ccd_type}")

                CCDFactory._instances[ccd_type] = ccd_instance
                logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"CCD '{ccd_type}' created successfully."))
            except ValueError as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unknown CCD type: {ccd_type}"))
                raise SimulationError(SimulationErrorCode.CCD_SETUP, f"Unknown CCD type: {ccd_type}", details=str(e))
            except Exception as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unexpected error when creating CCD '{ccd_type}': {e}"))
                raise SimulationError(SimulationErrorCode.CCD_SETUP, f"Unexpected error when creating CCD '{ccd_type}'", details=str(e))

        return CCDFactory._instances[ccd_type]