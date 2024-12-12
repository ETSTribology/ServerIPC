import logging
from abc import ABC, abstractmethod
from typing import Any

import ipctk
import numpy as np
from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import ParametersBase
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

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
        self.alpha = None

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

    def __init__(self, params: ParametersBase):
        super().__init__(params)

    def update_params(self, params: ParametersBase):
        """
        Update the parameters used by the CCD instance.

        Args:
            params (ParametersBase): The new parameters to set.
        """
        if not isinstance(params, ParametersBase):
            raise ValueError("params must be an instance of ParametersBase.")
        self.params = params
        logger.info("CCD parameters updated successfully.")

    def _compute_surface_positions(self, x: np.ndarray, dx: np.ndarray) -> tuple:
        """Compute surface positions at start and end points."""
        try:
            BXt0 = to_surface(x, self.params.mesh, self.params.cmesh)
            BXt1 = to_surface(x + dx, self.params.mesh, self.params.cmesh)
            return BXt0, BXt1
        except Exception as e:
            self.logger.error(f"Surface position computation failed: {e}")
            raise SimulationError(
                SimulationErrorCode.CCD_SETUP,
                "Surface position computation failed",
                str(e)
            )

    def _compute_collision_free_step(self, BXt0: np.ndarray, BXt1: np.ndarray) -> float:
        """Compute collision-free stepsize."""
        try:
            max_alpha = ipctk.compute_collision_free_stepsize(
                self.params.cmesh,
                BXt0,
                BXt1,
                broad_phase_method=ipctk.BroadPhaseMethod.HASH_GRID,
                min_distance=self.params.dmin
            )
            return max_alpha
        except Exception as e:
            self.logger.error(f"Stepsize computation failed: {e}")
            raise SimulationError(
                SimulationErrorCode.CCD_SETUP,
                "Stepsize computation failed",
                str(e)
            )

    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        """Compute collision-free stepsize using CCD."""
        try:

            self.logger.debug(
                SimulationLogMessageCode.COMMAND_STARTED.details(
                    f"Computing CCD stepsize with method={self.params.broad_phase_method}"
                )
            )

            # Compute surface positions
            BXt0, BXt1 = self._compute_surface_positions(x, dx)

            # Compute collision-free stepsize
            max_alpha = self._compute_collision_free_step(BXt0, BXt1)

            # Store and return result
            self.alpha = max_alpha

            self.logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Computed CCD stepsize: alpha={max_alpha}"
                )
            )
            return max_alpha
        except SimulationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in CCD computation: {e}")
            raise SimulationError(
                SimulationErrorCode.CCD_SETUP,
                "Unexpected error in CCD computation",
                str(e)
            )


class CCDFactory(metaclass=SingletonMeta):
    """Factory class for creating CCD instances."""

    _instances = {}

    @staticmethod
    def create(params: ParametersBase) -> Any:
        """
        Create or update a CCD instance with the given parameters.

        Args:
            params (ParametersBase): Parameters for CCD initialization

        Returns:
            Any: Instance of the CCD class

        Raises:
            SimulationError: If creation fails
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating or updating CCD..."))

        instance_key = "default"
        try:
            if instance_key in CCDFactory._instances:
                # Update the existing instance with new parameters
                instance = CCDFactory._instances[instance_key]
                instance.update_params(params)
                logger.info("CCD instance updated with new parameters.")
            else:
                # Create a new instance if not already present
                if params is None:
                    raise ValueError("Parameters required for CCD creation")

                ccd_instance = CCD(params)
                CCDFactory._instances[instance_key] = ccd_instance
                logger.info(
                    SimulationLogMessageCode.COMMAND_SUCCESS.details("CCD created successfully.")
                )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to create or update CCD: {e}")
            )
            raise SimulationError(SimulationErrorCode.CCD_SETUP, "Failed to create or update CCD", str(e))

        return CCDFactory._instances[instance_key]

