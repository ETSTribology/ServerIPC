# barrier.py
import logging
from abc import ABC
from typing import Tuple

import ipctk
import numpy as np

from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import ParametersBase
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class BarrierBase(ABC):
    """Base class for barrier operations."""

    def __init__(self, params: ParametersBase):
        self.params = params

    def update_params(self, new_params: ParametersBase) -> None:
        """
        Update the parameters dynamically.

        Args:
            new_params (ParametersBase): The new parameters to update.

        Raises:
            ValueError: If `new_params` is invalid.
        """
        if not isinstance(new_params, ParametersBase):
            raise ValueError("Invalid parameters provided.")
        self.params = new_params
        logger.info("Barrier parameters updated successfully.")

    def _compute_surface_position(self, x: np.ndarray) -> np.ndarray:
        """Compute surface position mapping."""
        try:
            return to_surface(x, self.params.mesh, self.params.cmesh)
        except Exception as e:
            logger.error(f"Failed to compute surface position: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_SETUP, "Failed to compute surface position", str(e)
            )

    def _compute_minimum_distance(self, BX: np.ndarray) -> float:
        """Compute minimum distance between collision objects."""
        try:
            return self.params.cconstraints.compute_minimum_distance(self.params.cmesh, BX)
        except Exception as e:
            logger.error(f"Failed to compute minimum distance: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_SETUP, "Failed to compute minimum distance", str(e)
            )

    def _update_parameters(self, kB: float, dprev: float, maxkB: float = None) -> None:
        """Update barrier parameters."""
        try:
            self.params.kB = kB
            self.params.dprev = dprev
            self.params.maxkB = maxkB
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_SETUP, "Failed to update parameters", str(e)
            )


class BarrierInitializer(BarrierBase):
    """Initialize barrier parameters."""

    def _compute_barrier_gradient(self, BX: np.ndarray) -> np.ndarray:
        """Compute barrier potential gradient."""
        try:
            barrier_potential = self.params.barrier_potential(
                self.params.cconstraints, self.params.cmesh, BX
            )
            return self.params.barrier_potential.gradient(
                self.params.cconstraints, self.params.cmesh, BX
            )
        except Exception as e:
            logger.error(f"Failed to compute barrier gradient: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_INITIALIZER_SETUP,
                "Failed to compute barrier gradient",
                str(e),
            )

    def _compute_initial_stiffness(self, gU: np.ndarray, gB: np.ndarray) -> Tuple[float, float]:
        """Compute initial barrier stiffness."""
        try:
            return ipctk.initial_barrier_stiffness(
                self.params.bboxdiag,
                self.params.barrier_potential.barrier,
                self.params.dhat,
                self.params.avgmass,
                gU,
                gB,
                dmin=self.params.dmin,
            )
        except Exception as e:
            logger.error(f"Failed to compute initial stiffness: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_INITIALIZER_SETUP,
                "Failed to compute initial stiffness",
                str(e),
            )

    def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray) -> None:
        """Initialize barrier parameters."""
        try:
            # Input validation
            if not all(isinstance(arg, np.ndarray) for arg in (x, gU, gB)):
                raise ValueError("All inputs must be numpy arrays")

            # Compute components
            BX = self._compute_surface_position(x)
            gB = self._compute_barrier_gradient(BX)
            kB, maxkB = self._compute_initial_stiffness(gU, gB)
            dprev = self._compute_minimum_distance(BX)

            # Update parameters
            self._update_parameters(kB, dprev, maxkB)

            logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Initialized barrier: kB={kB}, maxkB={maxkB}, dprev={dprev}"
                )
            )
        except Exception as e:
            logger.error(f"Failed to initialize barrier: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_INITIALIZER_SETUP,
                "Failed to initialize barrier",
                str(e),
            )


class BarrierUpdater(BarrierBase):
    """Update barrier stiffness."""

    def __call__(self, xk: np.ndarray) -> None:
        """Update barrier parameters."""
        try:
            # Compute surface position
            BX = self._compute_surface_position(xk)

            # Get current distance
            dcurrent = self._compute_minimum_distance(BX)

            # Update stiffness
            kB_new = ipctk.update_barrier_stiffness(
                self.params.dprev,
                dcurrent,
                self.params.maxkB,
                self.params.kB,
                self.params.bboxdiag,
                dmin=self.params.dmin,
            )

            # Update parameters
            self._update_parameters(kB_new, dcurrent)

            logger.debug(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Updated barrier: kB={kB_new}, dprev={dcurrent}"
                )
            )
        except Exception as e:
            logger.error(f"Failed to update barrier: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_UPDATER_SETUP, "Failed to update barrier", str(e)
            )


class BarrierFactory(metaclass=SingletonMeta):
    """Factory for creating barrier objects."""

    _initializer_instances = {}
    _updater_instances = {}

    @staticmethod
    def create_initializer(params: ParametersBase) -> BarrierInitializer:
        """Create or update a barrier initializer instance."""
        instance_key = "default"
        try:
            if instance_key in BarrierFactory._initializer_instances:
                instance = BarrierFactory._initializer_instances[instance_key]
                instance.update_params(params)  # Update the existing instance
                logger.info("Updated existing barrier initializer.")
            else:
                if params is None:
                    raise ValueError("Parameters required")
                instance = BarrierInitializer(params)
                BarrierFactory._initializer_instances[instance_key] = instance
                logger.info("Created new barrier initializer.")
            return instance
        except Exception as e:
            raise SimulationError(
                SimulationErrorCode.BARRIER_INITIALIZER_SETUP,
                "Failed to create or update initializer",
                str(e),
            )

    @staticmethod
    def create_updater(params: ParametersBase) -> BarrierUpdater:
        """Create or update a barrier updater instance."""
        instance_key = "default"
        try:
            if instance_key in BarrierFactory._updater_instances:
                instance = BarrierFactory._updater_instances[instance_key]
                instance.update_params(params)  # Update the existing instance
                logger.info("Updated existing barrier updater.")
            else:
                if params is None:
                    raise ValueError("Parameters required")
                instance = BarrierUpdater(params)
                BarrierFactory._updater_instances[instance_key] = instance
                logger.info("Created new barrier updater.")
            return instance
        except Exception as e:
            raise SimulationError(
                SimulationErrorCode.BARRIER_UPDATER_SETUP,
                "Failed to create or update updater",
                str(e),
            )
