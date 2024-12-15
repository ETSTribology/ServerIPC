import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from simulation.core.contact.barrier import BarrierInitializer
from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class GradientBase(ABC):
    def __init__(self, params: Parameters):
        """
        Initialize the GradientBase with parameters.

        Args:
            params (Parameters): The parameters for the gradient computation.
        """
        self.params = params

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient vector.

        Args:
            x (np.ndarray): Current positions.

        Returns:
            np.ndarray: The computed gradient vector.
        """
        pass


class Gradient(GradientBase):
    def __init__(self, params: Parameters):
        super().__init__(params)

    def update_params(self, params: Parameters):
        """
        Update the parameters used by the Gradient instance.

        Args:
            params (Parameters): The new parameters to set.
        """
        if not isinstance(params, Parameters):
            raise ValueError("params must be an instance of Parameters.")
        self.params = params
        logger.info("Gradient parameters updated successfully.")

    def _compute_elastic_gradient(self, x: np.ndarray):
        """Compute the elastic gradient."""
        self.params.hep.compute_element_elasticity(x, grad=True, hessian=False)
        return self.params.hep.gradient()

    def _compute_velocity_and_surface(self, x: np.ndarray):
        """Compute velocity and surface representation."""
        v = (x - self.params.xt) / self.params.dt
        BX = to_surface(x, self.params.mesh, self.params.cmesh)
        return v, BX

    def _build_constraints(self, BX):
        """Build collision constraints."""
        self.params.cconstraints.build(
            self.params.cmesh, BX, self.params.dhat, dmin=self.params.dmin
        )

    def _compute_barrier_gradient(self, x: np.ndarray, gU: np.ndarray, BX):
        """Compute the barrier gradient and initialize barrier stiffness if needed."""
        gB = self.params.barrier_potential.gradient(self.params.cconstraints, self.params.cmesh, BX)
        gB = self.params.cmesh.to_full_dof(gB)

        if self.params.kB is None:
            logger.debug("Initializing barrier parameters...")
            binit = BarrierInitializer(self.params)
            binit(x, gU, gB)
            if self.params.kB is None:
                raise ValueError("Barrier stiffness initialization failed")

        return gB, self.params.kB

    def _compute_friction_gradient(self, v, BX):
        """
        Compute the friction gradient.

        Args:
            v (np.ndarray): Velocity vector
            BX (np.ndarray): Surface positions
            https://github.com/ETSTribology/ServerIPC/tree/a325486d8aed7ae90b7b44355b67dd7fc7f9c9ba/simulation/core/math
        Returns:
            np.ndarray: Friction gradient

        Raises:
            ValueError: If parameters are invalid
        """
        BXdot = to_surface(v, self.params.mesh, self.params.cmesh)
        self.params.fconstraints.build(
            self.params.cmesh,
            BX,
            self.params.cconstraints,
            self.params.barrier_potential,
            self.params.kB,
            self.params.mu,
        )
        friction_potential = self.params.friction_potential(
            self.params.fconstraints, self.params.cmesh, BXdot
        )
        gF = self.params.friction_potential.gradient(
            self.params.fconstraints, self.params.cmesh, BXdot
        )
        return self.params.cmesh.to_full_dof(gF)

    def _assemble_gradient(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray, gF: np.ndarray):
        """Assemble the final gradient vector."""
        return (
            self.params.M @ (x - self.params.xtilde)
            + self.params.dt2 * gU
            + self.params.kB * gB
            + self.params.dt * gF
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient vector with protected blocks."""
        try:
            gU = self._compute_elastic_gradient(x)
            v, BX = self._compute_velocity_and_surface(x)
            self._build_constraints(BX)
            gB, kB = self._compute_barrier_gradient(x, gU, BX)
            gF = self._compute_friction_gradient(v, BX)
            g = self._assemble_gradient(x, gU, gB, gF)

            logger.info("Gradient vector computed successfully")
            logger.info(f"Barrier stiffness: {kB}")
            return g
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            raise SimulationError(
                SimulationErrorCode.GRADIENT_SETUP, "Gradient computation failed", str(e)
            )


class GradientFactory(metaclass=SingletonMeta):
    """Factory class for creating gradient instances."""

    _instances = {}

    @staticmethod
    def create(params: ParametersBase) -> Any:
        """
        Create and return a default gradient instance or update the existing instance.

        Args:
            params (ParametersBase): Parameters for gradient initialization.

        Returns:
            Any: Instance of the Gradient class.

        Raises:
            SimulationError: If creation fails.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating gradient..."))

        instance_key = "default"

        try:
            if instance_key in GradientFactory._instances:
                # Update the existing instance with new parameters
                instance = GradientFactory._instances[instance_key]
                instance.update_params(params)
                logger.info("Gradient instance updated with new parameters.")
            else:
                # Create a new instance if not already present
                if params is None:
                    raise ValueError("Parameters required for gradient creation")

                gradient_instance = Gradient(params)
                GradientFactory._instances[instance_key] = gradient_instance
                logger.info(
                    SimulationLogMessageCode.COMMAND_SUCCESS.details(
                        "Gradient created successfully."
                    )
                )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Failed to create or update gradient: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.GRADIENT_SETUP, "Failed to create or update gradient", str(e)
            )

        return GradientFactory._instances[instance_key]
