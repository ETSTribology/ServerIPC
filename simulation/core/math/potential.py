import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class PotentialBase(ABC):
    def __init__(self, params: Parameters):
        """
        Initialize the PotentialBase with parameters.

        Args:
            params (Parameters): The parameters for the potential computation.
        """
        self.params = params

    @abstractmethod
    def __call__(self, *args, **kwds):
        """
        Compute the potential energy.

        Args:
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            The computed potential energy.
        """
        pass


class Potential(PotentialBase):
    def __init__(self, params: Parameters):
        super().__init__(params)

    def update_params(self, params: Parameters):
        """
        Update the parameters used by the Potential instance.

        Args:
            params (Parameters): The new parameters to set.
        """
        if not isinstance(params, Parameters):
            raise ValueError("params must be an instance of Parameters.")
        self.params = params
        logger.info("Potential parameters updated successfully.")

    def _compute_elastic_energy(self, x: np.ndarray) -> float:
        """
        Compute elastic potential energy.

        Args:
            x (np.ndarray): Position vector

        Returns:
            float: Elastic potential energy
        """
        try:
            self.params.hep.compute_element_elasticity(x, grad=False, hessian=False)
            return self.params.hep.eval()
        except Exception as e:
            logger.error(f"Failed to compute elastic energy: {e}")
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP, "Failed to compute elastic energy", str(e)
            )

    def _compute_velocity_surface(self, x: np.ndarray) -> tuple:
        """
        Compute velocity and surface mappings.

        Args:
            x (np.ndarray): Position vector

        Returns:
            tuple: (velocity, surface position, surface velocity)
        """
        try:
            v = (x - self.params.xt) / self.params.dt
            BX = to_surface(x, self.params.mesh, self.params.cmesh)
            BXdot = to_surface(v, self.params.mesh, self.params.cmesh)
            return v, BX, BXdot
        except Exception as e:
            logger.error(f"Failed to compute velocity/surface mappings: {e}")
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP,
                "Failed to compute velocity/surface mappings",
                str(e),
            )

    def _setup_constraints(self, BX: np.ndarray) -> None:
        """
        Setup collision and friction constraints.

        Args:
            BX (np.ndarray): Surface positions
        """
        try:
            self.params.cconstraints.use_area_weighting = True
            self.params.cconstraints.use_improved_max_approximator = True
            self.params.cconstraints.build(
                self.params.cmesh, BX, self.params.dhat, dmin=self.params.dmin
            )

            self.params.fconstraints.build(
                self.params.cmesh,
                BX,
                self.params.cconstraints,
                self.params.barrier_potential,
                self.params.kB,
                self.params.mu,
            )
        except Exception as e:
            logger.error(f"Failed to setup constraints: {e}")
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP, "Failed to setup constraints", str(e)
            )

    def _compute_constraint_energies(self, BX: np.ndarray, BXdot: np.ndarray) -> tuple:
        """
        Compute barrier and friction potential energies.

        Args:
            BX (np.ndarray): Surface positions
            BXdot (np.ndarray): Surface velocities

        Returns:
            tuple: (barrier energy, friction energy)
        """
        try:
            EB = self.params.barrier_potential(self.params.cconstraints, self.params.cmesh, BX)
            EF = self.params.friction_potential(self.params.fconstraints, self.params.cmesh, BXdot)
            return EB, EF
        except Exception as e:
            logger.error(f"Failed to compute constraint energies: {e}")
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP, "Failed to compute constraint energies", str(e)
            )

    def _compute_total_energy(self, x: np.ndarray, U: float, EB: float, EF: float) -> float:
        """
        Compute total potential energy.

        Args:
            x (np.ndarray): Position vector
            U (float): Elastic energy
            EB (float): Barrier energy
            EF (float): Friction energy

        Returns:
            float: Total potential energy
        """
        try:
            energy = (
                0.5 * (x - self.params.xtilde).T @ self.params.M @ (x - self.params.xtilde)
                + self.params.dt**2 * U
                + self.params.kB * EB
                + self.params.dt**2 * EF
            )
            logger.debug(f"Computed total energy: {energy}")
            if not np.isfinite(energy):
                logger.error(f"Potential energy is not finite: {energy}")
                raise ValueError("Potential energy is NaN or Inf.")
            return energy
        except Exception as e:
            logger.error(f"Failed to compute total energy: {e}")
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP, "Failed to compute total energy", str(e)
            )

    def __call__(self, x: np.ndarray) -> float:
        """
        Compute total potential energy.

        Args:
            x (np.ndarray): Current positions

        Returns:
            float: Computed potential energy

        Raises:
            SimulationError: If potential computation fails
        """
        try:
            logger.debug(f"Starting computation with input: {x}")
            U = self._compute_elastic_energy(x)
            _, BX, BXdot = self._compute_velocity_surface(x)
            self._setup_constraints(BX)
            EB, EF = self._compute_constraint_energies(BX, BXdot)
            energy = self._compute_total_energy(x, U, EB, EF)
            logger.info("Potential energy computed successfully")
            logger.info(f"Potential energy: {energy}")
            logger.info(f"Elastic potential: {U}")
            logger.info(f"Barrier potential: {EB}")
            logger.info(f"Friction potential: {EF}")
            logger.info(f"Barrier stiffness: {self.params.kB}")
            logger.info(f"Friction stiffness: {self.params.mu}")
            return energy
        except SimulationError as se:
            logger.error(f"Simulation error occurred: {se}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in potential computation: {e}")
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP,
                "Unexpected error in potential computation",
                str(e),
            )


class PotentialFactory(metaclass=SingletonMeta):
    """Factory class for creating potential instances."""

    _instances = {}

    @staticmethod
    def create(params: ParametersBase) -> Any:
        """
        Create and return a default potential instance or update the existing instance.

        Args:
            params (ParametersBase): Parameters for potential initialization.

        Returns:
            Any: Instance of the Potential class.

        Raises:
            SimulationError: If creation fails.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating potential..."))

        instance_key = "default"

        try:
            if instance_key in PotentialFactory._instances:
                # Update the existing instance with new parameters
                instance = PotentialFactory._instances[instance_key]
                instance.update_params(params)
                logger.info("Potential instance updated with new parameters.")
            else:
                # Create a new instance if not already present
                if params is None:
                    raise ValueError("Parameters required for potential creation")

                logger.debug(f"Parameters received for potential creation: {params}")
                instance = Potential(params)
                PotentialFactory._instances[instance_key] = instance
                logger.info(
                    SimulationLogMessageCode.COMMAND_SUCCESS.details(
                        "Potential created successfully."
                    )
                )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Failed to create or update potential: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP, "Failed to create or update potential", str(e)
            )

        return instance
