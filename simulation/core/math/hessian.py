import logging
from abc import ABC, abstractmethod
from typing import Any

import ipctk
import numpy as np
import scipy as sp

from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)

class HessianBase(ABC):
    def __init__(self, params: Parameters):
        """
        Initialize the HessianBase with parameters.

        Args:
            params (Parameters): The parameters for the Hessian computation.
        """
        self.params = params

    @abstractmethod
    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        """
        Compute the Hessian matrix.

        Args:
            x (np.ndarray): Current positions.

        Returns:
            sp.sparse.csc_matrix: The computed Hessian matrix.
        """
        pass


class Hessian(HessianBase):
    def __init__(self, params: Parameters):
        super().__init__(params)

    def update_params(self, params: Parameters):
        """
        Update the parameters used by the Hessian instance.

        Args:
            params (Parameters): The new parameters to set.
        """
        if not isinstance(params, Parameters):
            raise ValueError("params must be an instance of Parameters.")
        self.params = params
        logger.info("Hessian parameters updated successfully.")

    def _ensure_float64(self, matrix: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
        """Ensure that sparse matrix has dtype float64."""
        if matrix.dtype != np.float64:
            logger.debug(f"Converting matrix dtype from {matrix.dtype} to float64.")
            return matrix.astype(np.float64)
        return matrix

    def _compute_elastic_hessian(self, x: np.ndarray, hep):
        """Compute elastic energy Hessian."""
        try:
            hep.compute_element_elasticity(x, grad=False, hessian=True)
            return self._ensure_float64(hep.hessian())
        except Exception as e:
            logger.error(f"Failed to compute elastic Hessian: {e}")
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP, 
                "Failed to compute elastic Hessian", 
                str(e)
            )

    def _compute_velocity_surface(self, x: np.ndarray) -> tuple:
        """Compute velocity and surface mappings."""
        try:
            v = (x - self.params.xt) / self.params.dt
            BX = to_surface(x, self.params.mesh, self.params.cmesh)
            BXdot = to_surface(v, self.params.mesh, self.params.cmesh)
            return v, BX, BXdot
        except Exception as e:
            logger.error(f"Failed to compute velocity/surface mappings: {e}")
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP,
                "Failed to compute velocity/surface mappings",
                str(e)
            )

    def _compute_barrier_hessian(self, BX: np.ndarray) -> sp.sparse.csc_matrix:
        """Compute barrier potential Hessian."""
        try:
            HB = self.params.barrier_potential.hessian(
                self.params.cconstraints,
                self.params.cmesh,
                BX,
                project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS,
            )
            return self._ensure_float64(self.params.cmesh.to_full_dof(HB))
        except Exception as e:
            logger.error(f"Failed to compute barrier Hessian: {e}")
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP,
                "Failed to compute barrier Hessian",
                str(e)
            )

    def _compute_friction_hessian(self, BXdot: np.ndarray) -> sp.sparse.csc_matrix:
        """Compute friction potential Hessian."""
        try:
            HF = self.params.friction_potential.hessian(
                self.params.fconstraints,
                self.params.cmesh,
                BXdot,
                project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS,
            )
            return self._ensure_float64(self.params.cmesh.to_full_dof(HF))
        except Exception as e:
            logger.error(f"Failed to compute friction Hessian: {e}")
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP,
                "Failed to compute friction Hessian",
                str(e)
            )

    def _assemble_hessian(
        self,
        HU: sp.sparse.csc_matrix,
        HB: sp.sparse.csc_matrix,
        HF: sp.sparse.csc_matrix,
    ) -> sp.sparse.csc_matrix:
        """Assemble final Hessian matrix."""
        try:
            logger.info("Assembling Hessian matrix...")
            H = self.params.M + self.params.dt2 * HU + self.params.kB * HB + HF
            return self._ensure_float64(H)
        except Exception as e:
            logger.error(f"Failed to assemble Hessian: {e}")
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP,
                "Failed to assemble Hessian",
                str(e)
            )

    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        """Compute Hessian matrix with error handling."""
        try:
            if not isinstance(x, np.ndarray):
                raise ValueError("Input x must be a numpy array.")

            # Compute components with error handling
            HU = self._compute_elastic_hessian(x, self.params.hep)
            _, BX, BXdot = self._compute_velocity_surface(x)
            HB = self._compute_barrier_hessian(BX)
            HF = self._compute_friction_hessian(BXdot)
            H = self._assemble_hessian(HU, HB, HF)

            logger.info("Hessian matrix computed successfully.")
            logger.info(f"Hessian matrix: {H}")
            logger.info(f"Barrier stiffness: {self.params.kB}")
            logger.info(f"Friction stiffness: {self.params.mu}")
            logger.info(f"Friction potential: {self.params.friction_potential}")
            logger.info(f"Barrier potential: {self.params.barrier_potential}")
            return H

        except SimulationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error computing Hessian: {e}")
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP,
                "Unexpected error computing Hessian",
                str(e)
            )


class HessianFactory(metaclass=SingletonMeta):
    """Factory class for creating Hessian instances."""

    _instances = {}

    @staticmethod
    def create(params: ParametersBase) -> Any:
        """
        Create and return a default Hessian instance or update the existing instance.

        Args:
            params (ParametersBase): Parameters for Hessian initialization.

        Returns:
            Any: Instance of the Hessian class.

        Raises:
            SimulationError: If creation fails.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating Hessian..."))

        instance_key = "default"

        try:
            if instance_key in HessianFactory._instances:
                # Update the existing instance with new parameters
                instance = HessianFactory._instances[instance_key]
                instance.update_params(params)
                logger.info("Hessian instance updated with new parameters.")
            else:
                # Create a new instance if not already present
                if params is None:
                    raise ValueError("Parameters required for Hessian creation")

                instance = Hessian(params)
                HessianFactory._instances[instance_key] = instance
                logger.info(
                    SimulationLogMessageCode.COMMAND_SUCCESS.details(
                        "Hessian created successfully."
                    )
                )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Failed to create or update Hessian: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP, "Failed to create or update Hessian", str(e)
            )

        return instance
