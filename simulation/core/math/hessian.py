import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import ipctk
import numpy as np
import scipy as sp

from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

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
    def __init__(self, params: ParametersBase):
        """
        Initialize the Hessian with parameters.

        Args:
            params (ParametersBase): The parameters for the Hessian computation.
        """
        self.params = params

    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        """
        Compute the Hessian matrix.

        Args:
            x (np.ndarray): Current positions.

        Returns:
            sp.sparse.csc_matrix: The computed Hessian matrix.
        """
        try:
            logger.debug(SimulationLogMessageCode.COMMAND_STARTED.details("Computing Hessian matrix."))
            dt = self.params.dt
            dt2 = self.params.dt2
            xt = self.params.xt
            M = self.params.M
            hep = self.params.hep
            mesh = self.params.mesh
            cmesh = self.params.cmesh
            cconstraints = self.params.cconstraints
            fconstraints = self.params.fconstraints
            dhat = self.params.dhat
            dmin = self.params.dmin
            mu = self.params.mu
            epsv = self.params.epsv
            kB = self.params.kB
            B = self.params.barrier_potential
            D = self.params.friction_potential

            hep.compute_element_elasticity(x, grad=False, hessian=True)
            HU = hep.hessian()
            v = (x - xt) / dt
            BX = to_surface(x, mesh, cmesh)
            BXdot = to_surface(v, mesh, cmesh)
            # Compute the Hessian of the barrier potential using the correct signature
            HB = B.hessian(
                cconstraints,
                cmesh,
                BX,
                project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS,
            )
            HB = cmesh.to_full_dof(HB)

            # Compute the Hessian of the friction dissipative potential
            HF = D.hessian(
                fconstraints,
                cmesh,
                BXdot,
                project_hessian_to_psd=ipctk.PSDProjectionMethod.ABS,
            )
            HF = cmesh.to_full_dof(HF)
            H = M + dt2 * HU + kB * HB + HF
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Hessian matrix computed successfully."))
            return H
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to compute Hessian matrix: {e}"))
            raise SimulationError(SimulationErrorCode.HESSIAN_SETUP, "Failed to compute Hessian matrix", details=str(e))


class HessianFactory(metaclass=SingletonMeta):
    """
    Factory class for creating Hessian instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a Hessian instance based on the configuration.

        Args:
            config: A dictionary containing the Hessian configuration.

        Returns:
            An instance of the Hessian class.

        Raises:
            SimulationError: If the Hessian type is unknown.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating Hessian..."))
        hessian_config = config.get("hessian", {})
        hessian_type = hessian_config.get("type", "default").lower()

        if hessian_type not in HessianFactory._instances:
            try:
                if hessian_type == "default":
                    hessian_instance = Hessian(config)
                else:
                    raise ValueError(f"Unknown Hessian type: {hessian_type}")

                HessianFactory._instances[hessian_type] = hessian_instance
                logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Hessian '{hessian_type}' created successfully."))
            except ValueError as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unknown Hessian type: {hessian_type}"))
                raise SimulationError(SimulationErrorCode.HESSIAN_SETUP, f"Unknown Hessian type: {hessian_type}", details=str(e))
            except Exception as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unexpected error when creating Hessian '{hessian_type}': {e}"))
                raise SimulationError(SimulationErrorCode.HESSIAN_SETUP, f"Unexpected error when creating Hessian '{hessian_type}'", details=str(e))

        return HessianFactory._instances[hessian_type]