import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import ipctk
import numpy as np
import pbatoolkit as pbat
import scipy as sp

from simulation.core.modifier.mesh import to_surface
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)


class ParametersBase(ABC):
    def __init__(self, *args, **params):
        pass


class Parameters(ParametersBase):
    def __init__(
        self,
        mesh: pbat.fem.Mesh,
        xt: np.ndarray,
        vt: np.ndarray,
        a: np.ndarray,
        M: sp.sparse.dia_array,
        hep: pbat.fem.HyperElasticPotential,
        dt: float,
        cmesh: Optional[ipctk.CollisionMesh] = None,
        cconstraints: Optional[ipctk.NormalCollisions] = None,
        fconstraints: Optional[ipctk.TangentialCollisions] = None,
        materials: Optional[List] = None,
        element_materials: Optional[List] = None,
        dhat: float = 1e-3,
        dmin: float = 1e-4,
        mu: float = 0.3,
        epsv: float = 1e-4,
        barrier_potential: Optional[ipctk.BarrierPotential] = None,
        friction_potential: Optional[ipctk.FrictionPotential] = None,
        broad_phase_method: ipctk.BroadPhaseMethod = ipctk.BroadPhaseMethod.SWEEP_AND_PRUNE,
    ):
        try:
            logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Initializing parameters..."))

            # Validate inputs
            if not isinstance(xt, np.ndarray) or not isinstance(vt, np.ndarray) or not isinstance(a, np.ndarray):
                raise ValueError("xt, vt, and a must be numpy arrays")

            if xt.shape != vt.shape or xt.shape != a.shape:
                raise ValueError("xt, vt, and a must have the same shape")

            if not isinstance(dt, (int, float)):
                raise ValueError("dt must be a number")

            if not sp.sparse.issparse(M):
                raise ValueError("M must be a sparse matrix")

            # Store attributes
            self.mesh = mesh
            self.xt = xt.copy()
            self.vt = vt.copy()
            self.a = a.copy()
            self.M = M
            self.hep = hep
            self.dt = float(dt)
            self.cmesh = cmesh
            self.cconstraints = cconstraints
            self.fconstraints = fconstraints
            self.dhat = float(dhat)
            self.dmin = float(dmin)
            self.mu = float(mu)
            self.epsv = float(epsv)

            # Compute derived values
            self.dt2 = self.dt * self.dt
            self.xtilde = self.xt + self.dt * self.vt + self.dt2 * self.a

            # Handle mass matrix
            self.avgmass = float(M.diagonal().mean())

            self.kB = None
            self.maxkB = None
            self.dprev = None
            self.dcurrent = None

            # Compute bounding box
            if self.mesh is not None and self.cmesh is not None:
                BX = to_surface(self.xt, self.mesh, self.cmesh)
                if isinstance(BX, np.ndarray) and BX.size > 1:
                    raise ValueError("Bounding box must be a scalar or single element array")
                self.bboxdiag = float(ipctk.world_bbox_diagonal_length(BX))
            else:
                self.bboxdiag = 0.0

            self.gU = None
            self.gB = None
            self.gF = None
            self.materials = materials or []
            self.element_materials = element_materials or []

            # Initialize initial states with copies
            self.initial_xt = self.xt.copy()
            self.initial_vt = self.vt.copy()
            self.initial_a = self.a.copy()

            # Create potentials if not provided
            self.barrier_potential = barrier_potential or ipctk.BarrierPotential(dhat=self.dhat)
            self.friction_potential = friction_potential or ipctk.FrictionPotential(eps_v=self.epsv)
            self.broad_phase_method = broad_phase_method

            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Parameters initialized successfully."))

        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to initialize parameters: {str(e)}"))
            raise SimulationError(
                SimulationErrorCode.PARAMETERS_SETUP,
                "Failed to initialize parameters",
                str(e)
            )

    def reset(self):
        """
        Reset positions, velocities, and accelerations to their initial state.
        """
        try:
            logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Resetting parameters to initial state."))
            self.xt = self.initial_xt.copy()
            self.vt = self.initial_vt.copy()
            self.a = self.initial_a.copy()
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Parameters reset to initial state."))
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to reset parameters: {e}"))
            raise SimulationError(SimulationErrorCode.PARAMETERS_SETUP, "Failed to reset parameters", details=str(e))

    def get_initial_state(self):
        """
        Get the initial state of positions, velocities, and accelerations.

        Returns:
            tuple: Initial positions, velocities, and accelerations.
        """
        try:
            logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Getting initial state of parameters."))
            return self.initial_xt.copy(), self.initial_vt.copy(), self.initial_a.copy()
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to get initial state: {e}"))
            raise SimulationError(SimulationErrorCode.PARAMETERS_SETUP, "Failed to get initial state", details=str(e))

    def set_initial_state(self, xt: np.ndarray, vt: np.ndarray, a: np.ndarray):
        """
        Set the initial state of positions, velocities, and accelerations.

        Args:
            xt (np.ndarray): Initial positions.
            vt (np.ndarray): Initial velocities.
            a (np.ndarray): Initial accelerations.
        """
        try:
            logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Setting initial state of parameters."))
            self.xt = xt.copy()
            self.vt = vt.copy()
            self.a = a.copy()
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Initial state of parameters set."))
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to set initial state: {e}"))
            raise SimulationError(SimulationErrorCode.PARAMETERS_SETUP, "Failed to set initial state", details=str(e))


class ParametersFactory(metaclass=SingletonMeta):
    _instances: Dict[str, Any] = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a Parameters instance based on the configuration.

        Args:
            config: A dictionary containing the parameters configuration.

        Returns:
            An instance of the Parameters class.

        Raises:
            SimulationError: If the parameters type is unknown.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating parameters..."))
        parameters_config = config.get("parameters", {})
        parameters_type = parameters_config.get("type", "default").lower()

        if parameters_type not in ParametersFactory._instances:
            try:
                if parameters_type == "default":
                    parameters_instance = Parameters(**config)
                else:
                    raise ValueError(f"Unknown parameters type: {parameters_type}")

                ParametersFactory._instances[parameters_type] = parameters_instance
                logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Parameters of type '{parameters_type}' created and cached."))
            except ValueError as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unknown parameters type: {parameters_type}"))
                raise SimulationError(SimulationErrorCode.PARAMETERS_SETUP, f"Unknown parameters type: {parameters_type}", details=str(e))
            except Exception as e:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to create parameters: {e}"))
                raise SimulationError(SimulationErrorCode.PARAMETERS_SETUP, "Failed to create parameters", details=str(e))
        else:
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Using cached parameters of type '{parameters_type}'."))

        return ParametersFactory._instances[parameters_type]
