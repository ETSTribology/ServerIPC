import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import ipctk
import numpy as np
import pbatoolkit as pbat
import scipy as sp
from scipy.sparse import dia_array

from simulation.core.integrator import IntegratorFactory
from simulation.core.modifier.mesh import to_surface
from simulation.core.utils.singleton import SingletonMeta
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class ParametersBase(ABC):
    """Abstract base class for parameters."""

    pass


class Parameters(ParametersBase):
    def __init__(
        self,
        mesh: pbat.fem.Mesh,
        config: Dict[str, Any],
        dt: float,
        xt: np.ndarray,
        vt: np.ndarray,
        a: np.ndarray,
        M: dia_array,
        hep: pbat.fem.HyperElasticPotential,
        cmesh: Optional[ipctk.CollisionMesh] = None,
        cconstraints: Optional[ipctk.NormalCollisions] = None,
        fconstraints: Optional[ipctk.TangentialCollisions] = None,
        materials: Optional[List] = [],
        element_materials: Optional[List] = [],
        barrier_potential: Optional[ipctk.BarrierPotential] = None,
        friction_potential: Optional[ipctk.FrictionPotential] = None,
        broad_phase_method: ipctk.BroadPhaseMethod = ipctk.BroadPhaseMethod.BRUTE_FORCE,
    ):
        """Initialize the parameters."""
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Initializing parameters..."))

        try:
            self.validate_inputs(xt, vt, a, dt, M)
            self.store_attributes(
                mesh,
                config,
                xt,
                vt,
                a,
                M,
                hep,
                cmesh,
                cconstraints,
                fconstraints,
                materials,
                element_materials,
            )
            self.setup_broad_phase_method(broad_phase_method)
            self.setup_integrator(config)
            self.initialize_derived_values(dt, a, M)
            self.setup_potentials(barrier_potential, friction_potential)

            logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    "Parameters initialized successfully."
                )
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Failed to initialize parameters: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.PARAMETERS_SETUP, "Failed to initialize parameters", str(e)
            )

    def validate_inputs(
        self, xt: np.ndarray, vt: np.ndarray, a: np.ndarray, dt: float, M: dia_array
    ):
        """Validate the inputs to the parameters."""
        try:
            logger.debug("Validating inputs...")

            # Check array shapes explicitly
            if xt.shape != vt.shape:
                raise ValueError(f"xt shape {xt.shape} does not match vt shape {vt.shape}")
            if xt.shape != a.shape:
                raise ValueError(f"xt shape {xt.shape} does not match a shape {a.shape}")

            # Validate dt
            if not isinstance(dt, (float, int)) or dt <= 0:
                raise ValueError("dt must be a positive number")

            # Validate mass matrix
            if not sp.sparse.issparse(M):
                raise ValueError("M must be a sparse matrix")

            # Check mass matrix dimensions against position vector
            if M.shape[0] != xt.size:
                raise ValueError(
                    f"Mass matrix dimensions {M.shape} incompatible with position vector size {xt.size}"
                )

            logger.debug("Input validation successful")

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise

    def store_attributes(
        self,
        mesh: pbat.fem.Mesh,
        config: Dict[str, Any],
        xt: np.ndarray,
        vt: np.ndarray,
        a: np.ndarray,
        M: dia_array,
        hep: pbat.fem.HyperElasticPotential,
        cmesh: Optional[ipctk.CollisionMesh],
        cconstraints: Optional[ipctk.NormalCollisions],
        fconstraints: Optional[ipctk.TangentialCollisions],
        materials: Optional[List],
        element_materials: Optional[List],
    ):
        """Store attributes after validation."""
        logger.debug("Storing attributes...")

        # Store mesh
        try:
            self.mesh = mesh
        except Exception as e:
            logger.error(f"Error storing mesh: {e}")
            raise

        # Store configuration and contact parameters
        try:
            self.config = config
            self.dhat = self.config.get("contact", {}).get("dhat", 0.0)
            self.dmin = self.config.get("contact", {}).get("dmin", 0.0)
            self.mu = self.config.get("contact", {}).get("friction", 0.0)
            self.epsv = self.config.get("contact", {}).get("epsv", 0.0)
        except Exception as e:
            logger.error(f"Error storing configuration parameters: {e}")
            raise

        # Store state vectors
        try:
            self.xt = xt
            self.vt = vt
            self.a = a
        except Exception as e:
            logger.error(f"Error storing state vectors: {e}")
            raise

        # Store mass matrix and potential
        try:
            self.M = M
            self.hep = hep
        except Exception as e:
            logger.error(f"Error storing mass matrix and potential: {e}")
            raise

        # Store collision-related attributes
        try:
            self.cmesh = cmesh
            self.cconstraints = cconstraints
            self.fconstraints = fconstraints
        except Exception as e:
            logger.error(f"Error storing collision attributes: {e}")
            raise

        # Store materials
        try:
            logger.debug("Storing materials...")

            # Handle materials list
            self.materials = [] if materials is None else list(materials)

            # Handle element materials list/array
            if element_materials is None:
                self.element_materials = []
            else:
                # Convert to list if numpy array
                self.element_materials = (
                    element_materials.tolist()
                    if isinstance(element_materials, np.ndarray)
                    else list(element_materials)
                )

        except Exception as e:
            logger.error(f"Error storing material attributes: {e}")
            raise

    def setup_broad_phase_method(self, broad_phase_method: ipctk.BroadPhaseMethod):
        """Configure the broad phase method."""
        try:
            self.broad_phase_method = broad_phase_method
        except Exception as e:
            logger.error(f"Broad phase method setup failed: {e}")
            raise

    def setup_integrator(self, config: Dict[str, Any]):
        """Setup the integrator based on the configuration."""
        try:
            integrator_type = (
                config.get("time", {}).get("integrator", {}).get("type", "implicit_euler")
            )
            self.integrator = IntegratorFactory.create_integrator(integrator_type)
        except Exception as e:
            logger.error(f"Integrator setup failed: {e}")
            raise

    def initialize_derived_values(self, dt: float, a: np.ndarray, M: dia_array):
        """Initialize derived values for simulation."""
        try:
            self.dt = dt
            self.dt2 = dt**2
            self.xtilde, _ = self.integrator.step(self.xt, self.vt, a, dt)
            self.avgmass = float(M.diagonal().mean())
        except Exception as e:
            logger.error(f"Error initializing derived values: {e}")
            raise

    def setup_potentials(
        self,
        barrier_potential: Optional[ipctk.BarrierPotential],
        friction_potential: Optional[ipctk.FrictionPotential],
    ):
        """Setup barrier and friction potentials."""
        try:
            self.kB = None
            self.maxkB = None
            self.dprev = None
            self.dcurrent = None
            self.BX = to_surface(self.xt, self.mesh, self.cmesh)
            self.bboxdiag = ipctk.world_bbox_diagonal_length(self.BX)
            self.gU = None
            self.gB = None
            self.barrier_potential = barrier_potential
            self.friction_potential = friction_potential

        except Exception as e:
            logger.error(f"Error setting up potentials: {e}")
            raise

    def update(self, **updates):
        """
        Update parameters dynamically.

        Args:
            updates: Dictionary of attributes to update.

        Raises:
            ValueError: If an update key is invalid.
        """
        logger.info("Updating parameters...")
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated {key} to {value}")
            else:
                logger.warning(f"Attempted to update invalid attribute: {key}")
                raise ValueError(f"Invalid attribute: {key}")
        logger.info("Parameters updated successfully.")


class ParametersFactory(metaclass=SingletonMeta):
    """Factory for creating and managing Parameters instances."""

    _instances: Dict[str, Parameters] = {}

    @staticmethod
    def create(config: Dict[str, Any], **updates) -> Parameters:
        """
        Create or update a Parameters instance.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            updates: Additional attributes to update if the instance exists.

        Returns:
            Parameters: The created or updated Parameters instance.
        """
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating or updating parameters..."))

        parameters_type = config.get("parameters", {}).get("type", "default").lower()

        try:
            if parameters_type in ParametersFactory._instances:
                # Update existing instance
                instance = ParametersFactory._instances[parameters_type]
                instance.update(**updates)
                logger.info(f"Updated existing parameters of type '{parameters_type}'.")
            else:
                # Create a new instance
                instance = Parameters(
                    **{k: v for k, v in config.items() if k != "parameters"}, config=config
                )
                ParametersFactory._instances[parameters_type] = instance
                logger.info(f"Created new parameters of type '{parameters_type}'.")

            return instance

        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Failed to create or update parameters: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.PARAMETERS_SETUP, "Failed to create or update parameters", str(e)
            )
