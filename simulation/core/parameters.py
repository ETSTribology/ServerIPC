import logging
from abc import ABC
from functools import lru_cache

import ipctk
import numpy as np
import pbatoolkit as pbat
import scipy as sp
from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register
from simulation.core.utils.modifier.mesh import to_surface

logger = logging.getLogger(__name__)


class ParametersBase(ABC):
    def __init__(self, *args, **params):
        pass


registry_container = RegistryContainer()
registry_container.add_registry("parameters", "simulation.core.parameters.ParametersBase")


@register(type="parameters", name="default")
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
        cmesh: ipctk.CollisionMesh,
        cconstraints: ipctk.NormalCollisions,
        fconstraints: ipctk.TangentialCollisions,
        materials: list,
        element_materials: list,
        dhat: float = 1e-3,
        dmin: float = 1e-4,
        mu: float = 0.3,
        epsv: float = 1e-4,
        barrier_potential: ipctk.BarrierPotential = None,
        friction_potential: ipctk.FrictionPotential = None,
        broad_phase_method: ipctk.BroadPhaseMethod = ipctk.BroadPhaseMethod.SWEEP_AND_PRUNE,
    ):
        logger.info("Initializing parameters...")
        self.mesh = mesh
        self.xt = xt
        self.vt = vt
        self.a = a
        self.M = M
        self.hep = hep
        self.dt = dt
        self.cmesh = cmesh
        self.cconstraints = cconstraints
        self.fconstraints = fconstraints
        self.dhat = dhat
        self.dmin = dmin
        self.mu = mu
        self.epsv = epsv

        self.dt2 = dt**2
        self.xtilde = xt + dt * vt + self.dt2 * a
        self.avgmass = M.diagonal().mean()
        self.kB = None
        self.maxkB = None
        self.dprev = None
        self.dcurrent = None
        BX = to_surface(xt, mesh, cmesh)
        self.bboxdiag = ipctk.world_bbox_diagonal_length(BX)
        self.gU = None
        self.gB = None
        self.gF = None
        self.materials = materials
        self.element_materials = element_materials
        self.barrier_potential = barrier_potential
        self.friction_potential = friction_potential
        self.broad_phase_method = broad_phase_method

        logger.info("Parameters initialized.")

    def reset(self):
        # Reset positions, velocities, accelerations to initial state
        self.xt = self.initial_xt.copy()
        self.vt = self.initial_vt.copy()
        self.a = self.initial_a.copy()

    def get_initial_state(self):
        return self.xt.copy(), self.vt.copy(), self.a.copy()

    def set_initial_state(self, xt, vt, a):
        self.xt = xt.copy()
        self.vt = vt.copy()
        self.a = a.copy()


class ParametersFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Parameters class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.parameters.get(type_lower)

    @staticmethod
    def create(type: str, *args, **kwargs) -> ParametersBase:
        """Factory method to create a Parameters configuration based on the provided type.

        :param type: The type of Parameters configuration to create.
        :param args: Positional arguments for the Parameters class.
        :param kwargs: Keyword arguments for the Parameters class.
        :return: An instance of the specified Parameters configuration.
        """
        type_lower = type.lower()
        logger.debug(f"Creating Parameters configuration '{type_lower}'.")
        try:
            # Retrieve the Parameters class from the generalized registry
            registry_container = RegistryContainer()
            parameters_cls = registry_container.parameters.get(type_lower)

            # Create an instance of the Parameters class
            parameters_instance = parameters_cls(*args, **kwargs)
            logger.info(
                f"Parameters configuration '{type_lower}' created successfully using class '{parameters_cls.__name__}'."
            )
            return parameters_instance
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(
                f"Failed to create Parameters configuration '{type_lower}': {e}"
            )
            raise RuntimeError(
                f"Error during Parameters initialization for method '{type_lower}': {e}"
            ) from e
