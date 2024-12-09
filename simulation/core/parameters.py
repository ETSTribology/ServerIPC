import logging
from abc import ABC
from typing import Any, Dict

import ipctk
import numpy as np
import pbatoolkit as pbat
import scipy as sp

from simulation.core.utils.singleton import SingletonMeta
from simulation.core.modifier.mesh import to_surface

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
        cmesh: ipctk.CollisionMesh = None,
        cconstraints: ipctk.NormalCollisions = None,
        fconstraints: ipctk.TangentialCollisions = None,
        materials: list = None,
        element_materials: list = None,
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
        self.materials = materials or []
        self.element_materials = element_materials or []

        # Create BarrierPotential with dhat parameter
        self.barrier_potential = (
            barrier_potential
            if barrier_potential is not None
            else ipctk.BarrierPotential(dhat=dhat)
        )

        # Create FrictionPotential with eps_v parameter
        self.friction_potential = (
            friction_potential
            if friction_potential is not None
            else ipctk.FrictionPotential(eps_v=epsv)
        )

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

class ParametersFactory(metaclass=SingletonMeta):
    _instance = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a potential instance based on the configuration.

        Args:
            config: A dictionary containing the potential configuration.

        Returns:
            An instance of the potential class.

        Raises:
            ValueError: 
        """
        logger.info("Creating parameters...")
        parameters_config = config.get("parameters", {})
        parameters_type = parameters_config.get("type", "default").lower()

        if parameters_type not in ParametersFactory._instances:
            if parameters_type == "default":
                parameters_instance = Parameters(config)
            else:
                raise ValueError(f"Unknown parameters type: {parameters_type}")

            ParametersFactory._instances[parameters_type] = parameters_instance

        return ParametersFactory._instances[parameters_type]