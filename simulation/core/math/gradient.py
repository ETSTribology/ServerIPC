import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from simulation.core.contact.barrier_initializer import BarrierInitializer
from simulation.core.utils.singleton import SingletonMeta
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.modifier.mesh import to_surface

logger = logging.getLogger(__name__)


class GradientBase(ABC):
    def __init__(self, params: Parameters):
        self.params = params

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class Gradient(GradientBase):
    def __init__(self, params: ParametersBase):
        self.params = params
        self.gradU = None
        self.gradB = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        logger.debug("Computing gradient vector.")
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        xtilde = self.params.xtilde
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

        hep.compute_element_elasticity(x, grad=True, hessian=False)
        gU = hep.gradient()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)
        gB = B.gradient(cconstraints, cmesh, BX)
        gB = cmesh.to_full_dof(gB)

        # Cannot compute gradient without barrier stiffness
        if self.params.kB is None:
            logger.debug("Barrier stiffness not initialized. Initializing barrier parameters.")
            binit = BarrierInitializer(self.params)
            binit(x, gU, gB)
            kB = self.params.kB

        kB = self.params.kB
        BXdot = to_surface(v, mesh, cmesh)

        # Use the BarrierPotential in the build method
        fconstraints.build(cmesh, BX, cconstraints, B, kB, mu)

        friction_potential = D(fconstraints, cmesh, BXdot)
        gF = D.gradient(fconstraints, cmesh, BXdot)
        gF = cmesh.to_full_dof(gF)
        g = M @ (x - xtilde) + dt2 * gU + kB * gB + dt * gF
        return g


class GradientFactory(metaclass=SingletonMeta):
    """
    Factory class for creating gradient instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a gradient instance based on the configuration.

        Args:
            config: A dictionary containing the gradient configuration.

        Returns:
            An instance of the gradient class.

        Raises:
            ValueError: 
        """
        logger.info("Creating gradient...")
        gradient_config = config.get("gradient", {})
        gradient_type = gradient_config.get("type", "default").lower()

        if gradient_type not in GradientFactory._instances:
            if gradient_type == "default":
                gradient_instance = Gradient(config)
            else:
                raise ValueError(f"Unknown gradient type: {gradient_type}")

            GradientFactory._instances[gradient_type] = gradient_instance

        return GradientFactory._instances[gradient_type]