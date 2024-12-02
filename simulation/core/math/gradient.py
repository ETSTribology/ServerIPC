import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from core.contact.barrier_initializer import BarrierInitializer
from core.parameters import Parameters, ParametersBase
from core.registry.container import RegistryContainer
from core.registry.decorators import register
from core.utils.modifier.mesh import to_surface

logger = logging.getLogger(__name__)


class GradientBase(ABC):
    def __init__(self, params: Parameters):
        self.params = params

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


registry_container = RegistryContainer()
registry_container.add_registry("gradient", "simulation.core.math.gradient.GradientBase")


@register(type="gradient", name="default")
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


class GradientFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Gradient class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.gradient.get(type_lower)

    @staticmethod
    def create(type: str, params: Parameters) -> GradientBase:
        """Creates a gradient computation object of the specified type."""
        type_lower = type.lower()
        try:
            gradient_cls = GradientFactory.get_class(type_lower)
            return gradient_cls(params)
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create gradient computation method '{type}': {e}")
            raise RuntimeError(
                f"Error during gradient initialization for method '{type}': {e}"
            ) from e
