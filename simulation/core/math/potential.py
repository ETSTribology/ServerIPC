import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from core.parameters import Parameters, ParametersBase
from core.registry.container import RegistryContainer
from core.registry.decorators import register
from core.utils.modifier.mesh import to_surface

logger = logging.getLogger(__name__)


class PotentialBase(ABC):
    def __init__(self, params: Parameters):
        self.params = params

    @abstractmethod
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


registry_container = RegistryContainer()
registry_container.add_registry("potential", "core.math.potential.PotentialBase")


@register(type="potential", name="default")
class Potential(PotentialBase):
    def __init__(self, params: ParametersBase):
        self.params = params

    def __call__(self, x: np.ndarray) -> float:
        logger.debug("Computing potential energy.")
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

        hep.compute_element_elasticity(x, grad=False, hessian=False)
        U = hep.eval()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)

        # Compute the barrier potential
        cconstraints.use_area_weighting = True
        cconstraints.use_improved_max_approximator = True
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)

        # Build friction constraints
        fconstraints.build(cmesh, BX, cconstraints, B, kB, mu)

        EB = B(cconstraints, cmesh, BX)
        EF = D(fconstraints, cmesh, BXdot)

        potential_energy = (
            0.5 * (x - xtilde).T @ M @ (x - xtilde) + dt**2 * U + kB * EB + dt**2 * EF
        )
        return potential_energy


class PotentialFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Potential class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.potential.get(type_lower)

    @staticmethod
    def create(type: str, params: ParametersBase) -> PotentialBase:
        """Factory method to create a potential energy computation object.

        :param type: The type of potential energy computation object to create.
        :param params: The simulation parameters.
        :return: A potential energy computation object.
        """
        type_lower = type.lower()
        logger.debug(
            f"Creating potential energy computation object of type '{type_lower}'."
        )
        try:
            # Retrieve the Potential class from the generalized registry
            registry_container = RegistryContainer()
            potential_cls = registry_container.potential.get(type_lower)

            # Create an instance of the Potential class
            potential_instance = potential_cls(params)
            logger.info(
                f"Potential energy computation method '{type_lower}' created successfully using class '{potential_cls.__name__}'."
            )
            return potential_instance
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(
                f"Failed to create potential energy computation method '{type_lower}': {e}"
            )
            raise RuntimeError(
                f"Error during potential initialization for method '{type_lower}': {e}"
            ) from e
