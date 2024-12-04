import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from simulation.core.contact.ccd import CCD
from simulation.core.utils.singleton import SingletonMeta
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.modifier.mesh import to_surface
logger = logging.getLogger(__name__)


class PotentialBase(ABC):
    def __init__(self, params: Parameters):
        self.params = params

    @abstractmethod
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


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

class PotentialFactory(metaclass=SingletonMeta):
    """
    Factory class for creating potential instances.
    """

    _instances = {}

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
        logger.info("Creating potential...")
        potential_config = config.get("potential", {})
        potential_type = potential_config.get("type", "default").lower()

        if potential_type not in PotentialFactory._instances:
            if potential_type == "default":
                potential_instance = Potential(config)
            else:
                raise ValueError(f"Unknown potential type: {potential_type}")

            PotentialFactory._instances[potential_type] = potential_instance

        return PotentialFactory._instances[potential_type]