import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import ipctk
import numpy as np
from core.parameters import ParametersBase
from core.registry.container import RegistryContainer
from core.registry.decorators import register
from core.utils.modifier.mesh import to_surface

logger = logging.getLogger(__name__)


class BarrierInitializerBase(ABC):
    def __init__(self, params: ParametersBase):
        self.params = params

    @abstractmethod
    def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray) -> None:
        """Initialize the barrier parameters based on the input data.

        Args:
            x (np.ndarray): Current positions.
            gU (np.ndarray): Gradient of the potential energy.
            gB (np.ndarray): Gradient of the barrier potential.

        """
        pass


registry_container = RegistryContainer()
registry_container.add_registry(
    "barrier_initializer", "core.contact.barrier_initializer.BarrierInitializerBase"
)


@register(type="barrier_initializer", name="default")
class BarrierInitializer(BarrierInitializerBase):
    def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray) -> None:
        params = self.params
        mesh = params.mesh
        cmesh = params.cmesh
        dhat = params.dhat
        dmin = params.dmin
        avgmass = params.avgmass
        bboxdiag = params.bboxdiag
        cconstraints = params.cconstraints
        B = params.barrier_potential

        BX = to_surface(x, mesh, cmesh)
        B(cconstraints, cmesh, BX)
        gB = B.gradient(cconstraints, cmesh, BX)
        kB, maxkB = ipctk.initial_barrier_stiffness(
            bboxdiag, B.barrier, dhat, avgmass, gU, gB, dmin=dmin
        )
        dprev = cconstraints.compute_minimum_distance(cmesh, BX)

        # Update parameters
        params.kB = kB
        params.maxkB = maxkB
        params.dprev = dprev

        logger.info(
            f"Initialized barrier stiffness: kB={kB}, maxkB={maxkB}, dprev={dprev}"
        )


class BarrierInitializerFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Barrier Initializer class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.barrier_initializer.get(type_lower)

    @staticmethod
    def create(type: str, params: ParametersBase) -> BarrierInitializerBase:
        """Create a Barrier Initializer instance based on the input type.

        Args:
            type (str): Name of the Barrier Initializer.
            params (ParametersBase): Simulation parameters.

        Returns:
            BarrierInitializerBase: Instance of the Barrier Initializer.

        """
        type_lower = type.lower()
        try:
            barrier_initializer_cls = BarrierInitializerFactory.get_class(type_lower)
            return barrier_initializer_cls(params)
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create Barrier Initializer '{type}': {e}")
            raise RuntimeError(
                f"Error during Barrier Initializer initialization for type '{type}': {e}"
            ) from e
