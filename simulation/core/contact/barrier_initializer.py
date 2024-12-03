import logging
from abc import ABC, abstractmethod

import ipctk
import numpy as np

from simulation.core.parameters import ParametersBase
from simulation.core.utils.modifier.mesh import to_surface
from simulation.core.utils.singleton import SingletonMeta

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


class DefaultBarrierInitializer(BarrierInitializerBase):
    def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray) -> None:
        # Implement a basic default initialization
        pass


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

        logger.info(f"Initialized barrier stiffness: kB={kB}, maxkB={maxkB}, dprev={dprev}")


class BarrierInitializerFactory(metaclass=SingletonMeta):
    """
    Factory class for creating barrier initializer instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a barrier initializer instance based on the configuration.

        Args:
            config: A dictionary containing the barrier initializer configuration.

        Returns:
            An instance of the barrier initializer class.

        Raises:
            ValueError: 
        """
        logger.info("Creating barrier initializer...")
        barrier_initializer_config = config.get("barrier_initializer", {})
        barrier_initializer_type = barrier_initializer_config.get("type", "default").lower()

        if barrier_initializer_type not in BarrierInitializerFactory._instances:
            if barrier_initializer_type == "default":
                barrier_initializer_instance = DefaultBarrierInitializer()
            else:
                raise ValueError(f"Unknown barrier initializer type: {barrier_initializer_type}")

            BarrierInitializerFactory._instances[barrier_initializer_type] = barrier_initializer_instance

        return BarrierInitializerFactory._instances[barrier_initializer_type]