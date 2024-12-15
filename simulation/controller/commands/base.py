# simulation/commands/base.py
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from typing import Callable, Optional

from simulation.controller.errors import CommandError, CommandFailedError
from simulation.controller.model import Request, Response, Status

logger = logging.getLogger(__name__)


class SimulationStateMachine(Enum):
    """Tracks valid simulation states"""
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


def validate_state(valid_states: list):
    """Decorator to validate simulation state before executing command"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.current_state not in valid_states:
                error_message = (
                    f"Invalid state {self.current_state.name} for {self.__class__.__name__}"
                )
                logger.error(error_message)
                raise CommandFailedError(error_message)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class ICommand(ABC):
    @abstractmethod
    def execute(self, request: Request) -> Response:
        pass


class BaseCommand(ICommand):
    def __init__(self, backend, simulation_manager, **kwargs):
        """
        Base class for all commands.
        Args:
            backend: Backend instance for handling data.
            simulation_manager: Simulation manager instance for parameter updates.
            **kwargs: Additional arguments for customization.
        """
        self.backend = backend
        self.simulation_manager = simulation_manager
        self.extra_params = kwargs
        self.current_state = SimulationStateMachine.STOPPED
        self.previous_state: Optional[SimulationStateMachine] = None

    def execute(self, request: Request) -> Response:
        """Template method for command execution"""
        try:
            response = self._execute_impl(request)
            return response
        except CommandError as e:
            error_response = Response(
                request_id=request.request_id, status=e.__class__.__name__, message=e.message
            )
            return error_response
        except Exception as e:
            error_response = Response(
                request_id=request.request_id,
                status=Status.COMMAND_FAILED.value,
                message=f"Unexpected error: {str(e)}",
            )
            return error_response

    @abstractmethod
    def _execute_impl(self, request: Request) -> Response:
        """Implementation specific to each command"""
        pass
