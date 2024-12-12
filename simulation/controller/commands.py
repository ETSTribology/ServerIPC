import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from typing import Callable, Optional

from simulation.controller.history import CommandHistory, CommandHistoryEntry
from simulation.controller.model import (
    CannotConnectError,
    CommandError,
    CommandFailedError,
    CommandType,
    InvalidParametersError,
    Request,
    Response,
    Status,
)
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class SimulationStateMachine(Enum):
    """Tracks valid simulation states"""

    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


def validate_state(valid_states: list[SimulationStateMachine]):
    """Decorator to validate simulation state before executing command"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.current_state not in valid_states:
                error_message = (
                    f"Invalid state {self.current_state.name} for {self.__class__.__name__}"
                )
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(error_message))
                raise CommandFailedError(error_message)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class ICommand(ABC):
    @abstractmethod
    def execute(self, request: Request) -> Response:
        pass

    @abstractmethod
    def undo(self) -> None:
        pass

    @abstractmethod
    def log(self, request: Request, response: Response) -> None:
        pass


class BaseCommand(ICommand):
    def __init__(self, history: CommandHistory):
        self.history = history
        self.current_state = SimulationStateMachine.STOPPED
        self.previous_state: Optional[SimulationStateMachine] = None

    def execute(self, request: Request) -> Response:
        """Template method for command execution"""
        try:
            response = self._execute_impl(request)
            self.log_history(request, response)
            return response
        except CommandError as e:
            error_response = Response(
                request_id=request.request_id, status=e.__class__.__name__, message=e.message
            )
            self.log_history(request, error_response)
            return error_response
        except Exception as e:
            error_response = Response(
                request_id=request.request_id,
                status=Status.COMMAND_FAILED.value,
                message=f"Unexpected error: {str(e)}",
            )
            self.log_history(request, error_response)
            return error_response

    @abstractmethod
    def _execute_impl(self, request: Request) -> Response:
        """Implementation specific to each command"""
        pass

    def undo(self) -> None:
        """Restore previous state if available"""
        if self.previous_state:
            self.current_state = self.previous_state
            self.previous_state = None

    def log_history(self, request: Request, response: Response):
        entry = CommandHistoryEntry(
            timestamp=datetime.now().isoformat(),
            command_name=self.__class__.__name__,
            request_id=request.request_id,
            status=response.status,
            message=response.message,
        )
        self.history.add_entry(entry)

    def _update_state(self, new_state: SimulationStateMachine):
        """Update simulation state with history"""
        self.previous_state = self.current_state
        self.current_state = new_state


class StartCommand(BaseCommand):
    @validate_state([SimulationStateMachine.STOPPED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            # Simulate connection logic
            success = self._connect()
            if not success:
                error_message = "Failed to connect to simulation engine."
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(error_message))
                raise CannotConnectError(error_message)
            self._update_state(SimulationStateMachine.RUNNING)
            message = CommandType.START.description
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(message))
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except CannotConnectError as e:
            raise e
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(str(e))

    def _connect(self) -> bool:
        # Placeholder for actual connection logic
        return True


class PauseCommand(BaseCommand):
    @validate_state([SimulationStateMachine.RUNNING])
    def _execute_impl(self, request: Request) -> Response:
        try:
            self._update_state(SimulationStateMachine.PAUSED)
            message = CommandType.PAUSE.description
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(message))
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(str(e))


class StopCommand(BaseCommand):
    @validate_state([SimulationStateMachine.RUNNING, SimulationStateMachine.PAUSED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            self._update_state(SimulationStateMachine.STOPPED)
            message = CommandType.STOP.description
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(message))
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(str(e))


class ResumeCommand(BaseCommand):
    @validate_state([SimulationStateMachine.PAUSED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            self._update_state(SimulationStateMachine.RUNNING)
            message = CommandType.RESUME.description
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(message))
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(str(e))


class ResetCommand(BaseCommand):
    def __init__(self, history: CommandHistory, reset_function: Callable[[], None]):
        super().__init__(history)
        self.reset_function = reset_function

    def _execute_impl(self, request: Request) -> Response:
        try:
            self.reset_function()
            self._update_state(SimulationStateMachine.STOPPED)
            message = "Simulation reset complete."
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(message))
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(f"Reset failed: {str(e)}")


class KillCommand(BaseCommand):
    def _execute_impl(self, request: Request) -> Response:
        try:
            message = "Simulation killed."
            response = Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
            self.log_history(request, response)
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(message))
            sys.exit(0)
            return response
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(str(e))


class SendDataCommand(BaseCommand):
    """Command to send data (e.g., meshes) to the backend."""

    @validate_state([SimulationStateMachine.RUNNING, SimulationStateMachine.PAUSED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            data = request.parameters.get("data")
            key = request.parameters.get("key")

            if not key or data is None:
                raise InvalidParametersError("Key and data must be provided.")

            self.backend.write(key, data)

            logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Data sent successfully with key: {key}"
                )
            )
            return Response(
                request_id=request.request_id,
                status=Status.SUCCESS.value,
                message=f"Data sent successfully with key: {key}",
            )
        except InvalidParametersError as e:
            raise e
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(str(e))


class UpdateParameterCommand(BaseCommand):
    """Command to update parameters in the simulation state."""

    @validate_state([SimulationStateMachine.RUNNING, SimulationStateMachine.PAUSED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            param_key = request.parameters.get("key")
            param_value = request.parameters.get("value")

            if not param_key:
                raise InvalidParametersError("Parameter key must be provided.")

            self.simulation_manager.update_params(param_key, param_value)

            logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Parameter '{param_key}' updated successfully."
                )
            )
            return Response(
                request_id=request.request_id,
                status=Status.SUCCESS.value,
                message=f"Parameter '{param_key}' updated successfully.",
            )
        except InvalidParametersError as e:
            raise e
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(str(e)))
            raise CommandFailedError(str(e))
