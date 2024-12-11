import logging
from typing import Callable

from simulation.controller.command import CommandRegistry
from simulation.controller.response import ResponseMessage
from simulation.controller.response import Status
from simulation.states.state import SimulationState

logger = logging.getLogger(__name__)

class CommandDispatcher:
    def __init__(self, simulation_state: SimulationState, reset_function: Callable[[], 'SimulationState']):
        self.simulation_state = simulation_state
        self.reset_function = reset_function

    def dispatch(self, command_name: str, request_id: str, **kwargs) -> ResponseMessage:
        """Dispatches the command to the appropriate handler.

        Args:
            command_name (str): The name or alias of the command.
            request_id (str): The unique identifier for the request.
            **kwargs: Additional arguments for the command execution.

        Returns:
            ResponseMessage: The response from executing the command.
        """
        try:
            command = CommandRegistry.get_command(
                name=command_name,
                reset_function=self.reset_function
            )
            response = command.execute(self.simulation_state, request_id, **kwargs)
            logger.info(f"Command '{command_name}' executed successfully.")
            return response
        except ValueError as e:
            logger.error(f"Dispatch error: {e}")
            return ResponseMessage(
                request_id=request_id,
                status=Status.ERROR,
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Error executing command '{command_name}': {e}")
            return ResponseMessage(
                request_id=request_id,
                status=Status.ERROR,
                message=str(e)
            )
