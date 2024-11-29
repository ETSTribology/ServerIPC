import logging
from typing import Callable

from nets.controller.command import CommandRegistry

logger = logging.getLogger(__name__)


class CommandDispatcher:
    def __init__(self, simulation_state: dict, reset_function: Callable[[], dict]):
        self.simulation_state = simulation_state
        self.reset_function = reset_function

    def dispatch(self, command_name: str, **kwargs) -> None:
        """Dispatches the command to the appropriate handler.

        Args:
            command_name (str): The name or alias of the command.
            **kwargs: Additional arguments for the command execution.

        """
        try:
            command = CommandRegistry.get_command(
                name=command_name, reset_function=self.reset_function
            )
            command.execute(self.simulation_state, **kwargs)
        except ValueError as e:
            logger.error(f"Dispatch error: {e}")
        except Exception as e:
            logger.error(f"Error executing command '{command_name}': {e}")
