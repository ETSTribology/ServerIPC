import logging
from typing import Optional

from simulation.controller.commands.base import ICommand
from simulation.controller.commands.kill import KillCommand
from simulation.controller.commands.pause import PauseCommand
from simulation.controller.commands.reset import ResetCommand
from simulation.controller.commands.start import StartCommand
from simulation.controller.commands.stop import StopCommand
from simulation.controller.model import CommandType


logger = logging.getLogger(__name__)


class CommandFactory:
    """
    Factory class for creating command instances based on CommandType enum.
    """

    _command_class_mapping = {
        CommandType.START: StartCommand,
        CommandType.PAUSE: PauseCommand,
        CommandType.STOP: StopCommand,
        CommandType.RESET: ResetCommand,
        CommandType.KILL: KillCommand,
    }

    def __init__(self, backend, simulation_manager):
        """
        Args:
            backend: The backend instance for handling data.
            simulation_manager: The simulation manager instance for parameter updates.
        """
        self.backend = backend
        self.simulation_manager = simulation_manager

    def create(self, command_type: str, **kwargs) -> Optional[ICommand]:
        """
        Create a command instance based on the command type.

        Args:
            command_type (str): The type of command to create (name or alias).
            **kwargs: Additional keyword arguments for command initialization.

        Returns:
            Optional[ICommand]: An instance of the specified command type or None if creation fails.
        """
        logger.debug(f"Attempting to create command: {command_type}")
        try:
            command_enum = CommandType.get_by_name(command_type)
        except ValueError as e:
            logger.error(f"CommandFactory: Unknown command type '{command_type}'. Error: {e}")
            return None

        command_class = self._command_class_mapping.get(command_enum)
        if not command_class:
            logger.error(f"No command class mapped for CommandType '{command_enum.name}'")
            return None

        try:
            # Initialize the command with the required arguments
            return command_class(backend=self.backend, simulation_manager=self.simulation_manager, **kwargs)
        except TypeError as e:
            logger.error(f"Failed to initialize command '{command_type}': {e}")
            return None

