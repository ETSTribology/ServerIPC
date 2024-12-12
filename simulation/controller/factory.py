import logging
from typing import Dict, Optional, Type

from simulation.controller.commands import (
    ICommand,
    KillCommand,
    PauseCommand,
    ResetCommand,
    ResumeCommand,
    StartCommand,
    StopCommand,
)
from simulation.controller.history import CommandHistory
from simulation.controller.model import CommandType
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)


class CommandFactory:
    """
    Factory class for creating command instances based on CommandType enum.
    """

    _command_class_mapping: Dict[CommandType, Type[ICommand]] = {
        CommandType.START: StartCommand,
        CommandType.PAUSE: PauseCommand,
        CommandType.STOP: StopCommand,
        CommandType.RESUME: ResumeCommand,
        CommandType.RESET: ResetCommand,
        CommandType.KILL: KillCommand,
    }

    def __init__(self, history: CommandHistory):
        self.history = history

    def create(self, command_type: str, **kwargs) -> Optional[ICommand]:
        """
        Create a command instance based on the command type.

        Args:
            command_type (str): The type of command to create (name or alias).
            **kwargs: Additional keyword arguments for command initialization.

        Returns:
            Optional[ICommand]: An instance of the specified command type or None if creation fails.
        """
        try:
            # Retrieve the CommandType enum member using the provided command_type string
            command_enum = CommandType.get_by_name(command_type)
            logger.debug(SimulationLogMessageCode.COMMAND_INITIALIZED.details(f"Command enum retrieved: {command_enum.name}"))

            # Get the corresponding command class from the mapping
            command_class = self._command_class_mapping.get(command_enum)

            if not command_class:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"No command class mapped for CommandType '{command_enum.name}'"))
                return None

            # If the command requires additional arguments, pass them
            if command_enum == CommandType.RESET:
                if "reset_function" not in kwargs:
                    logger.error(SimulationLogMessageCode.COMMAND_FAILED.details("ResetCommand requires a 'reset_function' argument"))
                    return None

            # Instantiate and return the command class
            return command_class(history=self.history, **kwargs)

        except ValueError as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Command creation failed: {e}"))
        except TypeError as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to instantiate command '{command_type}': {e}"))
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unexpected error when creating command '{command_type}': {e}"))

        return None