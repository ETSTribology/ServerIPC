from simulation.controller.history import CommandHistory
from simulation.controller.model import CommandType
from simulation.controller.commands import (
    ICommand,
    StartCommand,
    PauseCommand,
    StopCommand,
    ResumeCommand,
    ResetCommand,
    KillCommand,
    SendDataCommand,
    UpdateParameterCommand,
    GetBackendStatusCommand,
)
from typing import Dict, Type, Optional

import logging

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
        CommandType.SEND: SendDataCommand,
        CommandType.UPDATE_PARAMS: UpdateParameterCommand,
        CommandType.STATUS: GetBackendStatusCommand,
    }

    def __init__(self, history: CommandHistory, backend, simulation_manager):
        """
        Args:
            history (CommandHistory): Command execution history.
            backend: The backend instance for handling data.
            simulation_manager: The simulation manager instance for parameter updates.
        """
        self.history = history
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
        try:
            command_enum = CommandType.get_by_name(command_type)
            command_class = self._command_class_mapping.get(command_enum)

            if not command_class:
                logger.error(f"No command class mapped for CommandType '{command_enum.name}'")
                return None

            # Pass additional dependencies based on command requirements
            if issubclass(command_class, SendDataCommand) or issubclass(command_class, GetBackendStatusCommand):
                return command_class(history=self.history, backend=self.backend)
            if issubclass(command_class, UpdateParameterCommand):
                return command_class(history=self.history, simulation_manager=self.simulation_manager)

            return command_class(history=self.history, **kwargs)

        except Exception as e:
            logger.error(f"Unexpected error when creating command '{command_type}': {e}")

        return None
