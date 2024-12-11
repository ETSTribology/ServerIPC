import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Type

from simulation.controller.response import Status
from simulation.controller.response import ResponseMessage
from simulation.states.state import SimulationState

logger = logging.getLogger(__name__)

class Command(ABC):
    @abstractmethod
    def execute(self, simulation_state: SimulationState, request_id: str, **kwargs) -> ResponseMessage:
        """Abstract method to execute the command.

        Args:
            simulation_state (SimulationState): The current simulation state.
            request_id (str): The unique identifier for the request.
            **kwargs: Additional parameters for the command execution.

        Returns:
            ResponseMessage: The response after executing the command.
        """
        pass

class CommandRegistry:
    _command_registry: Dict[str, Type[Command]] = {}

    @classmethod
    def register(cls, name: str, aliases: Optional[list] = None):
        """Decorator to register a command class with a given name and optional aliases.

        Args:
            name (str): The primary name of the command.
            aliases (list, optional): Additional aliases for the command.
        """
        def decorator(command_cls: Type[Command]):
            if not issubclass(command_cls, Command):
                raise TypeError(
                    f"Cannot register {command_cls.__name__} as it is not a subclass of Command."
                )

            # Register primary name
            cls._command_registry[name.lower()] = command_cls
            logger.debug(f"Registered command '{name.lower()}' with class {command_cls.__name__}.")

            # Register aliases
            if aliases:
                for alias in aliases:
                    cls._command_registry[alias.lower()] = command_cls
                    logger.debug(
                        f"Registered alias '{alias.lower()}' for command '{name.lower()}'."
                    )

            return command_cls

        return decorator

    @classmethod
    def get_command(cls, name: str, **kwargs) -> Command:
        """Retrieve a command instance by name.

        Args:
            name (str): The name or alias of the command.
            **kwargs: Additional parameters for command initialization.

        Returns:
            Command: An instance of the requested command.
        """
        command_cls = cls._command_registry.get(name.lower())
        if not command_cls:
            available_commands = ", ".join(set(cls._command_registry.keys()))
            raise ValueError(
                f"Unsupported command type: '{name}'. Available commands: {available_commands}."
            )
        logger.debug(f"Instantiating command '{name.lower()}'.")
        return command_cls(**kwargs)