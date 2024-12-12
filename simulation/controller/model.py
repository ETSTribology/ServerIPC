# model.py
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
import json
import logging

from simulation.controller.history import CommandHistory, CommandHistoryEntry

logger = logging.getLogger(__name__)

@dataclass
class Request:
    command_name: str
    request_id: str
    parameters: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(data: str) -> 'Request':
        dict_data = json.loads(data)
        return Request(**dict_data)

@dataclass
class Response:
    request_id: str
    status: str
    message: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(data: str) -> 'Response':
        dict_data = json.loads(data)
        return Response(**dict_data)

class Status(Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    COMMAND_FAILED = "COMMAND_FAILED"
    CANNOT_CONNECT = "CANNOT_CONNECT"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"

    @staticmethod
    def from_str(status: str) -> 'Status':
        status = status.upper()
        if status in Status.__members__:
            return Status[status]
        return Status.UNKNOWN

    @staticmethod
    def to_str(status: 'Status') -> str:
        return status.value

class CommandType(Enum):
    """Enum representing all possible simulation commands and their aliases."""
    START = {
        "name": "start",
        "aliases": ["run", "begin"],
        "description": "Starts the simulation if not running",
    }
    PAUSE = {
        "name": "pause",
        "aliases": ["suspend"],
        "description": "Pauses a running simulation",
    }
    STOP = {
        "name": "stop",
        "aliases": ["halt"],
        "description": "Stops the simulation",
    }
    RESUME = {
        "name": "resume",
        "aliases": ["continue"],
        "description": "Resumes a paused simulation",
    }
    PLAY = {
        "name": "play",
        "aliases": ["continue"],
        "description": "Plays the simulation",
    }
    KILL = {
        "name": "kill",
        "aliases": ["exit", "terminate"],
        "description": "Terminates the simulation",
    }
    RESET = {
        "name": "reset",
        "aliases": ["reinitialize"],
        "description": "Resets the simulation state",
    }

    @property
    def command_name(self) -> str:
        return self.value["name"]

    @property
    def command_aliases(self) -> list:
        return self.value["aliases"]

    @property
    def description(self) -> str:
        return self.value["description"]

    @classmethod
    def get_by_name(cls, name: str) -> 'CommandType':
        """Get command type by name or alias."""
        name = name.lower()
        for command in cls:
            if name == command.command_name or name in command.command_aliases:
                return command
        raise ValueError(f"Unknown command: {name}")

class CommandError(Exception):
    """Base class for command-related errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class CommandFailedError(CommandError):
    """Exception raised when a command fails to execute."""
    pass

class CannotConnectError(CommandError):
    """Exception raised when a connection error occurs."""
    pass

class InvalidParametersError(CommandError):
    """Exception raised when invalid parameters are provided."""
    pass