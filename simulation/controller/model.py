import base64
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    """Supported encoding types for requests/responses"""

    JSON = "json"
    BASE64 = "base64"
    PLAIN = "plain"

    @staticmethod
    def from_str(encoding: str) -> "EncodingType":
        """Convert string to EncodingType"""
        encoding = encoding.lower()
        if encoding in EncodingType.__members__:
            return EncodingType[encoding.upper()]
        return EncodingType.JSON


@dataclass
class Request:
    command_name: str
    request_id: str
    parameters: Dict[str, Any]
    encoding: str = EncodingType.JSON.value

    def to_json(self) -> str:
        """Serialize request to JSON string"""
        data = self.__dict__
        if self.encoding == EncodingType.BASE64.value:
            # Encode parameters as base64 if needed
            data["parameters"] = base64.b64encode(json.dumps(self.parameters).encode()).decode()
        return json.dumps(data)

    @staticmethod
    def from_json(data: str) -> "Request":
        """Create Request from JSON string"""
        dict_data = json.loads(data)
        encoding = dict_data.get("encoding", EncodingType.JSON.value)

        if encoding == EncodingType.BASE64.value:
            # Decode base64 parameters
            params_b64 = dict_data.get("parameters", "")
            try:
                params_json = base64.b64decode(params_b64).decode()
                dict_data["parameters"] = json.loads(params_json)
            except Exception as e:
                logger.error(f"Failed to decode base64 parameters: {e}")
                dict_data["parameters"] = {}

        return Request(**dict_data)


@dataclass
class Response:
    request_id: str
    status: str
    message: str
    encoding: str = EncodingType.JSON.value
    data: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Serialize response to JSON string"""
        data = self.__dict__
        if self.encoding == EncodingType.BASE64.value and self.data:
            # Encode data payload as base64 if present
            data["data"] = base64.b64encode(json.dumps(self.data).encode()).decode()
        return json.dumps(data)

    @staticmethod
    def from_json(data: str) -> "Response":
        """Create Response from JSON string"""
        dict_data = json.loads(data)
        encoding = dict_data.get("encoding", EncodingType.JSON.value)

        if encoding == EncodingType.BASE64.value:
            # Decode base64 data payload if present
            data_b64 = dict_data.get("data")
            if data_b64:
                try:
                    data_json = base64.b64decode(data_b64).decode()
                    dict_data["data"] = json.loads(data_json)
                except Exception as e:
                    logger.error(f"Failed to decode base64 data: {e}")
                    dict_data["data"] = None

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
    def from_str(status: str) -> "Status":
        status = status.upper()
        if status in Status.__members__:
            return Status[status]
        return Status.UNKNOWN

    @staticmethod
    def to_str(status: "Status") -> str:
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
    },
    STATUS = {
        "name": "status",
        "aliases": ["state"],
        "description": "Returns the current simulation status",
    },
    SEND = {
        "name": "send",
        "aliases": ["message"],
        "description": "Send a message to the simulation",
    }
    UPDATE_PARAMS = {
        "name": "update_params",
        "aliases": ["update"],
        "description": "Update simulation parameters",
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
    def get_by_name(cls, name: str) -> "CommandType":
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
