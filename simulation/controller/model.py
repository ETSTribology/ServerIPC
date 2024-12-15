import base64
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .encoding import EncodingType, decode_parameters, encode_parameters


@dataclass
class Request:
    command_name: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parameters: Dict[str, Any] = field(default_factory=dict)
    encoding: str = EncodingType.JSON.value

    def to_json(self) -> str:
        """Serialize request to JSON string"""
        data = encode_parameters(self.__dict__, EncodingType.from_str(self.encoding))
        return json.dumps(data)

    @staticmethod
    def from_json(data: str) -> "Request":
        """Create Request from JSON string"""
        dict_data = json.loads(data)
        dict_data = decode_parameters(dict_data)
        dict_data["encoding"] = dict_data.get("encoding", EncodingType.JSON.value)
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
        if self.encoding == EncodingType.BASE64.value and self.data:
            self.data = base64.b64encode(json.dumps(self.data).encode()).decode()
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(data: str) -> "Response":
        """Create Response from JSON string"""
        dict_data = json.loads(data)
        encoding = dict_data.get("encoding", EncodingType.JSON.value)
        if encoding == EncodingType.BASE64.value and dict_data.get("data"):
            dict_data["data"] = base64.b64decode(dict_data["data"]).decode()
            dict_data["data"] = json.loads(dict_data["data"])
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
    RESET = (
        {
            "name": "reset",
            "aliases": ["reinitialize"],
            "description": "Resets the simulation state",
        },
    )
    STATUS = (
        {
            "name": "status",
            "aliases": ["state"],
            "description": "Returns the current simulation status",
        },
    )
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
