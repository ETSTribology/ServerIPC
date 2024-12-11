import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class Status(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ResponseMessage:
    request_id: str
    status: Status
    message: str