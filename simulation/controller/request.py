
from dataclasses import dataclass

@dataclass
class RequestMessage:
    request_id: str
    command_name: str
    parameters: dict