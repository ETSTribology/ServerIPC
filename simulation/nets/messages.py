import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


def generate_request_id() -> str:
    """Generates a unique request ID using UUID.

    Returns
    -------
    str
        A unique string identifier for the request.

    """
    return str(uuid.uuid4())


class Status(Enum):
    """Represents the status of a response.

    Attributes
    ----------
    SUCCESS : str
        Indicates the operation was successful.
    ERROR : str
        Indicates an error occurred during the operation.
    WARNING : str
        Indicates the operation completed with warnings.

    """

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class RequestMessage:
    """Represents a request message sent to the simulation server.

    Attributes:
    ----------
    request_id : str
        The unique identifier for the request. Typically generated using `generate_request_id`.
    command : str
        The command to be executed (e.g., 'start', 'pause', 'stop').
    payload : Optional[Dict[str, Any]]
        Additional data required to execute the command (e.g., simulation parameters).

    Example:
    -------
    Create a new request message:
    >>> request = RequestMessage(
            request_id="1234",
            command="start",
            payload={"param1": 10, "param2": "value"}
        )
    >>> print(request)
    RequestMessage(request_id='1234', command='start', payload={'param1': 10, 'param2': 'value'})

    Notes:
    -----
    - The `request_id` ensures each request is uniquely identifiable for tracking purposes.
    - The `command` field is mandatory and determines the action to be performed.

    """

    request_id: str
    command: str
    payload: Optional[Dict[str, Any]] = None

    def ensure_payload_id(self) -> None:
        """Ensures that the payload contains an 'id' field. If missing, it generates a unique ID.

        Returns
        -------
        None

        """
        if self.payload is None:
            self.payload = {}
        if "id" not in self.payload:
            self.payload["id"] = generate_request_id()
            logging.info(f"Generated missing 'id' for payload: {self.payload['id']}")


@dataclass
class ResponseMessage:
    """Represents a response message from the simulation server.

    Attributes:
    ----------
    request_id : str
        The unique identifier of the request. Matches the `request_id` from the corresponding `RequestMessage`.
    status : str
        The status of the response. Must be one of `Status.SUCCESS`, `Status.ERROR`, or `Status.WARNING`.
    message : Optional[str]
        A detailed message describing the response, error, or warning.
    result : Optional[Dict[str, Any]]
        The result or output of the response, typically in a dictionary format.

    Example:
    -------
    Create a success response:
    >>> response = ResponseMessage(
            request_id="1234",
            status=Status.SUCCESS.value,
            message="Simulation started successfully.",
            result={"simulation_id": "sim123"}
        )
    >>> print(response)
    ResponseMessage(request_id='1234', status='success', message='Simulation started successfully.', result={'simulation_id': 'sim123'})

    Notes:
    -----
    - The `request_id` links the response to a specific request for traceability.
    - The `status` field reflects the outcome of the request processing.
    - The `result` field is optional and typically contains the outcome of the operation, if applicable.

    """

    request_id: str
    status: str
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def ensure_request_id(self, request_id: str) -> None:
        """Ensures that the response contains a valid request ID.

        Parameters
        ----------
        request_id : str
            The request ID to set in the response.

        Returns
        -------
        None

        """
        if self.request_id is None:
            self.request_id = request_id
            logging.info(f"Set missing 'request_id' for response: {request_id}")

    def ensure_result_id(self) -> None:
        """Ensures that the result contains an 'id' field. If missing, it generates a unique ID.

        Returns
        -------
        None

        """
        if self.result is None:
            self.result = {}
        if "id" not in self.result:
            self.result["id"] = generate_request_id()
            logging.info(f"Generated missing 'id' for result: {self.result['id']}")
