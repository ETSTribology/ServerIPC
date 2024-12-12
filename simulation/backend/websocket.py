import json
import logging
from typing import Any, Dict, Optional

import websocket

from simulation.backend.backend import Backend
from simulation.controller.model import Request, Response
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


class WebSocketBackend(Backend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config(["host", "port", "path"])
        self.ws = None

    def validate_config(self, required_keys):
        for key in required_keys:
            if key not in self.config:
                raise SimulationError(
                    SimulationErrorCode.CONFIGURATION, f"Missing required config key: {key}"
                )

    def connect(self):
        try:
            url = f"ws://{self.config['host']}:{self.config['port']}{self.config['path']}"
            if self.config.get("secure", False):
                url = url.replace("ws://", "wss://")
            self.ws = websocket.create_connection(url)
            self.connected = True
            logger.info(
                SimulationLogMessageCode.WEBSOCKET_CONNECTED.details(
                    "Connected to WebSocket backend."
                )
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.WEBSOCKET_CONNECTION_FAILED.details(
                    f"Failed to connect to WebSocket: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to connect to WebSocket",
                details=str(e),
            )

    def disconnect(self):
        try:
            if self.ws:
                self.ws.close()
                self.connected = False
                logger.info(
                    SimulationLogMessageCode.WEBSOCKET_DISCONNECTED.details(
                        "Disconnected from WebSocket backend."
                    )
                )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.WEBSOCKET_DISCONNECTION_FAILED.details(
                    f"Error while disconnecting WebSocket backend: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Error while disconnecting WebSocket",
                details=str(e),
            )

    def write(self, key: str, value: Any) -> None:
        try:
            message = {"key": key, "value": value}
            self.ws.send(json.dumps(message))
            logger.info(
                SimulationLogMessageCode.WEBSOCKET_WRITE_SUCCESS.details(
                    f"Written data to key '{key}'."
                )
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.WEBSOCKET_WRITE_FAILURE.details(
                    f"Failed to write data to key '{key}': {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.FILE_IO, f"Failed to write data to key '{key}'", details=str(e)
            )

    def read(self, key: str) -> Any:
        try:
            message = self.ws.recv()
            data = json.loads(message)
            if data.get("key") == key:
                logger.info(
                    SimulationLogMessageCode.WEBSOCKET_READ_SUCCESS.details(
                        f"Read data from key '{key}'."
                    )
                )
                return data.get("value")
            else:
                logger.warning(
                    SimulationLogMessageCode.WEBSOCKET_READ_FAILURE.details(
                        f"Key mismatch for '{key}'."
                    )
                )
                return None
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.WEBSOCKET_READ_FAILURE.details(
                    f"Failed to read data from key '{key}': {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                f"Failed to read data from key '{key}'",
                details=str(e),
            )

    def send_response(self, response: Response) -> None:
        try:
            message = response.to_json()
            self.ws.send(message)
            logger.info(
                SimulationLogMessageCode.WEBSOCKET_WRITE_SUCCESS.details(
                    "Published response to WebSocket."
                )
            )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.WEBSOCKET_WRITE_FAILURE.details(
                    f"Failed to publish response: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to publish response",
                details=str(e),
            )

    def get_command(self) -> Optional[Request]:
        try:
            message = self.ws.recv()
            if message:
                command = Request.from_json(message)
                logger.info(
                    SimulationLogMessageCode.WEBSOCKET_READ_SUCCESS.details(
                        "Received command from WebSocket."
                    )
                )
                return command
            return None
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.WEBSOCKET_READ_FAILURE.details(
                    f"Failed to retrieve command: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.NETWORK_COMMUNICATION,
                "Failed to retrieve command",
                details=str(e),
            )
