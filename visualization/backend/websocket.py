import websocket
from typing import Dict, Any
from visualization.backend.backend import Backend

class WebSocketBackend(Backend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config(["host", "port", "path"])
        self.ws = None

    def connect(self):
        try:
            url = f"ws://{self.config['host']}:{self.config['port']}{self.config['path']}"
            if self.config.get("secure", False):
                url = url.replace("ws://", "wss://")
            self.ws = websocket.create_connection(url)
            self.connected = True
            logger.info("Connected to WebSocket backend.")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

    def disconnect(self):
        if self.ws:
            self.ws.close()
            self.connected = False
            logger.info("Disconnected from WebSocket backend.")

    def write_data(self, key: str, value: Any) -> None:
        self.ws.send(f"{key}:{value}")

    def read_data(self, key: str) -> Any:
        return self.ws.recv()

    def delete_data(self, key: str) -> None:
        logger.warning("Delete operation is not supported for WebSocket backend.")
