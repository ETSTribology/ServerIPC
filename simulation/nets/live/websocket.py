import asyncio
import logging
import queue
import threading
from typing import Any, Dict, Optional

import websockets
from simulation.nets.nets import Nets
from simulation.nets.serialization.factory import SerializerFactory

logger = logging.getLogger(__name__)


class WebSocket(Nets):
    def __init__(self, uri: str, serializer_method: str = "pickle"):
        self.uri = uri
        self.serializer_method = serializer_method
        self.serializer_factory = SerializerFactory()
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.websocket = None
        self.listener_thread = threading.Thread(target=self.run_listener, daemon=True)
        self.listener_thread.start()
        logger.info("WebSocket initialized and listener thread started.")

    async def connect_async(self):
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"Connected to WebSocket server at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server at {self.uri}: {e}")
            raise

    async def listen_async(self):
        try:
            async for message in self.websocket:
                command = message.strip().lower()
                logger.info(f"Received command from WebSocket: {command}")
                self.command_queue.put(command)
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed.")
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")

    def run_listener(self):
        asyncio.run(self.start())

    async def start(self):
        await self.connect_async()
        await self.listen_async()

    def get_command(self) -> Optional[str]:
        try:
            command = self.command_queue.get_nowait()
            logger.debug(f"Command retrieved from WebSocket queue: {command}")
            return command
        except queue.Empty:
            return None

    def set_data(self, key: str, data: str) -> None:
        raise NotImplementedError(
            "set_data is not implemented for WebSocketCommunicationClient."
        )

    def get_data(self, key: str) -> Optional[Any]:
        raise NotImplementedError(
            "get_data is not implemented for WebSocketCommunicationClient."
        )

    def publish_data(self, channel: str, data: str) -> None:
        # WebSocket doesn't inherently support channels; handle as per your protocol
        try:
            logger.info(f"Sending data over WebSocket: {data}")
            asyncio.run(self.websocket.send(data))
            logger.debug("Data sent successfully over WebSocket.")
        except Exception as e:
            logger.error(f"Failed to send data over WebSocket: {e}")

    def serialize_data(
        self, data: Dict[str, Any], method: str = "pickle"
    ) -> Optional[str]:
        try:
            logger.info("Serializing data using SerializerFactory.")
            serializer = self.serializer_factory.get_serializer(method)
            serialized = serializer.serialize(data)
            if serialized:
                logger.debug("Data serialized successfully.")
                return serialized
            logger.error("Serialization returned None.")
            return None
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            return None

    def deserialize_data(self, data_str: str, method: str = "pickle") -> Dict[str, Any]:
        try:
            logger.info("Deserializing data using SerializerFactory.")
            serializer = self.serializer_factory.get_serializer(method)
            data = serializer.deserialize(data_str)
            logger.debug("Data deserialized successfully.")
            return data
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise

    def close(self) -> None:
        """Gracefully close the WebSocket connection and stop the listener thread."""
        logger.info("Closing WebSocketCommunicationClient.")
        self.stop_event.set()
        if self.websocket:
            asyncio.run(self.websocket.close())
            logger.info("WebSocket connection closed.")
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
            logger.info("Listener thread terminated.")
