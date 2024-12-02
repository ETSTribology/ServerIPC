"""WebSocket communication backend."""

import json
import asyncio
import websockets
from typing import Any, Dict, Callable, Optional

from .base import BaseCommunicationBackend
from .registry import CommunicationBackendRegistry


class WebSocketCommunicationBackend(BaseCommunicationBackend):
    """WebSocket-based communication backend."""

    def __init__(self):
        """Initialize WebSocket backend."""
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._uri: Optional[str] = None
        self._subscriptions: Dict[str, Callable[[Any], None]] = {}
        self._receive_task: Optional[asyncio.Task] = None

    async def _receive_messages(self) -> None:
        """Background task to receive and process messages."""
        try:
            while self._websocket and not self._websocket.closed:
                try:
                    message = await self._websocket.recv()
                    data = json.loads(message)
                    topic = data.get('topic')
                    
                    if topic in self._subscriptions:
                        self._subscriptions[topic](data.get('payload'))
                except websockets.ConnectionClosed:
                    break
                except json.JSONDecodeError:
                    print("Invalid message format")
        except Exception as e:
            print(f"WebSocket receive error: {e}")

    def connect(self, config: Dict[str, Any]) -> None:
        """
        Establish WebSocket connection.
        
        Args:
            config: WebSocket connection configuration
        """
        try:
            self._uri = f"ws://{config.get('host', 'localhost')}:{config.get('port', 8765)}"
            
            # Use asyncio to establish connection
            async def _connect():
                self._websocket = await websockets.connect(self._uri)
                self._receive_task = asyncio.create_task(self._receive_messages())
            
            asyncio.run(_connect())
        except Exception as e:
            raise ConnectionError(f"WebSocket connection failed: {e}")

    def disconnect(self) -> None:
        """Close WebSocket connection."""
        async def _close():
            if self._websocket:
                await self._websocket.close()
            if self._receive_task:
                self._receive_task.cancel()
        
        asyncio.run(_close())

    def send_message(self, topic: str, message: Any) -> None:
        """
        Send a message via WebSocket.
        
        Args:
            topic: Message destination/channel
            message: Payload to send
        """
        if not self._websocket:
            raise RuntimeError("WebSocket not connected")
        
        async def _send():
            payload = json.dumps({
                'topic': topic,
                'payload': message
            })
            await self._websocket.send(payload)
        
        asyncio.run(_send())

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to a WebSocket topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Function to call when message is received
        """
        self._subscriptions[topic] = callback

    def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from a WebSocket topic.
        
        Args:
            topic: Topic to unsubscribe from
        """
        if topic in self._subscriptions:
            del self._subscriptions[topic]


# Register the WebSocket backend with the registry
CommunicationBackendRegistry.register('websocket', WebSocketCommunicationBackend)
