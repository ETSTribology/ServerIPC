import asyncio
import logging
import threading
from queue import Queue
from typing import Any, Dict, Optional

from surrealdb import SurrealDB as SurrealClient
from surrealdb import SurrealDbError


class SurrealStreaming(SurrealStreamingNets):
    """Implementation of SurrealDB as a streaming communication backend."""

    def __init__(
        self,
        url: str = "ws://localhost:8000/rpc",
        namespace: str = "test",
        database: str = "test",
        username: str = "root",
        password: str = "rootpassword",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.url = url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self.db = None
        self.command_queue = Queue()
        self.listener_thread = None
        self.stop_event = threading.Event()

        self.connect()
        self.start_listener()

    def connect(self) -> None:
        """Establishes the connection to SurrealDB."""
        try:
            self.db = SurrealClient(self.url)
            self.db.signin({"user": self.username, "pass": self.password})
            self.db.use(self.namespace, self.database)
            self.logger.info("Connected to SurrealDB.")
        except SurrealDbError as e:
            self.logger.error(f"Failed to connect to SurrealDB: {e}")
            raise

    def start_listener(self) -> None:
        """Starts a listener thread for incoming SurrealDB messages."""
        self.listener_thread = threading.Thread(
            target=self.listen_commands, daemon=True
        )
        self.listener_thread.start()
        self.logger.info("SurrealDB listener thread started.")

    def listen_commands(self) -> None:
        """Listens for incoming messages from the SurrealDB streaming channel."""
        self.logger.info("Listening for commands on SurrealDB...")
        try:
            asyncio.run(self._listen_async())
        except Exception as e:
            self.logger.error(f"Error in listener thread: {e}")

    async def _listen_async(self) -> None:
        """Asynchronous method to handle streaming from SurrealDB."""
        try:
            async with self.db.subscribe("simulation_commands") as stream:
                async for message in stream:
                    if self.stop_event.is_set():
                        break
                    self.logger.debug(f"Received message: {message}")
                    self.command_queue.put(message)
        except Exception as e:
            self.logger.error(f"Error in SurrealDB streaming: {e}")

    def get_command(self) -> Optional[Dict[str, Any]]:
        """Retrieves the latest command from the queue."""
        try:
            return self.command_queue.get_nowait()
        except Queue.Empty:
            return None

    def set_data(self, key: str, data: str) -> None:
        """Stores data in SurrealDB."""
        try:
            self.db.query(f"UPDATE {key} CONTENT {data}")
            self.logger.info(f"Data stored with key: {key}")
        except SurrealDbError as e:
            self.logger.error(f"Failed to set data in SurrealDB: {e}")

    def get_data(self, key: str) -> Optional[Any]:
        """Retrieves data from SurrealDB."""
        try:
            result = self.db.query(f"SELECT * FROM {key}")
            self.logger.info(f"Data retrieved for key: {key}")
            return result
        except SurrealDbError as e:
            self.logger.error(f"Failed to get data from SurrealDB: {e}")
            return None

    def publish_data(self, channel: str, data: str) -> None:
        """Publishes data to a SurrealDB channel."""
        try:
            self.db.query(f"LET publish := {channel} CONTENT {data}")
            self.logger.info(f"Data published to channel: {channel}")
        except SurrealDbError as e:
            self.logger.error(f"Failed to publish data to channel '{channel}': {e}")

    def close(self) -> None:
        """Closes the SurrealDB connection."""
        self.stop_event.set()
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
        if self.db:
            self.db.close()
            self.logger.info("SurrealDB connection closed.")
