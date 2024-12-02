import asyncio
import logging
import queue
import threading
from typing import Any, Dict, Optional

import grpc

import simulation.nets.live.proto.simulation_pb2 as simulation_pb2
import simulation.nets.live.proto.simulation_pb2_grpc as simulation_grpc
from simulation.nets.nets import Nets
from simulation.nets.serialization.factory import SerializerFactory

logger = logging.getLogger(__name__)


class GRPC(Nets):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        serializer_method: str = "pickle",
    ):
        self.host = host
        self.port = port
        self.serializer_method = serializer_method
        self.serializer_factory = SerializerFactory()
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.channel = None
        self.stub = None
        self.listener_thread = threading.Thread(target=self.run_listener, daemon=True)
        self.listener_thread.start()
        logger.info("GRPC initialized and listener thread started.")

    async def connect_async(self):
        try:
            self.channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}")
            await self.channel.channel_ready()
            self.stub = simulation_grpc.SimulationServiceStub(self.channel)
            logger.info(f"Connected to gRPC server at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to gRPC server at {self.host}:{self.port}: {e}")
            raise

    async def listen_async(self):
        try:
            async for command_msg in self.stub.SubscribeCommands(simulation_pb2.Empty()):
                command = command_msg.command.strip().lower()
                logger.info(f"Received command from gRPC: {command}")
                self.command_queue.put(command)
        except grpc.RpcError as e:
            logger.error(f"gRPC listener encountered an error: {e}")

    def run_listener(self):
        asyncio.run(self.start())

    async def start(self):
        await self.connect_async()
        await self.listen_async()

    def get_command(self) -> Optional[str]:
        try:
            command = self.command_queue.get_nowait()
            logger.debug(f"Command retrieved from gRPC queue: {command}")
            return command
        except queue.Empty:
            return None

    def set_data(self, key: str, data: str) -> None:
        # Implement as needed based on your gRPC service definitions
        raise NotImplementedError("set_data is not implemented for GRPCCommunicationClient.")

    def get_data(self, key: str) -> Optional[Any]:
        # Implement as needed based on your gRPC service definitions
        raise NotImplementedError("get_data is not implemented for GRPCCommunicationClient.")

    def publish_data(self, channel: str, data: str) -> None:
        try:
            logger.info(f"Sending data to gRPC server: {data}")
            asyncio.run(
                self.stub.PublishData(simulation_pb2.DataRequest(channel=channel, data=data))
            )
            logger.debug("Data sent successfully to gRPC server.")
        except Exception as e:
            logger.error(f"Failed to send data to gRPC server: {e}")

    def serialize_data(self, data: Dict[str, Any], method: str = "pickle") -> Optional[str]:
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

    async def close_async(self):
        logger.info("Closing GRPCCommunicationClient.")
        if self.channel:
            await self.channel.close()
            logger.info("gRPC channel closed.")

    def close(self) -> None:
        """Gracefully stop the listener thread and close gRPC connections."""
        logger.info("Closing GRPCCommunicationClient.")
        self.stop_event.set()
        if self.channel:
            asyncio.run(self.close_async())
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
            logger.info("Listener thread terminated.")
