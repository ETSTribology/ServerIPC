from typing import Type
from simulation.nets.live.grpc import GRPC
from simulation.nets.nets import Nets
from simulation.nets.live.redis import Redis
from simulation.nets.live.websocket import WebSocket
from simulation.core.utils.singleton import SingletonMeta


class NetsFactory(metaclass=SingletonMeta):
    @staticmethod
    def create_client(method: str, **kwargs) -> Nets:
        method = method.lower()
        if method == "redis":
            return Redis(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 6379),
                db=kwargs.get("db", 0),
                password=kwargs.get("password"),
            )
        if method == "websocket":
            return WebSocket(
                uri=kwargs.get("uri"),
                serializer_method=kwargs.get("serializer_method", "pickle"),
            )
        if method == "grpc":
            return GRPC(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 50051),
                serializer_method=kwargs.get("serializer_method", "pickle"),
            )
        raise ValueError(f"Unsupported communication method: {method}")
