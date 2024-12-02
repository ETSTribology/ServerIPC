import grpc
from typing import Dict, Any

from visualization.backend.backend import Backend


class GrpcBackend(Backend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.channel = None
        self.stub = None

    def connect(self):
        try:
            credentials = None
            if self.config["credentials"].get("certificate"):
                with open(self.config["credentials"]["certificate"], "rb") as cert, \
                        open(self.config["credentials"]["privateKey"], "rb") as key:
                    credentials = grpc.ssl_channel_credentials(cert.read(), key.read())
            target = f"{self.config['host']}:{self.config['port']}"
            self.channel = grpc.secure_channel(target, credentials) if credentials else grpc.insecure_channel(target)
            self.stub = getattr(__import__(self.config["service"]), self.config["service"]).Stub(self.channel)
            self.connected = True
            logger.info("Connected to gRPC backend.")
        except Exception as e:
            logger.error(f"Failed to connect to gRPC: {e}")
            raise

    def disconnect(self):
        if self.channel:
            self.channel.close()
            self.connected = False
            logger.info("Disconnected from gRPC backend.")

    def write_data(self, key: str, value: Any) -> None:
        logger.warning("Write operation is not directly supported for gRPC backend.")

    def read_data(self, key: str) -> Any:
        logger.warning("Read operation is not directly supported for gRPC backend.")

    def delete_data(self, key: str) -> None:
        logger.warning("Delete operation is not directly supported for gRPC backend.")
