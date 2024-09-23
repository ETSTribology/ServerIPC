import redis
import bson
import zlib
import base64
import logging
from typing import Dict, Any
import threading

logger = logging.getLogger(__name__)

class SimulationRedisClient:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: str = None):
        try:
            self.redis_client = redis.StrictRedis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}, db={db}")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe('simulation_commands')  # Subscribe to the simulation_commands channel
        self.command = None
        self.lock = threading.Lock()
        self.listener_thread = threading.Thread(target=self.listen_commands, daemon=True)
        self.listener_thread.start()
        logger.info("SimulationRedisClient initialized and listener thread started.")

    def listen_commands(self):
        logger.info("Listener thread started, waiting for simulation commands.")
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                command = message['data'].strip().lower()
                logger.info(f"Received command from Redis: {command}")
                with self.lock:
                    self.command = command
                # Optionally, handle the command immediately or leave it to the main loop

    def get_command(self):
        with self.lock:
            cmd = self.command
            self.command = None
        return cmd

    def set_data(self, key: str, data: str):
        try:
            logger.info(f"Storing data in Redis with key: {key}")
            self.redis_client.set(key, data)
            logger.debug(f"Data stored successfully for key: {key}")
        except redis.RedisError as e:
            logger.error(f"Failed to set data in Redis for key {key}: {e}")

    def get_data(self, key: str) -> Any:
        try:
            logger.info(f"Retrieving data from Redis with key: {key}")
            data = self.redis_client.get(key)
            logger.debug(f"Data retrieved for key {key}: {data}")
            return data
        except redis.RedisError as e:
            logger.error(f"Failed to get data from Redis for key {key}: {e}")
            return None

    def publish_data(self, channel: str, data: str):
        try:
            logger.info(f"Publishing data to channel: {channel}")
            self.redis_client.publish(channel, data)
            logger.debug(f"Data published successfully to channel: {channel}")
        except redis.RedisError as e:
            logger.error(f"Failed to publish data to channel {channel}: {e}")

    def serialize_mesh_data(self, mesh_data: Dict[str, Any]) -> str:
        try:
            logger.info("Serializing mesh data.")
            mesh_data_bson = bson.dumps(mesh_data)
            mesh_data_compressed = zlib.compress(mesh_data_bson)
            mesh_data_b64 = base64.b64encode(mesh_data_compressed).decode('utf-8')
            logger.debug("Mesh data serialized successfully.")
            return mesh_data_b64
        except Exception as e:
            logger.error(f"Failed to serialize mesh data: {e}")
            return None

    def deserialize_mesh_data(self, data_b64: str) -> Dict[str, Any]:
        try:
            data_compressed = base64.b64decode(data_b64)
            data_bson = zlib.decompress(data_compressed)
            mesh_data = bson.loads(data_bson)
            logger.debug("Mesh data decoded, decompressed, and deserialized.")
            return mesh_data
        except Exception as e:
            logger.error(f"Failed to deserialize mesh data: {e}")
            raise

    # Command methods
    def start(self):
        with self.lock:
            self.publish_data('simulation_commands', 'start')

    def stop(self):
        with self.lock:
            self.publish_data('simulation_commands', 'stop')

    def pause(self):
        with self.lock:
            self.publish_data('simulation_commands', 'pause')

    def resume(self):
        with self.lock:
            self.publish_data('simulation_commands', 'resume')

    def play(self):
        with self.lock:
            self.publish_data('simulation_commands', 'play')

    def kill(self):
        with self.lock:
            self.publish_data('simulation_commands', 'kill')
