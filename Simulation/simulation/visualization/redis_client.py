import redis
import bson
import zlib
import base64
import logging
from typing import Dict, Any
import queue
import numpy as np

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        try:
            self.redis_client = redis.StrictRedis(host=self.host, port=self.port, db=self.db)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}, db={db}")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        self.pubsub = self.redis_client.pubsub()
        self.subscribed_channels = []
        logger.info(f"Initialized RedisClient with host={host}, port={port}, db={db}")

    def subscribe(self, channel: str):
        if channel not in self.subscribed_channels:
            try:
                self.pubsub.subscribe(channel)
                self.subscribed_channels.append(channel)
                logger.info(f"Subscribed to Redis channel: {channel}")
            except redis.RedisError as e:
                logger.error(f"Failed to subscribe to channel '{channel}': {e}")
                raise

    def unsubscribe(self, channel: str):
        if channel in self.subscribed_channels:
            try:
                self.pubsub.unsubscribe(channel)
                self.subscribed_channels.remove(channel)
                logger.info(f"Unsubscribed from Redis channel: {channel}")
            except redis.RedisError as e:
                logger.error(f"Failed to unsubscribe from channel '{channel}': {e}")
                raise

    def listen(self, channels: list, mesh_queue: queue.Queue):
        for channel in channels:
            self.subscribe(channel)
        logger.info(f"Listening to channels: {channels}")
        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        mesh_data_b64 = message['data']
                        mesh_data = self.deserialize_mesh_data(mesh_data_b64)
                        mesh_queue.put(mesh_data)
                        logger.debug(f"Received and queued mesh data from channel '{message['channel']}': Step {mesh_data.get('step', 'N/A')}")
                    except Exception as e:
                        logger.error(f"Error processing message from channel '{message['channel']}': {e}")
        except Exception as e:
            logger.error(f"Error while listening to Redis channels: {e}")
            raise

    @staticmethod
    def serialize_mesh_data(mesh_data: Dict[str, Any]) -> str:
        try:
            mesh_data_bson = bson.dumps(mesh_data)
            mesh_data_compressed = zlib.compress(mesh_data_bson)
            mesh_data_b64 = base64.b64encode(mesh_data_compressed).decode('utf-8')
            logger.debug("Mesh data serialized, compressed, and encoded.")
            return mesh_data_b64
        except Exception as e:
            logger.error(f"Error during mesh data serialization: {e}")
            raise

    @staticmethod
    def deserialize_mesh_data(data_b64: str) -> Dict[str, Any]:
        try:
            data_compressed = base64.b64decode(data_b64)
            data_bson = zlib.decompress(data_compressed)
            mesh_data = bson.loads(data_bson)
            logger.debug("Mesh data decoded, decompressed, and deserialized.")
            return mesh_data
        except Exception as e:
            logger.error(f"Error during mesh data deserialization: {e}")
            raise

    def send_command(self, command: str):
        try:
            self.redis_client.publish('simulation_commands', command)
            logger.info(f"Sent command to simulation server: {command}")
        except redis.RedisError as e:
            logger.error(f"Error sending command '{command}': {e}")
            raise

    def publish_data(self, channel: str, data: str):
        try:
            self.redis_client.publish(channel, data)
            logger.info(f"Published data to channel '{channel}'.")
        except redis.RedisError as e:
            logger.error(f"Error publishing data to channel '{channel}': {e}")
            raise

    def set_data(self, key: str, data: str):
        try:
            self.redis_client.set(key, data)
            logger.info(f"Set data for key '{key}' in Redis.")
        except redis.RedisError as e:
            logger.error(f"Error setting data for key '{key}': {e}")
            raise

    def get_data(self, key: str) -> Any:
        try:
            data = self.redis_client.get(key)
            logger.info(f"Retrieved data for key '{key}' from Redis.")
            return data
        except redis.RedisError as e:
            logger.error(f"Error getting data for key '{key}': {e}")
            raise
