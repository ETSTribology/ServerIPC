import redis
import bson
import zlib
import base64
import logging
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
from typing import Dict, Any, Generator
from threading import Thread
import queue

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = redis.StrictRedis(host=self.host, port=self.port, db=self.db)
        self.pubsub = self.redis_client.pubsub()
        self.subscribed_channels = []
        logger.info(f"Initialized RedisClient with host={host}, port={port}, db={db}")

    def subscribe(self, channel: str):
        if channel not in self.subscribed_channels:
            self.pubsub.subscribe(channel)
            self.subscribed_channels.append(channel)
            logger.info(f"Subscribed to Redis channel: {channel}")

    def unsubscribe(self, channel: str):
        if channel in self.subscribed_channels:
            self.pubsub.unsubscribe(channel)
            self.subscribed_channels.remove(channel)
            logger.info(f"Unsubscribed from Redis channel: {channel}")

    def listen(self, channels: list, mesh_queue: queue.Queue):
        for channel in channels:
            self.subscribe(channel)
        logger.info(f"Listening to channels: {channels}")
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    mesh_data_b64 = message['data'].decode('utf-8')
                    mesh_data = self.deserialize_mesh_data(mesh_data_b64)
                    mesh_queue.put(mesh_data)
                except Exception as e:
                    logger.error(f"Error processing message from channel '{message['channel'].decode('utf-8')}': {e}")

    @staticmethod
    def serialize_mesh_data(mesh_data: Dict) -> str:
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
    def deserialize_mesh_data(data_b64: str) -> Dict:
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
        except Exception as e:
            logger.error(f"Error sending command '{command}': {e}")
            raise

class PolyscopeVisualizer:
    def __init__(self, mesh_queue: queue.Queue, redis_client: RedisClient):

        ps.set_up_dir("z_up")
        ps.init()
        self.mesh_states = []
        self.step_names = []
        self.current_step_index = 0
        self.is_paused = False
        self.visual_mesh = None
        self.initialized = False
        self.mesh_queue = mesh_queue
        self.redis_client = redis_client

    def register_initial_mesh(self, BX: np.ndarray, faces: np.ndarray):
        if not self.initialized:
            self.visual_mesh = ps.register_surface_mesh("Visual Mesh", BX, faces)
            self.mesh_states.append(BX.copy())
            self.step_names.append("Initial State")
            self.initialized = True
            logger.info("Initial mesh registered with Polyscope.")

    def update_mesh(self, BX: np.ndarray, step_name: str):
        if self.visual_mesh:
            self.visual_mesh.update_vertex_positions(BX)
            self.mesh_states.append(BX.copy())
            self.step_names.append(step_name)
            self.current_step_index = len(self.mesh_states) - 1
            logger.info(f"Mesh updated to {step_name}.")

    def reset_to_initial_state(self):
        if self.mesh_states:
            self.current_step_index = 0
            self.visual_mesh.update_vertex_positions(self.mesh_states[0])
            logger.info("Mesh reset to initial state.")

    def process_queue(self):
        while not self.mesh_queue.empty():
            mesh_data = self.mesh_queue.get()
            try:
                BX = np.frombuffer(mesh_data['BX'], dtype=mesh_data['BX_dtype']).reshape(mesh_data['BX_shape'])
                step_name = f"Step {len(self.mesh_states)}"
                self.update_mesh(BX, step_name)
            except Exception as e:
                logger.error(f"Failed to process mesh update: {e}")

    def render_ui(self):
        # Handle Pause and Play buttons
        if imgui.Button("Pause"):
            self.is_paused = True
            self.redis_client.send_command("pause")
            logger.info("Visualization Paused.")
        imgui.SameLine()
        if imgui.Button("Play"):
            self.is_paused = False
            self.redis_client.send_command("play")
            logger.info("Visualization Resumed.")

        # Handle Stop and Start buttons
        if imgui.Button("Stop"):
            self.redis_client.send_command("stop")
            logger.info("Sent stop command to simulation server.")
            self.reset_to_initial_state()
            self.mesh_states = []
            self.step_names = []
            self.is_paused = False
            logger.info("Stopped simulation and reset mesh to initial state.")
        imgui.SameLine()
        if imgui.Button("Start"):
            self.redis_client.send_command("start")
            logger.info("Sent start command to simulation server.")
        imgui.Separator()
        if imgui.Button("Kill Simulation"):
            self.redis_client.send_command("kill")
            logger.info("Sent kill command to simulation server.")

        # Handle Reset to Initial State button
        if imgui.Button("Reset to Initial State"):
            self.reset_to_initial_state()

        # Add UI list to select steps
        if len(self.mesh_states) > 1:
            changed, new_step_index = imgui.Combo("Select Step", self.current_step_index, self.step_names)
            if changed:
                self.current_step_index = new_step_index
                self.visual_mesh.update_vertex_positions(self.mesh_states[self.current_step_index])
                logger.info(f"Switched to {self.step_names[self.current_step_index]}.")

        # Process any pending mesh updates
        self.process_queue()

    def show(self):
        ps.set_user_callback(self.render_ui)
        ps.show()

class ClientApplication:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = RedisClient(host=redis_host, port=redis_port, db=redis_db)
        self.mesh_queue = queue.Queue()
        self.visualizer = PolyscopeVisualizer(self.mesh_queue, self.redis_client)
        self.initial_mesh_loaded = False

    def fetch_initial_mesh(self):
        logging.info("Fetching 'mesh_state' from Redis.")
        serialized_mesh_data = self.redis_client.redis_client.get('mesh_state')
        if serialized_mesh_data:
            logging.info("'mesh_state' fetched successfully from Redis.")
            try:
                mesh_data = self.redis_client.deserialize_mesh_data(serialized_mesh_data.decode('utf-8'))
                x = np.frombuffer(mesh_data['x'], dtype=mesh_data['x_dtype']).reshape(mesh_data['x_shape'])
                BX = np.frombuffer(mesh_data['BX'], dtype=mesh_data['BX_dtype']).reshape(mesh_data['BX_shape'])
                faces = np.frombuffer(mesh_data['faces'], dtype=mesh_data['faces_dtype']).reshape(mesh_data['faces_shape'])
                self.visualizer.register_initial_mesh(BX, faces)
                self.initial_mesh_loaded = True
                logging.info("Initial mesh loaded and visualized.")
            except Exception as e:
                logging.error(f"Failed to deserialize and load initial mesh: {e}")
        else:
            logging.warning("No 'mesh_state' available in Redis.")

    def listen_for_updates(self):
        self.redis_client.listen(['simulation_updates'], self.mesh_queue)

    def run(self):
        self.fetch_initial_mesh()
        if not self.initial_mesh_loaded:
            logging.error("Cannot start visualization without initial mesh.")
        listener_thread = Thread(target=self.listen_for_updates, daemon=True)
        listener_thread.start()
        logging.info("Started listener thread for mesh updates.")
        self.visualizer.show()

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="3D FEM Simulation Client with Redis and Polyscope Visualization.")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis server host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis server port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database number")
    return parser.parse_args()

def main():
    args = parse_arguments()
    client_app = ClientApplication(redis_host=args.redis_host, redis_port=args.redis_port, redis_db=args.redis_db)
    client_app.run()

if __name__ == "__main__":
    main()
