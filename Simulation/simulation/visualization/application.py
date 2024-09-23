import logging
from threading import Thread
import queue
from redis_client import RedisClient
from visualizer import PolyscopeVisualizer
import numpy as np

logger = logging.getLogger(__name__)

class ClientApplication:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = RedisClient(host=redis_host, port=redis_port, db=redis_db)
        self.mesh_queue = queue.Queue()
        self.visualizer = PolyscopeVisualizer(self.mesh_queue, self.redis_client)
        self.initial_mesh_loaded = False

    def fetch_initial_mesh(self):
        logging.info("Fetching 'mesh_state' from Redis.")
        try:
            serialized_mesh_data = self.redis_client.get_data('mesh_state')
            if serialized_mesh_data:
                logging.info("'mesh_state' fetched successfully from Redis.")
                mesh_data = self.redis_client.deserialize_mesh_data(serialized_mesh_data)
                x = np.frombuffer(mesh_data['x'], dtype=mesh_data['x_dtype']).reshape(mesh_data['x_shape'])
                BX = np.frombuffer(mesh_data['BX'], dtype=mesh_data['BX_dtype']).reshape(mesh_data['BX_shape'])
                faces = np.frombuffer(mesh_data['faces'], dtype=mesh_data['faces_dtype']).reshape(mesh_data['faces_shape'])
                self.visualizer.register_initial_mesh(BX, faces)
                self.initial_mesh_loaded = True
                logging.info("Initial mesh loaded and visualized.")
            else:
                logging.warning("No 'mesh_state' available in Redis.")
        except Exception as e:
            logging.error(f"Failed to deserialize and load initial mesh: {e}")

    def listen_for_updates(self):
        self.redis_client.listen(['simulation_updates'], self.mesh_queue)

    def run(self):
        self.fetch_initial_mesh()
        if not self.initial_mesh_loaded:
            logging.error("Cannot start visualization without initial mesh.")
            return
        listener_thread = Thread(target=self.listen_for_updates, daemon=True)
        listener_thread.start()
        logging.info("Started listener thread for mesh updates.")
        self.visualizer.show()
