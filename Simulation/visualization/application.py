import logging
from threading import Thread
import queue
import numpy as np
from visualizer import PolyscopeVisualizer
from redis_client import RedisClient
from minio_client import MinioClient

logger = logging.getLogger(__name__)


class ClientApplication:
    def __init__(
        self,
        redis_host='localhost',
        redis_port=6379,
        redis_db=0,
        minio_endpoint='localhost:9000',
        minio_access_key='minioadmin',
        minio_secret_key='minioadmin',
    ):
        self.redis_client = RedisClient(
            host=redis_host, port=redis_port, db=redis_db)
        self.minio_client = MinioClient(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
        )   
        self.mesh_queue = queue.Queue()
        self.visualizer = PolyscopeVisualizer(
            self.mesh_queue,
            self.redis_client,
        )
        self.initial_mesh_loaded = False

    def fetch_initial_mesh(self):
        serialized_mesh_data = self.redis_client.get_data('mesh_state')
        if serialized_mesh_data:
            mesh_data = self.redis_client.deserialize_mesh_data(
                serialized_mesh_data)
            BX = np.frombuffer(
                mesh_data['BX'], dtype=mesh_data['BX_dtype']
            ).reshape(mesh_data['BX_shape'])
            faces = np.frombuffer(
                mesh_data['faces'], dtype=mesh_data['faces_dtype']
            ).reshape(mesh_data['faces_shape'])
            face_materials = np.frombuffer(
                mesh_data['face_materials'], dtype=mesh_data['face_materials_dtype']
            ).reshape(mesh_data['face_materials_shape'])
            materials = mesh_data['materials']
            self.visualizer.register_initial_mesh(
                BX, faces, face_materials, materials)
            self.initial_mesh_loaded = True
        else:
            logger.warning("No 'mesh_state' available in Redis.")
            # load a fake mesh for testing
            self.visualizer.register_initial_mesh(
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                np.array([[0, 1, 2]]),
                np.array([[0]]),
                ['default']
            )

    def listen_for_updates(self):
        try:
            self.redis_client.listen(['simulation_updates'], self.mesh_queue)
        except Exception as e:
            logger.error(f"Error in listen_for_updates: {e}")

    def run(self):
        self.fetch_initial_mesh()
        if not self.initial_mesh_loaded:
            logger.error("Cannot start visualization without initial mesh.")
            return
        listener_thread = Thread(target=self.listen_for_updates, daemon=True)
        listener_thread.start()
        try:
            self.visualizer.show()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            listener_thread.join()
            logger.info("Listener thread terminated.")
