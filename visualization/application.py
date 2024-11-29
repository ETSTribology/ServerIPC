import argparse
import logging
import json
from threading import Thread
import queue
import numpy as np
from visualizer import PolyscopeVisualizer
from redis_client import RedisClient
from minio_client import MinioClient
from mesh_manager import MeshManager
from screenshot_manager import ScreenshotManager

logger = logging.getLogger(__name__)


class ClientApplication:
    """
    ClientApplication initializes all components and runs the application.
    """
    def __init__(
        self,
        redis_host='localhost',
        redis_port=6379,
        redis_db=0,
        minio_endpoint='localhost:9000',
        minio_access_key='minioadmin',
        minio_secret_key='minioadmin',
        minio_secure=False,
        mesh_bucket='mesh-bucket',
        mesh_folder='meshes',
        screenshot_dir='screenshots',
    ):
        self.redis_client = RedisClient(
            host=redis_host, port=redis_port, db=redis_db)
        self.minio_client = MinioClient(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
        )
        self.mesh_manager = MeshManager(
            minio_client=self.minio_client,
            directory=mesh_folder,  # Changed to mesh_folder for consistency
            bucket_name=mesh_bucket,
            folder_name=mesh_folder,
        )
        self.screenshot_manager = ScreenshotManager(
            directory=screenshot_dir
        )
        self.mesh_queue = queue.Queue()
        self.visualizer = PolyscopeVisualizer(
            self.mesh_queue,
            self.redis_client,
            self.mesh_manager,
            self.screenshot_manager,
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
            # Load a fake mesh for testing
            self.visualizer.register_initial_mesh(
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                np.array([[0, 1, 2]]),
                np.array([[0]]),
                [{'color': [128, 128, 128, 1]}]  # Default material
            )
            self.initial_mesh_loaded = True  # Even fake mesh is loaded

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="3D FEM Simulation Client with Redis and Polyscope Visualization."
    )
    parser.add_argument(
        "--json", type=str, default="config.json", help="Path to JSON configuration file"
    )
    return parser.parse_args()


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise


def main():
    args = parse_arguments()
    config = load_config(args.json)

    # Extract configuration parameters with defaults
    redis_config = config.get('redis', {})
    minio_config = config.get('minio', {})
    mesh_config = config.get('mesh', {})
    screenshot_config = config.get('screenshot', {})

    client_app = ClientApplication(
        redis_host=redis_config.get('host', 'localhost'),
        redis_port=redis_config.get('port', 6379),
        redis_db=redis_config.get('db', 0),
        minio_endpoint=minio_config.get('endpoint', 'localhost:9000'),
        minio_access_key=minio_config.get('access_key', 'minioadmin'),
        minio_secret_key=minio_config.get('secret_key', 'minioadmin'),
        minio_secure=minio_config.get('secure', False),
        mesh_bucket=mesh_config.get('bucket', 'mesh-bucket'),
        mesh_folder=mesh_config.get('folder', 'meshes'),
        screenshot_dir=screenshot_config.get('directory', 'screenshots'),
    )
    client_app.run()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()