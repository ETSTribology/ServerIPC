import os
import logging
import meshio
import polyscope as ps

logger = logging.getLogger(__name__)


class MeshManager:
    def __init__(self, directory="screenshots", location="local"):
        self.directory = directory
        if location == "local":
            self.location = "local"
        elif location == "server":
            self.location = "server"
            
        self.location = location
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Mesh directory set to '{self.directory}'.")

    def save_mesh(self, mesh, step_name):
        filename = os.path.join(
            self.directory, f"mesh_{step_name.replace(' ', '_')}.vtk")
        meshio.write(filename, mesh)
        logger.info(f"Mesh saved to {filename}")
