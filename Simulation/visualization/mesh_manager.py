import os
import logging
import meshio
from minio_client import MinioClient

logger = logging.getLogger(__name__)

class MeshManager:
    """
    MeshManager handles saving meshes locally and uploading them to MinIO.
    """
    def __init__(self, minio_client: MinioClient, directory="meshes", bucket_name="mesh-bucket", folder_name="meshes"):
        self.directory = directory
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.folder_name = folder_name
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Mesh directory set to '{self.directory}'.")

    def save_mesh(self, mesh, step_name):
        """
        Saves the mesh locally and uploads it to MinIO.
        """
        filename = os.path.join(
            self.directory, f"mesh_{step_name.replace(' ', '_')}.obj")
        try:
            meshio.write(filename, mesh, file_format="obj")
            logger.info(f"Mesh saved locally to {filename}")
        except Exception as e:
            logger.error(f"Failed to save mesh to {filename}: {e}")
            return

        # Upload to MinIO
        try:
            self.minio_client.upload_file(
                bucket_name=self.bucket_name,
                folder_name=self.folder_name,
                file_path=filename
            )
        except Exception as e:
            logger.error(f"Failed to upload mesh '{filename}' to MinIO: {e}")